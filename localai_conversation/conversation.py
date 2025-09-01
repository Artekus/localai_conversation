from __future__ import annotations

import json
import logging
from typing import Any, Literal

from homeassistant.components.conversation import (
    ConversationInput,
    ConversationResult,
    ConversationEntity,
    ConversationEntityFeature,
    ChatLog,
    ConverseError,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers import llm, intent
from voluptuous_openapi import convert


from .const import (
    DOMAIN,
    CONF_URL,
    CONF_MODEL,
    CONF_SYSTEM_PROMPT,
    CONF_TOOL_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_MAX_TOKENS,
    CONF_DEBUG_LOGGING,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TOOL_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_TOKENS,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_DEBUG_LOGGING,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = LocalAIAgent(hass, config_entry)
    async_add_entities([agent])


class LocalAIAgent(ConversationEntity):
    """LocalAI conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry

    @property
    def supported_features(self) -> ConversationEntityFeature:
        """Return the supported features of the agent."""
        return ConversationEntityFeature.CONTROL

    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return self.entry.title

    @property
    def unique_id(self) -> str:
        """Return the unique ID of the agent."""
        return self.entry.entry_id

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return "*"

    async def _async_handle_message(
        self, user_input: ConversationInput, chat_log: ChatLog
    ) -> ConversationResult:
        """Handle a conversation turn."""
        llm_context = user_input.as_llm_context(DOMAIN)
        system_prompt = self.entry.options.get(
            CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT
        )
        tool_prompt = self.entry.options.get(CONF_TOOL_PROMPT, DEFAULT_TOOL_PROMPT)
        debug_logging = self.entry.options.get(
            CONF_DEBUG_LOGGING, DEFAULT_DEBUG_LOGGING
        )
        max_function_calls = self.entry.options.get(
            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        )
        function_call_count = 0

        try:
            await chat_log.async_provide_llm_data(
                llm_context,
                user_llm_hass_api="localai_conversation",
                user_llm_prompt=system_prompt,
                user_extra_system_prompt=tool_prompt,
            )
        except ConverseError as err:
            return err.as_conversation_result()

        messages = []
        for entry in chat_log.content:
            role = entry.role
            if role == "tool_result":
                role = "tool"

            content = entry.content
            if not isinstance(content, str):
                content = json.dumps(content)

            messages.append({"role": role, "content": content})

        tools = []
        if chat_log.llm_api:
            tools = chat_log.llm_api.tools

        try:
            while function_call_count < max_function_calls:
                # Initial query to LocalAI
                response = await self._query_localai(messages, tools, debug_logging)
                response_message = response["choices"][0]["message"]
                messages.append(response_message)

                tool_calls = response_message.get("tool_calls")
                content_str = response_message.get("content")

                # Models sometimes return tool calls as a JSON string in the content field
                if not tool_calls and isinstance(content_str, str):
                    try:
                        json_start_index = content_str.find("[")
                        if json_start_index != -1:
                            json_str = content_str[json_start_index:]
                            parsed_json = json.loads(json_str)
                            if isinstance(parsed_json, list) and all(
                                "function" in item for item in parsed_json
                            ):
                                tool_calls = parsed_json
                    except (json.JSONDecodeError, TypeError):
                        _LOGGER.debug(
                            "Content is not a valid tool call JSON: %s", content_str
                        )
                        tool_calls = None

                # Check for tool calls and execute them
                if not tool_calls:
                    break

                for tool_call in tool_calls:
                    function_call_count += 1
                    if "function" not in tool_call:
                        continue

                    tool_name = tool_call["function"]["name"]
                    raw_args = tool_call["function"]["arguments"]

                    if isinstance(raw_args, str):
                        try:
                            tool_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            _LOGGER.error("Failed to parse tool arguments: %s", raw_args)
                            continue
                    else:
                        tool_args = raw_args
                    
                    # Clean up empty arguments
                    tool_args = {k: v for k, v in tool_args.items() if v}


                    tool = next((t for t in tools if t.name == tool_name), None)
                    tool_response_content = ""

                    if tool:
                        try:
                            # Defensively clean arguments to only include what the tool schema expects.
                            # This prevents the AI from passing invalid extra parameters.
                            cleaned_args = {}
                            for key in tool.parameters.schema:
                                key_str = str(key)
                                if key_str in tool_args:
                                    cleaned_args[key_str] = tool_args[key_str]
                            
                            if tool_name == "HassLightSet":
                                if "brightness" in cleaned_args and "color" not in cleaned_args:
                                    cleaned_args.pop("color", None)
                                    cleaned_args.pop("temperature", None)
                                elif "color" in cleaned_args:
                                    cleaned_args.pop("brightness", None)
                                    cleaned_args.pop("temperature", None)
                                elif "temperature" in cleaned_args:
                                    cleaned_args.pop("brightness", None)
                                    cleaned_args.pop("color", None)

                            tool_response = await tool.async_call(
                                self.hass,
                                llm.ToolInput(tool_name, cleaned_args),
                                llm_context,
                            )
                            tool_response_content = json.dumps(tool_response)

                            # After a successful tool call, get a summary from the AI
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.get("id", ""),
                                "name": tool_name,
                                "content": tool_response_content,
                            })
                            # Make one final call to the AI for a summary, but do not provide any tools
                            summary_response = await self._query_localai(messages, [], debug_logging)
                            final_response_content = summary_response["choices"][0]["message"].get("content")
                            intent_response = intent.IntentResponse(language=user_input.language)
                            intent_response.async_set_speech(final_response_content)
                            return ConversationResult(
                                response=intent_response,
                                conversation_id=user_input.conversation_id,
                            )

                        except Exception as e:
                            _LOGGER.error("Error calling tool %s: %s", tool_name, e)
                            tool_response_content = f"Error: {e}. The description for tool '{tool_name}' is: {tool.description}"
                    else:
                        _LOGGER.warning("Tool %s not found", tool_name)
                        tool_response_content = f"Error: Tool '{tool_name}' not found."

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", ""),
                            "name": tool_name,
                            "content": tool_response_content,
                        }
                    )
                # This line is removed to allow the loop to continue with the full tool list on the next iteration
                # tools = []

            final_response_content = messages[-1].get("content", "Sorry, I'm not sure how to respond to that.")
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_speech(final_response_content)
            return ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
            )

        except Exception as e:
            _LOGGER.error("Error querying LocalAI: %s", e)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Error communicating with LocalAI: {e}",
            )
            return ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
            )

    async def _query_localai(
        self, messages: list[dict], tools: list[llm.Tool], debug_logging: bool
    ) -> dict[str, Any]:
        """Query the LocalAI API."""
        session = async_get_clientsession(self.hass)
        url = self.entry.data.get(CONF_URL)
        model = self.entry.options.get(CONF_MODEL, DEFAULT_MODEL)
        temperature = self.entry.options.get(
            CONF_TEMPERATURE, DEFAULT_TEMPERATURE
        )
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)

        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = [self._tool_to_openai_tool(tool) for tool in tools]
        
        if debug_logging:
            _LOGGER.info("LocalAI Request: %s", json.dumps(payload, indent=2))

        async with session.post(
            f"{url}/v1/chat/completions", headers=headers, json=payload
        ) as response:
            response.raise_for_status()
            response_json = await response.json()
            if debug_logging:
                _LOGGER.info("LocalAI Response: %s", json.dumps(response_json, indent=2))
            return response_json

    def _tool_to_openai_tool(self, tool: llm.Tool) -> dict[str, Any]:
        """Convert a tool to the OpenAI tool format."""
        try:
            parameters = convert(
                tool.parameters, custom_serializer=llm.selector_serializer
            )
            if "type" not in parameters:
                parameters["type"] = "object"
            if "properties" not in parameters:
                parameters["properties"] = {}
        except Exception as e:
            _LOGGER.warning(
                "Could not convert parameters for tool %s: %s", tool.name, e
            )
            parameters = {"type": "object", "properties": {}}

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            },
        }

