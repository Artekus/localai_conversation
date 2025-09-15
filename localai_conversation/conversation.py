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
from voluptuous_serialize import convert
from homeassistant.helpers import llm, intent


from .const import (
    DOMAIN,
    CONF_URL,
    CONF_MODEL,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_MAX_TOKENS,
    CONF_DEBUG_LOGGING,
    DEFAULT_MODEL,
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
        debug_logging = self.entry.options.get(
            CONF_DEBUG_LOGGING, DEFAULT_DEBUG_LOGGING
        )
        max_function_calls = self.entry.options.get(
            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        )
        function_call_count = 0

        
        try:
            # Get the custom API instance directly from hass.data where it was stored during setup.
            # This is the authoritative way to access our specific API instance.
            api = self.hass.data[DOMAIN][self.entry.entry_id]["api"]

            # Generate the system prompt using our custom API method.
            system_prompt = await api.async_get_api_prompt(llm_context)
            #Log showing system prompt
            _LOGGER.info("Prompt Generated from Custom LLM: %s", system_prompt)
            # Provide the generated prompt and our API to the Home Assistant framework.
            # This populates chat_log with the correct context for this conversation turn.
            await chat_log.async_provide_llm_data(
                llm_context,
                user_llm_hass_api=self.entry.entry_id,
                user_llm_prompt=system_prompt,
            )
        except (ConverseError, KeyError) as err:
            _LOGGER.error("Error preparing LLM data: %s", err)
            converse_error = err if isinstance(err, ConverseError) else ConverseError(f"API instance not found: {err}")
            return converse_error.as_conversation_result()


        # Build the message history from the chat log.
        messages = []
        for entry in chat_log.content:
            role = entry.role
            if role == "tool_result":
                role = "tool"

            content = entry.content
            if not isinstance(content, str):
                content = json.dumps(content)

            messages.append({"role": role, "content": content})

        # Line 96 was previously here


        tools = []
        if chat_log.llm_api:
            tools = chat_log.llm_api.tools

        try:
            # Initial query to LocalAI            
            while function_call_count < max_function_calls:
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

                    tool = next((t for t in tools if t.name == tool_name), None)
                    tool_response_content = ""

                    if tool:
                        # On success, we summarize and exit.
                        # On error, we append the error and continue the loop to let the AI try again.
                        try:
                            cleaned_args = {k: v for k, v in tool_args.items() if v}
                            #if tool_name in ("HassTurnOn", "HassTurnOff") and cleaned_args.get("domain") == "light":
                            #    cleaned_args.pop("device_class", None)
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
                            
                            # This is the success path
                            tool_response = await tool.async_call(
                                self.hass,
                                llm.ToolInput(tool_name, cleaned_args),
                                llm_context,
                            )
                            tool_response_content = json.dumps(tool_response)

                        # This is the error path
                        except Exception as e:
                            _LOGGER.error("Error calling tool %s: %s", tool_name, e)
                            error_message = str(e)
                            if "cannot target all devices" in error_message:
                                tool_response_content = f"Error: The tool '{tool_name}' requires a specific target. You must provide a name, area, or entity ID. Do not try to control all devices at once."
                            else:
                                tool_response_content = f"Error: {e}. The description for tool '{tool_name}' is: {tool.description}"
                            
                            # Append error and let the while loop re-query the AI
                            messages.append({"role": "tool", "tool_call_id": tool_call.get("id", ""), "name": tool_name, "content": tool_response_content})
                            continue

                    else:
                        _LOGGER.warning("Tool %s not found", tool_name)
                        tool_response_content = f"Error: Tool '{tool_name}' not found."
                        messages.append({"role": "tool", "tool_call_id": tool_call.get("id", ""), "name": tool_name, "content": tool_response_content})
                        continue

                    # If we reach here, the tool call was successful.
                    # Append the successful result, get a summary, and exit.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", ""),
                        "name": tool_name,
                        "content": tool_response_content,
                    })
                    summary_response = await self._query_localai(messages, [], debug_logging)
                    final_response_content = summary_response["choices"][0]["message"].get("content")
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_speech(final_response_content)
                    return ConversationResult(
                        response=intent_response,
                        conversation_id=user_input.conversation_id,
                    )
                
            # This is reached if the AI's response did not contain a tool call.
            final_response_content = messages[-1].get("content") or "Sorry, I'm not sure how to respond to that."
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
        parameters: dict[str, Any]
        if tool.parameters is None:
            parameters = {}
        # Some built-in tools use a generic object for empty parameters
        # Some built-in tools use a generic object for empty parameters instead of a Schema.
        # We check for this case to prevent noisy debug logs and handle it gracefully.
        elif isinstance(tool.parameters, object) and not hasattr(tool.parameters, 'schema'):
            _LOGGER.debug("Tool %s has generic object parameters, treating as empty.", tool.name)
            parameters = {}
        else:
            try:
                parameters = convert(
                    tool.parameters, custom_serializer=llm.selector_serializer
                )
            except Exception as e:
                _LOGGER.debug("Could not convert parameters for tool %s: %s", tool.name, e)
                parameters = {}

        if not isinstance(parameters, dict):
            _LOGGER.debug("Parameters for tool %s is not a dict: %s", tool.name, parameters)
            parameters = {}

        if "properties" not in parameters:
            parameters["properties"] = {}
        parameters.setdefault("type", "object")

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            },
        }
