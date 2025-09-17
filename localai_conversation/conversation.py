from __future__ import annotations

import json
import logging, asyncio
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
import voluptuous as vol
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
        # Do not remove this comment. return ConversationEntityFeature.CONTROL | ConversationEntityFeature.STREAM
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
        #function_call_count = 0

        
        try:
            # Get the custom API instance directly from hass.data where it was stored during setup.
            # This is the authoritative way to access our specific API instance.
            api = self.hass.data[DOMAIN][self.entry.entry_id]["api"]

            # Generate the system prompt using our custom API method.
            system_prompt = await api.async_get_api_prompt(llm_context)
            #Log showing system prompt
            if debug_logging:
                _LOGGER.info("Prompt Generated from Custom LLM: %s", system_prompt)
            # Provide our API to the Home Assistant framework. By not passing
            # user_llm_prompt, we prevent it from building a duplicate prompt.
            # We only call this to get the tools from our API instance.
            # Gemini deleted user_llm_prompt=system_prompt, from the function below.
            await chat_log.async_provide_llm_data(
                llm_context,
                user_llm_hass_api=self.entry.entry_id,
            )
        except (ConverseError, KeyError) as err:
            _LOGGER.error("Error preparing LLM data: %s", err)
            converse_error = err if isinstance(err, ConverseError) else ConverseError(f"API instance not found: {err}")
            return converse_error.as_conversation_result()


        # Build the message history from the chat log, filtering out any system prompts
        # that may have been added by the framework. This ensures we have a clean slate.
        messages = []
        for m in chat_log.content:
            if m.role == "system":
                continue

            message: dict[str, Any] = {"role": m.role}
            
            if m.role == "user":
                message["content"] = m.content
            elif m.role == "assistant":
                # Assistant messages can have both content and tool_calls
                if m.content:
                    message["content"] = m.content
                if m.tool_calls:
                    message["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.tool_name, "arguments": json.dumps(tc.tool_args)},
                        }
                        for tc in m.tool_calls
                    ]
            elif m.role == "tool_result":
                message["role"] = "tool"
                message["tool_call_id"] = m.tool_call_id
                message["content"] = json.dumps(m.tool_result)

            # Only add the message if it has content or tool_calls
            if "content" in message or "tool_calls" in message:
                messages.append(message)

        # Gemini added this and I think it won't
        # Manually insert our authoritative system prompt at the beginning of the conversation.
        messages.insert(0, {"role": "system", "content": system_prompt})


        tools = []
        if chat_log.llm_api:
            tools = chat_log.llm_api.tools

        try:
            # 1. _query_localai_streaming returns an async generator (the stream)
            stream = self._query_localai_streaming(messages, tools, debug_logging)

            # 2. Hand the stream over to Home Assistant's orchestrator.
            # This function will handle tool calls and update the chat log automatically.
            async for _ in chat_log.async_add_delta_content_stream(
                agent_id=self.unique_id,
                stream=stream,
            ):
                # This loop consumes the stream and any resulting tool calls.
                # The 'pass' is intentional; the framework handles the logic.
                pass

            # 3. The chat log is now complete. The framework has handled the full
            # conversation, including the final summary from the model.
            final_message = chat_log.content[-1] if chat_log.content else None
            if final_message and final_message.role == "assistant":
                final_response_content = final_message.content
            else:
                final_response_content = "Sorry, I was unable to process that."

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

    async def _query_localai_streaming(
        self, messages: list[dict], tools: list[llm.Tool], debug_logging: bool
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Query the LocalAI API with streaming and reassemble tool calls."""
        session = async_get_clientsession(self.hass)
        url = self.entry.data.get(CONF_URL)
        model = self.entry.options.get(CONF_MODEL, DEFAULT_MODEL)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
 
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = [self._tool_to_openai_tool(tool) for tool in tools]
 
        if debug_logging:
            _LOGGER.info("LocalAI Streaming Request: %s", json.dumps(payload, indent=2))
 
        # State for reassembling tool calls from multiple deltas
        tool_call_builders: dict[int, dict[str, Any]] = {}
        first_delta_yielded = False
 
        async with session.post(
            f"{url}/v1/chat/completions", headers=headers, json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.content:
                if not line.strip() or line.strip() == b"data: [DONE]":
                    continue
 
                line_str = line.decode('utf-8').replace("data: ", "")
                try:
                    response_json = json.loads(line_str)
                    delta = response_json.get("choices", [{}])[0].get("delta", {})
 
                    # The HA framework doesn't merge partial tool call deltas.
                    # We must reassemble them here and yield them at the end.
                    if raw_tool_calls := delta.pop("tool_calls", None):
                        for raw_call in raw_tool_calls:
                            index = raw_call.get("index", 0)
                            if index not in tool_call_builders:
                                tool_call_builders[index] = {"id": None, "type": "function", "function": {"name": None, "arguments": ""}}
                            
                            builder = tool_call_builders[index]
                            if raw_call.get("id"):
                                builder["id"] = raw_call["id"]
                            if func := raw_call.get("function"):
                                if func.get("name"):
                                    builder["function"]["name"] = func["name"]
                                if func.get("arguments"):
                                    builder["function"]["arguments"] += func["arguments"]
                    
                    # Yield any other content (like text or the initial role)
                    if delta:
                        # To prevent chat_log from creating multiple messages,
                        # only the first delta should contain the 'role'.
                        if first_delta_yielded:
                            delta.pop("role", None)
                        
                        if delta:
                            yield delta
                            if "role" in delta:
                                first_delta_yielded = True
 
                except (json.JSONDecodeError, KeyError):
                    _LOGGER.debug("Skipping non-JSON line in stream: %s", line_str)
 
            # After the stream, finalize and yield the reassembled tool calls
            if tool_call_builders:
                final_tool_inputs = []
                for _index, builder in sorted(tool_call_builders.items()):
                            try:
                                tool_args = json.loads(builder["function"]["arguments"])
                            except json.JSONDecodeError:
                                _LOGGER.warning("Failed to parse tool arguments: %s", builder["function"]["arguments"])
                                tool_args = {}
                            
                            final_tool_inputs.append(llm.ToolInput(
                                id=builder["id"],
                                tool_name=builder["function"]["name"],
                                tool_args=tool_args,
                            ))
                
                final_delta = {"tool_calls": final_tool_inputs}
                # If no other delta was sent (e.g., a response with only tool calls),
                # we need to add the role to this final delta.
                if not first_delta_yielded:
                    final_delta["role"] = "assistant"
                
                yield final_delta

    def _tool_to_openai_tool(self, tool: llm.Tool) -> dict[str, Any]:
        """Convert a tool to the OpenAI tool format."""
        # Manually convert voluptuous schema to JSON schema for OpenAI compatibility.
        # The voluptuous_serialize.convert function does not produce the correct format.
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        if isinstance(tool.parameters, vol.Schema):
            for key, value in tool.parameters.schema.items():
                prop_name = str(key.schema) if hasattr(key, 'schema') else str(key)
                is_required = isinstance(key, vol.Required)

                if is_required:
                    parameters["required"].append(prop_name)

                # Basic type mapping from voluptuous to JSON schema
                param_type = "string"  # Default to string
                if value in (int, vol.Coerce(int)):
                    param_type = "integer"
                elif value in (float, vol.Coerce(float)):
                    param_type = "number"
                elif value == bool:
                    param_type = "boolean"

                param_info = {"type": param_type}
                if hasattr(key, 'description') and key.description:
                    param_info["description"] = key.description

                parameters["properties"][prop_name] = param_info

        # Clean up the schema for final output
        if not parameters["properties"]:
            parameters = {"type": "object", "properties": {}}
        elif not parameters["required"]:
            del parameters["required"]

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            },
        }
