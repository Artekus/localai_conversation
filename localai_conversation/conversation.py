from __future__ import annotations

import json
import logging, asyncio, types
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, TypedDict, cast

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
import voluptuous as vol
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Generator
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


class LocalAIAgent(conversation.ConversationEntity):
    """LocalAI conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry

    @property
    def supported_features(self) -> conversation.ConversationEntityFeature:
        """Return the supported features of the agent."""
        # Do not remove this comment. return ConversationEntityFeature.CONTROL | ConversationEntityFeature.STREAM
        return conversation.ConversationEntityFeature.CONTROL

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
        self, user_input: conversation.ConversationInput, chat_log: conversation.ChatLog
    ) -> conversation.ConversationResult:
        """Handle a conversation turn."""
        llm_context = user_input.as_llm_context(DOMAIN)
        debug_logging = self.entry.options.get(
            CONF_DEBUG_LOGGING, DEFAULT_DEBUG_LOGGING
        )
        
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

            # MONKEY-PATCH: Replace the default stream handler with our custom one.
            chat_log.async_add_delta_content_stream = types.MethodType(self._custom_async_add_delta_content_stream.__func__, chat_log)

            # We must call async_provide_llm_data to register our API and get tools.
            # However, it will create a system prompt that we don't want.
            await chat_log.async_provide_llm_data(
                llm_context,
                user_llm_hass_api=self.entry.entry_id,
                user_llm_prompt="",
            )

            # Immediately overwrite the incorrect system prompt with our own.
            # This makes the chat_log the single source of truth for the conversation.
            chat_log.content[0] = conversation.SystemContent(content=system_prompt)

        except (conversation.ConverseError, KeyError) as err:
            _LOGGER.error("Error preparing LLM data: %s", err)
            converse_error = err if isinstance(err, conversation.ConverseError) else conversation.ConverseError(f"API instance not found: {err}")
            return converse_error.as_conversation_result()

        max_function_calls = self.entry.options.get(
            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        )
        function_call_count = 0

        try:
            while function_call_count < max_function_calls:
                # Build the message history from the chat_log for each turn.
                messages = []
                for m in chat_log.content:
                    message: dict[str, Any] = {"role": m.role}
                    
                    if m.role == "system":
                        message["content"] = m.content
                    elif m.role == "user":
                        message["content"] = m.content
                    elif m.role == "assistant":
                        if m.content:
                            message["content"] = m.content
                        if m.tool_calls:
                            message["tool_calls"] = [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.tool_name,
                                        "arguments": json.dumps(tc.tool_args),
                                    },
                                }
                                for tc in m.tool_calls
                            ]
                    elif m.role == "tool_result":
                        message["role"] = "tool"
                        message["tool_call_id"] = m.tool_call_id
                        message["content"] = json.dumps(m.tool_result)

                    if "content" in message or "tool_calls" in message:
                        messages.append(message)

                # With our monkey-patch in place, we can now use a simple, clean streaming approach.
                tools = chat_log.llm_api.tools if chat_log.llm_api else []
                stream = self._query_localai_streaming(messages, tools, debug_logging)

                # Process the stream and tool calls
                async for _ in chat_log.async_add_delta_content_stream(
                    agent_id=self.unique_id,
                    stream=stream,
                ):
                    pass

                last_message = chat_log.content[-1]

                if not isinstance(last_message, conversation.ToolResultContent):
                    # The LLM didn't call a tool, so it must be a final answer.
                    break
                
                if debug_logging:
                    _LOGGER.info("Tool executed, looping for summarization.")

                # A tool was called and executed. Loop again for summarization.
                function_call_count += 1

            # 3. The chat log is now complete. The framework has handled the full
            # conversation, including the final summary from the model.
            final_message = chat_log.content[-1] if chat_log.content else None
            if final_message and final_message.role == "assistant":
                final_response_content = final_message.content
            else:
                final_response_content = "Sorry, I was unable to process that."

            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_speech(final_response_content)
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
            )

        except Exception as e:
            _LOGGER.error("Error querying LocalAI: %s", e, exc_info=e)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Error communicating with LocalAI: {e}",
            )
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
            )

    async def _custom_async_add_delta_content_stream(
        self,
        agent_id: str,
        stream: AsyncIterable[
            conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
        ],
    ) -> AsyncGenerator[conversation.AssistantContent | conversation.ToolResultContent, None]:
        """
        Custom stream handler that reflects the original Home Assistant logic
        but is modified to handle LocalAI's specific tool-calling stream format.
        'self' here is the chat_log instance due to monkey-patching.
        """
        chat_log = self

        # Retrieve the debug_logging setting for this agent instance.
        entry = chat_log.hass.config_entries.async_get_entry(agent_id)
        debug_logging = entry.options.get(
            CONF_DEBUG_LOGGING, DEFAULT_DEBUG_LOGGING
        )        
        if debug_logging:
            _LOGGER.info("Async Delta Content Stream is called")
            
        current_content = ""
        current_thinking_content = ""
        current_native: Any = None
        current_tool_calls: list[llm.ToolInput] = []
        tool_call_builders: dict[int, dict[str, Any]] = {}
        tool_call_tasks: dict[str, asyncio.Task] = {}
        message_started = False

        async for delta in stream:
            _LOGGER.debug("Async Delta Content Stream: Received delta: %s", delta)

            # Indicates update to current message
            if "role" not in delta or (
                "tool_calls" in delta
                and delta["tool_calls"]
                and delta["tool_calls"][0].get("function", {}).get("arguments")
            ):
                # ToolResultContentDeltaDict will always have a role
                assistant_delta = cast(conversation.AssistantContentDeltaDict, delta)
                if debug_logging:
                    _LOGGER.info("Async Delta Content Stream: No Role or second part of Tool Call recieved")
                    _LOGGER.info("Async Delta Content Stream: assistant_delta: %s", assistant_delta)
                if delta_content := assistant_delta.get("content"):
                    current_content += delta_content
                if delta_thinking_content := assistant_delta.get("thinking_content"):
                    current_thinking_content += delta_thinking_content
                if delta_native := assistant_delta.get("native"):
                    if current_native is not None:
                        raise RuntimeError(
                            "Native content already set, cannot overwrite"
                        )
                    current_native = delta_native
                if delta_tool_calls := assistant_delta.get("tool_calls"):
                    if debug_logging:
                        _LOGGER.info("Async Delta Content Stream: Pre current_tool_calls: %s", current_tool_calls)
                        _LOGGER.info("Async Delta Content Stream: Pre delta_tool_calls: %s", delta_tool_calls)                  
                    current_tool_calls[-1]['function']['arguments'] = delta_tool_calls[0]['function']['arguments']
                    delta_tool_calls[0] = current_tool_calls[-1]
                    delta_tool_calls = [
                        llm.ToolInput(
                            id=tc.get("id"),
                            tool_name=tc.get("function", {}).get("name"),
                            tool_args=json.loads(tc.get("function", {}).get("arguments", "{}")),
                        )
                        for tc in delta_tool_calls
                    ]
                    current_tool_calls[-1]=delta_tool_calls[0]
                    if debug_logging:
                        _LOGGER.info("Async Delta Content Stream: Post current_tool_calls: %s", current_tool_calls)
                        _LOGGER.info("Async Delta Content Stream: Post delta_tool_calls: %s", delta_tool_calls)   
                    # Start processing the tool calls as soon as we know about them
                    for tool_call in delta_tool_calls:
                        if not tool_call.external:
                            if chat_log.llm_api is None:
                                raise ValueError("No LLM API configured")

                            tool_call_tasks[tool_call.id] = chat_log.hass.async_create_task(
                                chat_log.llm_api.async_call_tool(tool_call),
                                name=f"llm_tool_{tool_call.id}",
                            )
                if debug_logging:
                    _LOGGER.info("Async Delta Content Stream: Outside if delta_tool_calls: %s", delta_tool_calls)                   
                if chat_log.delta_listener:
                    if filtered_delta := {
                        k: v for k, v in assistant_delta.items() if k != "native"
                    }:
                        # We do not want to send the native content to the listener
                        # as it is not JSON serializable
                        chat_log.delta_listener(self, filtered_delta)
                continue

            # Starting a new message
            # Yield the previous message if it has content
            if (
                current_content
                or current_thinking_content
                or current_tool_calls
                or current_native
            ):
                content: conversation.AssistantContent | conversation.ToolResultContent = conversation.AssistantContent(
                    agent_id=agent_id,
                    content=current_content or None,
                    thinking_content=current_thinking_content or None,
                    tool_calls=current_tool_calls or None,
                    native=current_native,
                )
                yield content
                async for tool_result in chat_log.async_add_assistant_content(
                    content, tool_call_tasks=tool_call_tasks
                ):
                    yield tool_result
                    if chat_log.delta_listener:
                        chat_log.delta_listener(self, asdict(tool_result))
                current_content = ""
                current_thinking_content = ""
                current_native = None
                current_tool_calls = []

            if delta["role"] == "assistant":
                current_content = delta.get("content") or ""
                current_thinking_content = delta.get("thinking_content") or ""
                current_tool_calls = delta.get("tool_calls") or []
                current_native = delta.get("native")
                if debug_logging:
                    _LOGGER.info("Async Delta Content Stream: Role Recieved: %s", delta["role"])

                if chat_log.delta_listener:
                    if filtered_delta := {
                        k: v for k, v in delta.items() if k != "native"
                    }:
                        chat_log.delta_listener(self, filtered_delta)
            elif delta["role"] == "tool_result":
                content = conversation.ToolResultContent(
                    agent_id=agent_id,
                    tool_call_id=delta["tool_call_id"],
                    tool_name=delta["tool_name"],
                    tool_result=delta["tool_result"],
                )
                yield content
                if chat_log.delta_listener:
                    chat_log.delta_listener(self, asdict(content))
                chat_log.async_add_assistant_content_without_tools(content)
            else:
                raise ValueError(
                    "Only assistant and tool_result roles expected."
                    f" Got {delta['role']}"
                )

        if (
            current_content
            or current_thinking_content
            or current_tool_calls
            or current_native
        ):
            content = conversation.AssistantContent(
                agent_id=agent_id,
                content=current_content or None,
                thinking_content=current_thinking_content or None,
                tool_calls=current_tool_calls or None,
                native=current_native,
            )
            yield content
            async for tool_result in chat_log.async_add_assistant_content(
                content, tool_call_tasks=tool_call_tasks
            ):
                yield tool_result
                if chat_log.delta_listener:
                    chat_log.delta_listener(self, asdict(tool_result))


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
            _LOGGER.info("Query Local AI Streaming: LocalAI Streaming Request: %s", json.dumps(payload, indent=2))
 
        async with session.post(
            f"{url}/v1/chat/completions", headers=headers, json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.content:
                line_str = line.decode("utf-8").strip()
                if not line_str or not line_str.startswith("data:"):
                    continue
                if line_str == "data: [DONE]":
                    break
                
                line_json = line_str[len("data: "):]
                try:
                    yield json.loads(line_json)
                except json.JSONDecodeError:
                    _LOGGER.warning("Received invalid JSON in stream: %s", line_json)

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
