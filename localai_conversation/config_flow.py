"""Config flow for LocalAI Conversation."""
from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigFlow, OptionsFlow, ConfigEntry
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    CONF_URL,
    CONF_MODEL,
    CONF_SYSTEM_PROMPT,
    CONF_TOOL_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_MAX_TOKENS,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_DEBUG_LOGGING,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TOOL_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_DEBUG_LOGGING,
    CONF_BASE_INSTRUCTIONS,
    DEFAULT_BASE_INSTRUCTIONS,
    CONF_AREA_AWARE_PROMPT,
    DEFAULT_AREA_AWARE_PROMPT,
    CONF_NO_AREA_PROMPT,
    DEFAULT_NO_AREA_PROMPT,
    CONF_TIMER_UNSUPPORTED_PROMPT,
    DEFAULT_TIMER_UNSUPPORTED_PROMPT,
    CONF_DYNAMIC_CONTEXT_PROMPT,
    DEFAULT_DYNAMIC_CONTEXT_PROMPT,
)

class LocalAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for LocalAI Conversation."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> dict[str, Any]:
        """Handle the initial step."""
        if user_input is not None:
            return self.async_create_entry(title="LocalAI", data=user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required(CONF_URL): str,
            }),
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Get the options flow for this handler."""
        return LocalAIOptionsFlowHandler(config_entry)


class LocalAIOptionsFlowHandler(OptionsFlow):
    """Handle an options flow for LocalAI Conversation."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> dict[str, Any]:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Optional(CONF_MODEL, default=self.config_entry.options.get(CONF_MODEL, DEFAULT_MODEL)): str,
                vol.Optional(CONF_SYSTEM_PROMPT, default=self.config_entry.options.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT)): selector.TemplateSelector(),
                vol.Optional(CONF_TOOL_PROMPT, default=self.config_entry.options.get(CONF_TOOL_PROMPT, DEFAULT_TOOL_PROMPT)): selector.TemplateSelector(),
                vol.Optional(CONF_BASE_INSTRUCTIONS, default=self.config_entry.options.get(CONF_BASE_INSTRUCTIONS, DEFAULT_BASE_INSTRUCTIONS)): selector.TemplateSelector(),
                vol.Optional(CONF_AREA_AWARE_PROMPT, default=self.config_entry.options.get(CONF_AREA_AWARE_PROMPT, DEFAULT_AREA_AWARE_PROMPT)): selector.TemplateSelector(),
                vol.Optional(CONF_NO_AREA_PROMPT, default=self.config_entry.options.get(CONF_NO_AREA_PROMPT, DEFAULT_NO_AREA_PROMPT)): selector.TemplateSelector(),
                vol.Optional(CONF_TIMER_UNSUPPORTED_PROMPT, default=self.config_entry.options.get(CONF_TIMER_UNSUPPORTED_PROMPT, DEFAULT_TIMER_UNSUPPORTED_PROMPT)): selector.TemplateSelector(),
                vol.Optional(CONF_DYNAMIC_CONTEXT_PROMPT, default=self.config_entry.options.get(CONF_DYNAMIC_CONTEXT_PROMPT, DEFAULT_DYNAMIC_CONTEXT_PROMPT)): selector.TemplateSelector(),
                vol.Optional(CONF_TEMPERATURE, default=self.config_entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)): vol.All(vol.Coerce(float), vol.Range(min=0, max=2)),
                vol.Optional(CONF_TOP_P, default=self.config_entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
                vol.Optional(CONF_MAX_TOKENS, default=self.config_entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)): int,
                vol.Optional(CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION, default=self.config_entry.options.get(CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION, DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION)): int,
                vol.Optional(CONF_DEBUG_LOGGING, default=self.config_entry.options.get(CONF_DEBUG_LOGGING, DEFAULT_DEBUG_LOGGING)): bool,
            }),
        )
