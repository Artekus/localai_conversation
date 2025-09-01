"""Config flow for LocalAI Conversation."""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers.aiohttp_client import async_get_clientsession
import homeassistant.helpers.config_validation as cv
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
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required("name", default="LocalAI"): str,
        vol.Required(CONF_URL): str,
    }
)


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Validate the user input allows us to connect."""
    session = async_get_clientsession(hass)
    try:
        # We check the /v1/models endpoint as a simple way to validate the URL
        async with session.get(f"{data[CONF_URL]}/v1/models") as response:
            if response.status != 200:
                raise ConnectionError
    except Exception as e:
        _LOGGER.error("Could not connect to LocalAI at %s: %s", data[CONF_URL], e)
        raise ConnectionError from e

    return {"title": data["name"]}


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle an options flow for LocalAI Conversation."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_MODEL,
                    default=self.config_entry.options.get(CONF_MODEL, DEFAULT_MODEL),
                ): str,
                vol.Optional(
                    CONF_SYSTEM_PROMPT,
                    default=self.config_entry.options.get(
                        CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT
                    ),
                ): selector.TextSelector(selector.TextSelectorConfig(multiline=True)),
                vol.Optional(
                    CONF_TOOL_PROMPT,
                    default=self.config_entry.options.get(
                        CONF_TOOL_PROMPT, DEFAULT_TOOL_PROMPT
                    ),
                ): selector.TextSelector(selector.TextSelectorConfig(multiline=True)),
                vol.Optional(
                    CONF_TEMPERATURE,
                    default=self.config_entry.options.get(
                        CONF_TEMPERATURE, DEFAULT_TEMPERATURE
                    ),
                ): cv.positive_float,
                vol.Optional(
                    CONF_TOP_P,
                    default=self.config_entry.options.get(CONF_TOP_P, DEFAULT_TOP_P),
                ): cv.positive_float,
                vol.Optional(
                    CONF_MAX_TOKENS,
                    default=self.config_entry.options.get(
                        CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS
                    ),
                ): cv.positive_int,
                vol.Optional(
                    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
                    default=self.config_entry.options.get(
                        CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
                        DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
                    ),
                ): cv.positive_int,
                vol.Optional(
                    CONF_DEBUG_LOGGING,
                    default=self.config_entry.options.get(
                        CONF_DEBUG_LOGGING, DEFAULT_DEBUG_LOGGING
                    ),
                ): bool,
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema)


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for LocalAI Conversation."""

    VERSION = 1

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> OptionsFlowHandler:
        """Get the options flow for this handler."""
        return OptionsFlowHandler(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                info = await validate_input(self.hass, user_input)
            except ConnectionError:
                errors["base"] = "cannot_connect"
            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                return self.async_create_entry(title=info["title"], data=user_input)

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )
