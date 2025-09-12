"""LLM API for LocalAI."""
from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from typing import Any
import logging

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
import voluptuous as vol
from homeassistant.util.json import JsonObjectType
from homeassistant.util import yaml as yaml_util
from homeassistant.components.homeassistant.exposed_entities import (
    async_should_expose,
)
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import area_registry as ar, floor_registry as fr, device_registry as dr, intent

from .const import (
    CONF_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    CONF_TOOL_PROMPT,
    DEFAULT_TOOL_PROMPT,
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
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

class CustomGetLiveContextTool(llm.Tool):
    """Tool for getting the current state of exposed entities, including color."""

    name = "GetLiveContext"
    description = (
        "Provides real-time information about the CURRENT state, value, or mode of devices, "
        "sensors, entities, or areas. Use this tool for: "
        "1. Answering questions about current conditions (e.g., 'Is the light on?', 'What color is the bedroom light?'). "
        "2. As the first step in conditional actions (e.g., 'If the weather is rainy, turn off sprinklers' requires checking the weather first)."
        "3. Answering questions about what a media player or location is listening to or playing (e.g. 'What is playing in the lounge?')."
    )
    parameters = vol.Schema({})

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        """Get the current state of exposed entities, preserving the original logic."""
        if llm_context.assistant is None:
            return {"success": False, "error": "No assistant configured"}

        entity_registry = er.async_get(hass)
        interesting_attributes = {
            "temperature", "current_temperature", "temperature_unit", "brightness",
            "humidity", "unit_of_measurement", "device_class", "current_position",
            "percentage", "volume_level", "media_title", "media_artist",
            "media_album_name", "rgb_color", "hs_color"
        }
        entities_by_domain = {}

        for state in sorted(hass.states.async_all(), key=lambda s: s.name):
            if not async_should_expose(hass, llm_context.assistant, state.entity_id):
                continue

            entity_entry = entity_registry.async_get(state.entity_id)
            names = [state.name]
            if entity_entry:
                names.extend(entity_entry.aliases)

            info: dict[str, Any] = {
                "names": ", ".join(names),
                "state": str(state.state),
            }

            attributes = {
                k: str(v) for k, v in state.attributes.items() if k in interesting_attributes
            }
            if attributes:
                info["attributes"] = attributes

            if state.domain not in entities_by_domain:
                entities_by_domain[state.domain] = []
            entities_by_domain[state.domain].append(info)

        if not entities_by_domain:
            return {"success": False, "error": llm.NO_ENTITIES_PROMPT}

        return {
            "success": True,
            "result": "Live Context:\n" + yaml_util.dump(entities_by_domain),
        }


class CustomLocalAI_API(llm.AssistAPI):
    """A custom LLM API for LocalAI with tailored tool descriptions."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the custom API."""
        super().__init__(hass=hass)
        self.id = entry.entry_id
        self.entry = entry
        self.name = f"LocalAI Custom Tools ({entry.title})"

    async def async_get_api_prompt(self, llm_context: llm.LLMContext) -> str:
        """
        Get the final prompt for the API, overriding the default Home Assistant behavior.

        This method takes full control of prompt generation to prevent duplication
        and ensure the static device context is placed correctly.
        """
        # 1. Start with the user-defined prompts passed from the conversation agent.
        # This method now takes full control and ignores llm_context.user_prompt
        # to prevent any duplication from upstream.

        options = self.entry.options

        # 2. Retrieve all prompt components from the configuration.
        system_prompt = options.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT)
        base_instructions = options.get(CONF_BASE_INSTRUCTIONS, DEFAULT_BASE_INSTRUCTIONS)
        area_aware_prompt = options.get(CONF_AREA_AWARE_PROMPT, DEFAULT_AREA_AWARE_PROMPT)
        no_area_prompt = options.get(CONF_NO_AREA_PROMPT, DEFAULT_NO_AREA_PROMPT)
        timer_unsupported_prompt = options.get(CONF_TIMER_UNSUPPORTED_PROMPT, DEFAULT_TIMER_UNSUPPORTED_PROMPT)
        dynamic_context_prompt = options.get(CONF_DYNAMIC_CONTEXT_PROMPT, DEFAULT_DYNAMIC_CONTEXT_PROMPT)
        tool_prompt = options.get(CONF_TOOL_PROMPT, DEFAULT_TOOL_PROMPT)

        # 3. Build the prompt in the correct order.
        prompt_parts = []
        for part in [system_prompt, base_instructions]:
            if part and part.strip():
                prompt_parts.append(part)

        # 4. Replicate the preamble logic here to ensure it's only done once.
        area: ar.AreaEntry | None = None
        floor: fr.FloorEntry | None = None
        if llm_context.device_id:
            device_reg = dr.async_get(self.hass)
            device = device_reg.async_get(llm_context.device_id)

            if device:
                area_reg = ar.async_get(self.hass)
                if device.area_id and (area := area_reg.async_get_area(device.area_id)):
                    floor_reg = fr.async_get(self.hass)
                    if area.floor_id:
                        floor = floor_reg.async_get_floor(area.floor_id)

        if floor and area:
            floor_info = f" (floor {floor.name})"
            if area_aware_prompt and area_aware_prompt.strip():
                prompt_parts.append(area_aware_prompt.format(area_name=area.name, floor_info=floor_info))
        elif area:
            if area_aware_prompt and area_aware_prompt.strip():
                prompt_parts.append(area_aware_prompt.format(area_name=area.name, floor_info=""))
        else:
            if no_area_prompt and no_area_prompt.strip():
                prompt_parts.append(no_area_prompt)

        if not llm_context.device_id or not intent.async_device_supports_timers(self.hass, llm_context.device_id):
            if timer_unsupported_prompt and timer_unsupported_prompt.strip():
                prompt_parts.append(timer_unsupported_prompt)

        for part in [dynamic_context_prompt, tool_prompt]:
            if part and part.strip():
                prompt_parts.append(part)

        # 5. Manually generate and append the static device context.
        # This logic is replicated from the CustomGetLiveContextTool to ensure it works
        # without relying on a non-public helper function.
        exposed_entities: dict[str, list[dict]] = {}
        entity_registry = er.async_get(self.hass)
        for state in self.hass.states.async_all():
            if not async_should_expose(self.hass, llm_context.assistant, state.entity_id):
                continue

            entity_entry = entity_registry.async_get(state.entity_id)
            names = [state.name]
            if entity_entry and entity_entry.aliases:
                names.extend(entity_entry.aliases)

            exposed_entities.setdefault(state.domain, []).append(
                {"names": ", ".join(names), "entity_id": state.entity_id}
            )
        if exposed_entities:
            prompt_parts.append(f"Static Context: An overview of the areas and the devices in this smart home:\n{yaml_util.dump(exposed_entities)}")

        # 6. Join all parts to create the final, complete prompt.
        return "\n".join(prompt_parts)

    def _async_get_tools(
        self, llm_context: llm.LLMContext, exposed_entities: dict | None
    ) -> list[llm.Tool]:
        """Get the tools for the API."""
        tools = super()._async_get_tools(llm_context, exposed_entities)

        for i, tool in enumerate(tools):
            if tool.name == "GetLiveContext":
                tools[i] = CustomGetLiveContextTool()
                break
        else:
            tools.append(CustomGetLiveContextTool())
            
        for tool in tools:
            if tool.name == "HassTurnOn":
                tool.description = (
                    "Turns on or opens a device, like a light, switch, or cover. "
                    "You MUST provide a target for this action. Use the 'name' parameter to specify the device by its name (e.g., 'living room lamp'). "
                    "You can also target an entire 'area' (e.g., 'living room'). "
                    "Example: To turn on a light named 'kitchen overhead', call with {'name': 'kitchen overhead', 'domain': 'light'}."
                )
            elif tool.name == "HassTurnOff":
                tool.description = (
                    "Turns off or closes a device, like a light, switch, or cover. "
                    "You MUST provide a target for this action. Use the 'name' parameter to specify the device by its name (e.g., 'living room lamp'). "
                    "You can also target an entire 'area' (e.g., 'living room'). "
                    "Example: To turn off a light named 'kitchen overhead', call with {'name': 'kitchen overhead', 'domain': 'light'}."
                )
            elif tool.name == "HassToggle":
                tool.description = (
                    "Toggles a device on or off. "
                    "Use for commands like 'toggle the living room switch'."
                )
            elif tool.name == "HassSetPosition":
                tool.description = (
                    "Sets the position of a cover entity, like blinds or a garage door. The 'position' should be a number between 0 and 100."
                )
            elif tool.name == "HassBroadcast":
                tool.description = "Broadcasts a message to all speakers."
            elif tool.name == "HassListAddItem":
                tool.description = "Adds an item to a to-do list."
            elif tool.name == "HassListCompleteItem":
                tool.description = "Marks an item on a to-do list as complete."
            elif tool.name == "HassClimateSetTemperature":
                tool.description = "Sets the temperature of a climate device."
            elif tool.name == "HassLightSet":
                tool.description = (
                    "Adjusts the properties of a light, like color or brightness. "
                    "Only use one of 'color', 'brightness', or 'temperature' at a time. "
                    "Brightness should be a number between 0 and 100."
                )
            elif tool.name == "HassMediaUnpause":
                tool.description = "Resumes a media player."
            elif tool.name == "HassMediaPause":
                tool.description = "Pauses a media player."
            elif tool.name == "HassMediaNext":
                tool.description = "Skips to the next track on a media player."
            elif tool.name == "HassMediaPrevious":
                tool.description = "Goes to the previous track on a media player."
            elif tool.name == "HassSetVolume":
                tool.description = "Sets the volume of a media player. The 'volume_level' should be a number between 0 and 100."
            elif tool.name == "HassMediaSearchAndPlay":
                tool.description = "Searches for media and plays it on a media player."
            elif tool.name == "HassTimerStatus":
                tool.description = "Checks the status of a timer."
            elif tool.name == "HassStartTimer":
                tool.description = "Starts a timer."
            elif tool.name == "HassCancelTimer":
                tool.description = "Cancels a timer."
            elif tool.name == "HassIncreaseTimer":
                tool.description = "Increases the duration of a timer."
            elif tool.name == "HassDecreaseTimer":
                tool.description = "Decreases the duration of a timer."
            elif tool.name == "HassPauseTimer":
                tool.description = "Pauses a timer."
            elif tool.name == "HassUnpauseTimer":
                tool.description = "Resumes a paused timer."
            elif tool.name == "HassGetCurrentDate":
                tool.description = "Gets the current date."
            elif tool.name == "HassGetCurrentTime":
                tool.description = "Gets the current time."

        return tools
