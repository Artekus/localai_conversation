"""LLM API for LocalAI."""
from __future__ import annotations

from typing import Any
import logging

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
import voluptuous as vol
from homeassistant.util.json import JsonObjectType
from homeassistant.util import yaml as yaml_util
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.helpers import entity_registry as er

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
        """Get the current state of exposed entities."""
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

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the custom API."""
        super().__init__(hass=hass)
        self.id = "localai_conversation"
        self.name = "LocalAI Custom Tools"

    async def async_get_api_prompt(self, llm_context: llm.LLMContext) -> str:
        """
        Get the final prompt for the API, overriding the default Home Assistant behavior.

        This method takes full control of prompt generation to prevent duplication
        and ensure the static device context is placed correctly.
        """
        # 1. Start with the user-defined prompts passed from the conversation agent.
        system_prompt_parts = [llm_context.user_prompt]

        # 2. Manually generate the static device context.
        exposed_entities = llm.async_get_exposed_entities(self.hass, llm_context)
        if exposed_entities:
            static_context_str = yaml_util.dump(exposed_entities)
            # The header is intentionally included to match the user's prompt expectation.
            system_prompt_parts.append(f"Static Context: An overview of the areas and the devices in this smart home:\n{static_context_str}")

        # 3. Join all parts to create the final, complete prompt.
        return "\n".join(system_prompt_parts)

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
