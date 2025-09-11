"""The LocalAI Conversation integration."""
from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm as hass_llm

from .const import DOMAIN
from .llm import CustomLocalAI_API

PLATFORMS: list[str] = ["conversation"]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up LocalAI Conversation from a config entry."""
    domain_data = hass.data.setdefault(DOMAIN, {"entries": {}, "api_registered": False})

    # Register the API only once, when the first entry is loaded
    if not domain_data.get("api_registered"):
        api = CustomLocalAI_API(hass)
        unregister_cb = hass_llm.async_register_api(hass, api)
        domain_data["unregister_api"] = unregister_cb
        domain_data["api_registered"] = True

    domain_data["entries"][entry.entry_id] = entry

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(update_listener))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        domain_data = hass.data.get(DOMAIN)
        if domain_data:
            domain_data["entries"].pop(entry.entry_id, None)

            # If this is the last entry being unloaded, unregister the API
            if not domain_data["entries"]:
                if unregister_cb := domain_data.pop("unregister_api", None):
                    unregister_cb()
                domain_data["api_registered"] = False
                hass.data.pop(DOMAIN)

    return unload_ok


async def update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)
