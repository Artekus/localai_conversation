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
    hass.data.setdefault(DOMAIN, {})

    api = CustomLocalAI_API(hass, entry)
    unregister_cb = hass_llm.async_register_api(hass, api)
    hass.data[DOMAIN][entry.entry_id] = {"api": api, "unregister_cb": unregister_cb}
    
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(update_listener))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok and entry.entry_id in hass.data[DOMAIN]:
        entry_data = hass.data[DOMAIN].pop(entry.entry_id)
        if unregister_cb := entry_data.get("unregister_cb"):
            unregister_cb()

    return unload_ok


async def update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)
