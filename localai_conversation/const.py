"""Constants for the LocalAI Conversation integration."""

DOMAIN = "localai_conversation"

CONF_URL = "url"
CONF_MODEL = "model"
CONF_SYSTEM_PROMPT = "system_prompt"
CONF_TOOL_PROMPT = "tool_prompt"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_MAX_TOKENS = "max_tokens"
CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION = "max_function_calls_per_conversation"
CONF_DEBUG_LOGGING = "debug_logging"

DEFAULT_MODEL = "gpt-4"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_TOOL_PROMPT = """
When calling a tool, only use the arguments specified in the function signature.
Do not add extra arguments that are not in the function signature.
For HassTurnOn and HassTurnOff, do not use the 'device_class' argument for entities in the 'light' domain.
Only provide the 'name' and 'domain' for lights. For other devices, you may use 'area' and other relevant fields if they are available in the function signature.
"""
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 150
DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION = 1
DEFAULT_DEBUG_LOGGING = False
