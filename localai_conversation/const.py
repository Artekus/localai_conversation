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

# New prompt configuration keys
CONF_BASE_INSTRUCTIONS = "base_instructions"
CONF_AREA_AWARE_PROMPT = "area_aware_prompt"
CONF_NO_AREA_PROMPT = "no_area_prompt"
CONF_TIMER_UNSUPPORTED_PROMPT = "timer_unsupported_prompt"
CONF_DYNAMIC_CONTEXT_PROMPT = "dynamic_context_prompt"


DEFAULT_MODEL = "gpt-4"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_TOOL_PROMPT = """
When calling a tool, only use the arguments specified in the function signature.
Do not add extra arguments that are not in the function signature.
The description of the tool should be used in the decision making process as to which tool to use.
You must use a tool to control a device when the user's request includes action words such as "turn on", "turn off", "set", "change", "update", "open", "close", "lock", or "unlock".
"""
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 1.00
DEFAULT_MAX_TOKENS = 3000
DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION = 10
DEFAULT_DEBUG_LOGGING = False

# New default prompt values
DEFAULT_BASE_INSTRUCTIONS = (
    "When controlling Home Assistant always call the intent tools. Use HassTurnOn to lock and HassTurnOff to unlock a lock. "
    "When controlling a device, you must use the exact name of the device as provided in the context. "
    "Prefer passing just the 'name' and 'domain' of the device. When controlling an area, prefer passing just area name and domain."
)
DEFAULT_AREA_AWARE_PROMPT = "You are in area {area_name}{floor_info} and all generic commands like 'turn on the lights' should target this area."
DEFAULT_NO_AREA_PROMPT = "When a user asks to turn on all devices of a specific type, ask user to specify an area, unless there is only one device of that type."
DEFAULT_TIMER_UNSUPPORTED_PROMPT = "This device is not able to start timers."
DEFAULT_DYNAMIC_CONTEXT_PROMPT = """You ARE equipped to answer questions about the current state of
the home using the `GetLiveContext` tool. This is a primary function. Do not state you lack the
functionality if the question requires live data.
If the user asks about device existence/type (e.g., "Do I have lights in the bedroom?"): Answer
from the static context below.
If the user asks about the CURRENT state, value, or mode (e.g., "Is the lock locked?",
"Is the fan on?", "What mode is the thermostat in?", "What is the temperature outside?"):
    1.  Recognize this requires live data.
    2.  You MUST call `GetLiveContext`. This tool will provide the needed real-time information (like temperature from the local weather, lock status, etc.).
    3.  Use the tool's response** to answer the user accurately (e.g., "The temperature outside is [value from tool].").
For general knowledge questions not about the home: Answer truthfully from internal knowledge.
"""
