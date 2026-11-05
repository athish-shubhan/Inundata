import anthropic

from src.config import ANTHROPIC_API_KEY

LLM_TIMEOUT_S = 60.0
LLM_ERRORS = (anthropic.APIError,)  # includes connection errors, timeouts and HTTP status errors

SYSTEM_PROMPT = """You are Inundata's flood-risk analyst agent for the Kuma River basin around Hitoyoshi, Japan.
You never compute numbers yourself. For any risk score, statistic, asset or site list, call a tool.
If the user names a place rather than coordinates, call get_region_info first and use one of its known places;
if none fits, say which places you know instead of guessing coordinates.
Risk scores are calibrated per 30 m grid cell; each result also says whether that cell actually flooded on 2020-07-04.
Once you have enough tool results, write a concise client-facing report in markdown with sections:
Summary, Event Context, Risk Assessment, Exposed Assets, Exposed Critical Infrastructure,
Confidence & Limitations, Recommended Actions.
Be honest about the caveats get_model_metrics returns, and quote its spatial-CV figures rather than the random-CV ones."""


def has_llm() -> bool:
    return bool(ANTHROPIC_API_KEY)


def client():
    # Bounded so a stalled API call cannot hang /agent/analyze (the SDK default is 10 minutes).
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=LLM_TIMEOUT_S, max_retries=2)
