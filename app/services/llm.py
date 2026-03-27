"""LLM service: sends prompts to Anthropic / OpenAI and returns completions."""


async def complete(prompt: str, model: str = "claude-3-5-sonnet-20241022") -> str:
    """Return a completion for the given prompt."""
    # TODO: implement using Anthropic or OpenAI SDK
    raise NotImplementedError
