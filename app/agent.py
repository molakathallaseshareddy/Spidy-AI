from app.llm import LLMProvider


SYSTEM_PROMPT = (
    "You are a helpful personal assistant. Answer the user's request clearly. "
    "You cannot access files, applications, the browser, or external services; "
    "do not claim to have performed actions or verified information you cannot access."
)


class Assistant:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def respond(self, message: str) -> str:
        return await self._llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_message=message,
        )