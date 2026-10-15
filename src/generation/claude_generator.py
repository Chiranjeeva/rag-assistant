from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()

from src.config.settings import (
    ANTHROPIC_API_KEY,
    GENERATION_MODEL,
)

class ClaudeGenerator:
    """
    Handles answer generation using Claude.
    """

    def __init__(self):

        self.client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    @staticmethod
    def extract_text_from_response(response) -> str:
        """
        Safely extract text from Anthropic responses.
        """

        text_parts = []

        for block in response.content:

            if hasattr(block, "text"):

                text_parts.append(
                    block.text
                )

        return "\n".join(
            text_parts
        ).strip()

    def generate(
        self,
        question: str,
        retrieved_chunks: list[dict],
        system_prompt: str | None = None,
    ) -> str:

        if not retrieved_chunks:

            return (
                "I could not find relevant "
                "information to answer that question."
            )

        default_system_prompt = """
            You are a helpful AI assistant.

            Answer ONLY using the provided context.

            If the context does not contain enough information,
            say that the information is unavailable.

            Do not invent facts.

            Always cite source documents when possible.
            """

        context_blocks = []

        for chunk in retrieved_chunks:

            source = chunk.get(
                "source",
                "unknown"
            )

            text = chunk.get(
                "text",
                ""
            )

            context_blocks.append(
                f"[Source: {source}]\n{text}"
            )

        context = "\n\n---\n\n".join(
            context_blocks
        )

        user_prompt = f"""
            Context:

            {context}

            Question:

            {question}
            """

        response = self.client.messages.create(
            model=GENERATION_MODEL,
            max_tokens=1024,
            system=(
                system_prompt
                or default_system_prompt
            ),
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        )

        print("\nResponse blocks:")

        for block in response.content:
            print(type(block))

        return self.extract_text_from_response(
                response
            )