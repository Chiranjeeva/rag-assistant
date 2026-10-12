from src.generation.claude_generator import (
    ClaudeGenerator
)

def main():

    generator = ClaudeGenerator()

    retrieved_chunks = [
        {
            "source": "security_policy.pdf",
            "text": """
                    Password reset requests must use
                    the company self-service portal.

                    Passwords require at least
                    14 characters.
            """
        }
    ]

    answer = generator.generate(
        question=(
            "How do I reset my password?"
        ),
        retrieved_chunks=retrieved_chunks,
    )

    print()
    print(answer)
    print()


if __name__ == "__main__":
    main()