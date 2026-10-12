from src.core.rag_factory import (
    create_rag_pipeline
)

def main():

    pipeline = create_rag_pipeline()

    question = input(
        "Ask a question: "
    )

    answer = pipeline.query(
        question
    )

    print("\nAnswer:\n")
    print(answer)

if __name__ == "__main__":
    main()