from src.ingestion.chunker import (
    TextChunker
)

def main():

    text = """
    Artificial intelligence is transforming
    software development.

    Engineers can now build applications
    much faster than before.

    Retrieval augmented generation enables
    language models to answer questions
    using organisation-specific information.
    """

    chunker = TextChunker(
        chunk_size=10,
        chunk_overlap=2,
    )

    chunks = chunker.chunk(text)

    for chunk in chunks:

        print(
            f"Chunk {chunk['chunk_index'] + 1}"
            f"/{chunk['total_chunks']}"
        )

        print(chunk["text"])
        print("-" * 40)

if __name__ == "__main__":
    main()