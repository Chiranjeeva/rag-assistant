from src.config.settings import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)

class TextChunker:
    """
    Splits text into overlapping chunks and
    returns chunk metadata.
    """

    def __init__(
        self,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        text: str,
    ) -> list[dict]:

        if not text.strip():
            return []

        words = text.split()

        step = self.chunk_size - self.chunk_overlap

        raw_chunks = []

        for i in range(
            0,
            len(words),
            step
        ):

            chunk_text = " ".join(
                words[i:i + self.chunk_size]
            )

            if chunk_text:
                raw_chunks.append(chunk_text)

        total_chunks = len(raw_chunks)

        chunks = []

        for index, chunk_text in enumerate(
            raw_chunks
        ):

            chunks.append(
                {
                    "text": chunk_text,
                    "chunk_index": index,
                    "total_chunks": total_chunks,
                }
            )

        return chunks