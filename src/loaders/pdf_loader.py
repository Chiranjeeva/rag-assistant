from pathlib import Path

from pypdf import PdfReader

from src.models.document import Document

class PDFLoader:
    """
    Loads PDF files and converts them
    into Document objects.
    """

    def load(
        self,
        pdf_path: str,
    ) -> Document:

        path = Path(pdf_path)

        if not path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        reader = PdfReader(
            str(path)
        )

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        full_text = "\n".join(
            pages
        )

        return Document(
            id=path.stem,
            text=full_text,
            metadata={
                "source": path.name,
                "file_type": "pdf",
                "page_count": len(reader.pages),
            },
        )
