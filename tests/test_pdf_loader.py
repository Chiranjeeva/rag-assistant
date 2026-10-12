from src.loaders.pdf_loader import (
    PDFLoader
)

def main():

    loader = PDFLoader()

    document = loader.load(
        "data/raw/security_policy.pdf"
    )

    print()

    print("Document ID:")
    print(document.id)

    print()

    print("Metadata:")
    print(document.metadata)

    print()

    print("Preview:")

    print(
        document.text[:500]
    )

    print()

if __name__ == "__main__":
    main()