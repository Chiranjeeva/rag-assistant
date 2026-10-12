from src.core.rag_factory import (
    create_rag_pipeline
)

def main():

    pipeline = create_rag_pipeline()

    print(type(pipeline))

if __name__ == "__main__":
    main()