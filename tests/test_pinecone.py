from src.vectorstores.pinecone_store import (
    PineconeStore
)

def main():

    store = PineconeStore()
    stats = store.describe_index()

    print(stats)

if __name__ == "__main__":
    main()