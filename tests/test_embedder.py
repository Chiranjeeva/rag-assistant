from src.embeddings.voyage_embeddings import VoyageEmbedder

def main():

    embedder = VoyageEmbedder()
    vector = embedder.embed_text(
        "Hello world"
    )

    print(type(vector))
    print(len(vector))

if __name__ == "__main__":
    main()