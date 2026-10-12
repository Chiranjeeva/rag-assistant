import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "rag-demo"
)

PINECONE_CLOUD = os.getenv(
    "PINECONE_CLOUD",
    "aws"
)

PINECONE_REGION = os.getenv(
    "PINECONE_REGION",
    "us-east-1"
)

EMBEDDING_MODEL = "voyage-4"

GENERATION_MODEL = "claude-sonnet-5-5"

EMBEDDING_DIMENSIONS = 1024

DEFAULT_TOP_K = 5

SIMILARITY_THRESHOLD = 0.5

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50