from langchain_google_vertexai import VertexAIEmbeddings
from langchain_postgres import PGVector


class MyVectorStore:

    def __init__(self) -> None:
        self.embeddings = VertexAIEmbeddings(
            model_name="textembedding-gecko-multilingual@001",
        )
        self.connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
        self.collection_name = "my_docs"

        self.vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=self.connection,
            use_jsonb=True,
        )
