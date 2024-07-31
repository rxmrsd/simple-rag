"""index.py"""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.vectorstore import MyVectorStore

DOC_PATH = Path("./docs/attack.txt")


def main() -> None:
    """Main_function"""
    my_vectorstore = MyVectorStore()

    with DOC_PATH.open() as f:
        state_of_the_union = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
    )

    texts = text_splitter.create_documents([state_of_the_union])
    for i, text in enumerate(texts):
        text.id = str(i)

    my_vectorstore.vectorstore.add_documents(
        texts,
        ids=[doc.id for doc in texts],
    )


if __name__ == "__main__":
    main()
