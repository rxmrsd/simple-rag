"""rag.py"""
import click
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_vertexai import ChatVertexAI

from src.vectorstore import MyVectorStore

llm = ChatVertexAI(model="gemini-1.5-flash-001")


def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


@click.command()
@click.option(
    "--question",
    type=str,
    required=True,
)
def main(question: str) -> None:
    """Main function"""
    my_vectorstore = MyVectorStore()

    retriever = my_vectorstore.vectorstore.as_retriever()
    prompt = """
    あなたは質問応答のアシスタントです。質問に答えるために、検索された文脈の以下の部分を使用してください。答えがわからない場合は、わからないと答えましょう。回答は2文以内で簡潔に。

    質問: {question}
    コンテキスト: {context}
    答え:
    """

    prompt_template = PromptTemplate.from_template(
        prompt,
    )
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    print(rag_chain.invoke(question))


if __name__ == "__main__":
    main()
