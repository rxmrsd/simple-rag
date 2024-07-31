# はじめに

LLMと検索をうまく組み合わせて，AIによる回答精度を高める，RAG(Retrieval-Augmented Generation)というフレームワークがあります．最近では，Advanced RAGやGraph RAGといった，更に進展したRAGが見られるようになり，これからさらに回答精度を高くするための研究が進んでいくと考えられます．今回は基本的な初期のRAGを今さらながら構築します．


# お題

今回は，私の最も好きな漫画である「進撃の巨人」についてLLMに回答させます．その際，元となるドキュメントを参照するとき(RAGあり)としないとき(RAGなし)で回答にどのような差が生まれるのか確認します．

元となるドキュメントは，[進撃の巨人のWikipedia](https://ja.wikipedia.org/wiki/%E9%80%B2%E6%92%83%E3%81%AE%E5%B7%A8%E4%BA%BA)から物語に関連する部分を自分で抜粋したものになります．

# 構成

今回は以下のような構成で構築します．LangchainとPGVectorを用いて構築します．
Embedding Modelは`textembedding-gecko-multilingual@001`，Chat Modelは`gemini-1.5-flash-001`を利用しました．

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3618319/62d5d1be-b696-8185-8231-be00eaf506cb.png)


# RAG構築
## DBの準備

Dockerを用いてPGVectorを構築します．`docker-compose.yml`に以下を記載し，コマンドを実行するのみです．

```bash
docker-compose up -d
```

```yaml:docker-compose.yml
services:
  pgvector:
    image: pgvector/pgvector:pg16
    platform: linux/amd64
    container_name: pgvector-container
    environment:
      POSTGRES_USER: langchain
      POSTGRES_PASSWORD: langchain
      POSTGRES_DB: langchain
    ports:
      - "6024:5432"
    volumes:
      - pgvector_data:/var/lib/postgresql/data

volumes:
  pgvector_data:
```


## Indexing
まずはデータベースを構築します．手順は以下です．

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3618319/68997d2c-8962-aa20-a3a1-74a22f0b3d79.png)

1． チャンクの作成： 元のドキュメントをある程度の単位で文章を区切る．
2． Embedding： 1で区切ったチャンクごとにベクトル化する
3． ベクトルDBに格納： ベクトル化したものをデータベースに格納する


```python:(抜粋)index.py
embeddings = VertexAIEmbeddings(
    model_name="textembedding-gecko-multilingual@001",
)
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "my_docs"

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

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
vectorstore.add_documents(texts, ids=[doc.id for doc in texts])

```

PGVectorには以下のように，Embeddingしたベクトルと元のドキュメントが紐づく形で格納されます．


![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3618319/2c662909-fa38-3205-653c-f5d6801c82f8.png)



## Retrieval and Generation

### Prompt
ユーザーからの質問に対して，データベースに格納したドキュメントを参照するように指示を与える必要があります．プロンプトにその旨を記載します．以下がプロンプトで，[LangSmith Hubにある`rag-prompt`](https://smith.langchain.com/hub/rlm/rag-prompt)を日本語にしたものです．


```python
prompt = """
あなたは質問応答のアシスタントです。質問に答えるために、検索された文脈の以下の部分を使用してください。答えがわからない場合は、わからないと答えましょう。回答は2文以内で簡潔に。

質問: {question}
コンテキスト: {context}
答え:
"""
```

### Chainの作成
上記のプロンプトと，LLMをつなぎ，ユーザーからの質問に対して回答させるためにチェーンを繋ぎます．

```python：(抜粋)rag.py
retriever = vectorstore.as_retriever()
llm = ChatVertexAI(model="gemini-1.5-flash-001")
prompt_template = PromptTemplate.from_template(prompt)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)
```

# 質問する

RAGの構築ができたので，実際にこのシステムを利用します．今回は，特定の文章をもとに回答する場合とそうでない場合で，回答がどのように異なるのか確認します．

**質問: 主人公は誰ですか？**
```python 
rag_chain.invoke("主人公は誰ですか？")
```

:::note info
RAGありの回答

*主人公はエレン・イェーガーです。彼は第104期訓練兵団の5席で、母親を巨人に殺された過去から巨人の駆逐を目標としています。*

:::

:::note alert

RAGなし(Geminiに直接聞いてみる)の回答

*「主人公」という言葉を理解していますが、具体的な物語や文脈がないため、誰が主人公なのか分かりません。
主人公を教えてください！どんな物語なのか教えてもらえれば、誰が主人公なのか答えることができます。 😊*

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3618319/3aa5ec01-8f29-fff2-4447-79b71f778d67.png)

:::

当たり前ですが，なんの脈略もなく「主人公は誰ですか？」と聞かれても，LLMは答えようがありません．一方RAGを構築した場合は，文書を元に回答するため，質問に対し，進撃の巨人の主人公を正しく回答できています．


**質問: 第13代調査兵団団長は誰ですか？**

:::note info
RAGありの回答

*第13代調査兵団団長はエルヴィン・スミスです。彼は卓越した統率力と決断力の持ち主で、くせ者揃いの調査兵団を取りまとめていました。*

:::

:::note alert

RAGなし(Geminiに直接聞いてみる)の回答

*進撃の巨人の第13代調査兵団団長は エレン・イェーガー です。
エレンは第12代団長であり、エルヴィン・スミスを継承しました。エルヴィンが亡くなった後、エレンは兵士たちを率いて壁外への攻撃を進めました。
エレンは調査兵団の団長として、巨人の脅威に立ち向かい、人類の未来を守るために戦いました。*

![スクリーンショット 2024-07-31 10.34.49.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3618319/bc1fc025-9744-a279-041f-fa78f370b0c3.png)

:::

正しい回答はエルヴィン・スミスです．RAGなしの場合は嘘が多分に含まれています．物語を知らない人がこのようにやみくもにLLMに尋ねても嘘を回答され，それを信じてしまう可能性があり，LLMの利用の際には注意が必要です．RAGありの場合，元のWikipediaに書かれている内容をもとに回答を生成していることがわかります．

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3618319/f02063cc-21b7-902b-4ae7-e8a56d957034.png)


# まとめ

このようにRAGを用いて，特定の文章に特化してAIに回答させることで，極力うそ(ハルシネーション)を回避することができます．
冒頭に少し触れた，Advanced RAGやGraph RAGを用いることで，更に回答精度を高めることができ，よりハルシネーションを回避できます．例えば，指示代名詞(彼，それ，あの，など)が文書に含まれていた場合，それ単独の文章では，何を指しているのかわかりません．その前の文や全体を理解しておく必要があり，その点がRAGの不得手な部分になっています．Graph RAGを用いると，単語同士の関係性をつなぎ，それを持って回答を生成するため，その点を改善してくれるようです．こちらについても自分の中で整理できたらまとめたいと思います．

なぐり書きですが，今回利用した環境やコードを[こちら](https://github.com/rxmrsd/simple-rag)にまとめております．


# 参考
- [進撃の巨人Wikipedia](https://ja.wikipedia.org/wiki/%E9%80%B2%E6%92%83%E3%81%AE%E5%B7%A8%E4%BA%BA)
- [Q&A with RAG](https://python.langchain.com/v0.1/docs/use_cases/question_answering/)
- [LangSmith Hub](https://smith.langchain.com/hub/rlm/rag-prompt)

