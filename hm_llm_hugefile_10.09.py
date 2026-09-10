# Добавте в створену базу даних файл
# data/lesson_rag/huge_file.txt про умови користування гуглом
# Оскільки файл надто великий, то його треба добавляти
# частинами. Для цього:
#  прочитайте вміст файлу
#  розділіть його на окремі блоки(між блоками два
# порожніх рядка, дивись файл)
#  отримайте перший рядок кожного блоку – це його
# назва
#  створіть документи для кожного блоку.
#  створіть ID та добавте все в існуючу базу даних
#  добавте ID у json файл
#  перевірте агента

import dotenv
import os
import json


from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from uuid import uuid4
from pinecone import ServerlessSpec
from pinecone import Pinecone

dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# модель для преведення текстів у вектори(набір чисел)
embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=api_key,
)

with open("data/lesson_rag/huge_file.txt", "r", encoding="utf-8") as file:
    text = file.read()

    text = text.split("\n\n\n")
    # print(text)
    # print (len(text))
    # block = text[0]
    # result = block.splitlines()
    # print(result)

    processed_docs = []

    for block in text[0 :]:
        blocks_list = block.splitlines()
        print(blocks_list[0])


        doc = Document(
            page_content= block,
                    )
        processed_docs.append(doc)

        print(len(processed_docs))

pc = Pinecone(api_key=pinecone_api_key)

index_name = "practiceit-docs"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=3072,    # кількість чисел у векторі
        metric="cosine",   # формула для пошуку схожих текстів
        spec=ServerlessSpec(
            cloud="aws",        # хмарна платформа(амазон)
            region="us-east-1"  # регіон
        ),
    )

index = pc.Index(index_name)

vector_store = PineconeVectorStore(
    index=index,          # база даних
    embedding=embedding   # модель для кодування
)


uuids = [str(uuid4()) for _ in range(len(processed_docs))]
print(f"довжина: {len(uuids)}")

with open("ids.json", "w", encoding="utf-8") as file:
    json.dump (uuids, file, indent=4, ensure_ascii=False)

vector_store.add_documents(
    documents=processed_docs,
    ids=uuids
)