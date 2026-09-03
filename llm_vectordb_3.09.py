# Створіть векторну базу даних, де кожен документ – це
# вміст файлу з папки data/lesson_rag/files
#  добавте в метадані шлях до файлу
#  створіть для кожного документу ID
#  збережіть створені ID та назви відповідних файлів в
# окремий json файл
# Перевірте чи працює правильно пошук

import dotenv
import os


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

with open("data/lesson_rag/files/future_of_ai.txt", "r", encoding="utf-8") as file:
    text1 = file.read()
    doc1 = Document(
        page_content=text1,
    )

with open("data/lesson_rag/files/intro.txt", "r", encoding="utf-8") as file:
    text2 = file.read()
    doc2 = Document(
        page_content=text2,
    )

with open("data/lesson_rag/files/machine_learning.txt", "r", encoding="utf-8") as file:
    text3 = file.read()
    doc3 = Document(
        page_content=text3,
    )

with open("data/lesson_rag/files/neural_networks.txt", "r", encoding="utf-8") as file:
    text4 = file.read()
    doc4 = Document(
        page_content=text4,
    )


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

doc1 = Document(
    page_content=text1,
)

doc2 = Document(
    page_content=text2,
)

doc3 = Document(
    page_content=text3,
)

doc4 = Document(
    page_content=text4,
)

documents = [doc1, doc2, doc3, doc4]
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(
    documents=documents,
    ids=uuids
)