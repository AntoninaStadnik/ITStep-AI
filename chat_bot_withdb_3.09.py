import dotenv
import os
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from pinecone import ServerlessSpec
from pinecone import Pinecone
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
    trim_messages,
)

# завантадити дані з .env
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
serper_key = os.getenv("SERPER_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


# # модель
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",  # назва моделі
    api_key=api_key  # ключ до сервера з моделлю
)

serper_search = GoogleSerperAPIWrapper(
    serper_api_key=serper_key
)

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=api_key,
)

pc = Pinecone(api_key=pinecone_api_key)

# створення бази даних

index_name = "practiceit-docs"  # назва бази даних

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

@tool
def document_search(query: str):
    """
    Searching of docs in the vector DB
    :param query: str -- user query
    :return: documents found
    """

    results = vector_store.similarity_search(
    query,  # текст для пошуку схожих документів
    k=1,  # кількість документів яку шукаємо
    )

    return results

agent = create_agent(
    model=llm,
    tools=[document_search]
)

messages = [
    SystemMessage("""
    Ти -- ввічлиіий чат бот

    ###ІНСТРУКЦІЯ###
    1. якщо користувач питає щось, що не пов'язане зі штучним інтелектом - відповідай
    "такої інформації немає"
    2. якщо користувач питає щось про штучний інтеелект використовуюй document_search
    """)
]

while True:
    user_query = input("Ви: ")

    if user_query == "":
        break

    user_message = HumanMessage(user_query)

    messages.append(user_message)

    data = {
        "messages": messages
    }

    data = agent.invoke(data)
    # агент так само повертає словник

    # дістаємо нову історію повідомлень
    messages = data["messages"]

    # відповідь моделі -- останнє повідомлення в історії
    response = messages[-1]

    # вивести відповідь на екран
    print(response.text)

    # виведення історії
    print()
    print("----------ІСТОРІЯ-----------")

    for message in messages:
        print(repr(message))  # вивести разом з назсою класу

    print("-----------------------------")
    print()

