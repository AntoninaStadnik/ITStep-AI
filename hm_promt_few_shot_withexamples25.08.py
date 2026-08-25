# Напишіть промпт для створення плану навчального
# курсу з певної теми для цільової айдиторії(початківці,
# професіонали, діти, тощо).
# Вхідні параметри: тема, опис цільової аудиторії
# Реалізуйте двома способами:
#  Zero-shot
#  Few-shot


import os
import dotenv
from langchain.chains.flare.prompts import PROMPT_TEMPLATE
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key

)

promt = PromptTemplate.from_template("""
        Ти -- професійни вчитель з багаторічним досвідом у навчанні студентів і практичним досвідом на великому SaaS проекті
        Задача: твоя задача створити курс навчання по QA manual. Потрібно реалізувати курс для різних рівнів:
                початківці, професіонали(люди з досвідом 5+ років), діти

        ###ПРАВИЛА###
        1. пиши теорію і приводь приклоади, як то можна застосувати на практиці
        2. пиши до кожного прикладу максимально просте і зрозуміле пояснення



        ###ПРИКЛАД###
        ПРИКЛАД1
        Задача: поясни, які є види тестування
        Результат: функціональне, як у себе включає: UI, API, E2E, Positive/Negative testing etc
        API - зазвичай проводиться на етапі інтеграції, або коли UI ще не готовий. E2E - фінальни йнабір тестових сценаріїв, коли фічі готова

        ПРИКЛАД2
        Задача: Які є види API
        Результат: RestAPI, GraphQL, GrCp
        Різниця між RestAPI і GraphQL, у RestAPI є єндпоінти і різні методи, у GraphQL один ендпоінт, один метод зазвичай швидше ніж RestAPI

        ###ОПИС ЗАДАЧІ###
        Задача: {customer_task}


"""
         )

education_program = "QA manual"
customer_task = "розкажи про види тестування API, які є види і в чому перевага"

data = {
    "education_program": "QA manual",
    "customer_task": customer_task
}

text = promt.invoke(data)
#print(text)

response = llm.invoke(text)
print(response)
