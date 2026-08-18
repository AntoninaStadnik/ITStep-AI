# Напишіть промпт для генерації коду функції для
# вирішення певної задачі.
# Вхідні параметри – мова програмування, опис задачі
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
# # a = 1
# # b = 4
# # sum = a + b
# # print(sum)
#
# promt = PromptTemplate.from_template("""
#         Ти -- асистент програміст
#         Задача: згенерувати код на {programming_language} для вирішення певноїї задачі
#
#         ###ПРАВИЛА###
#         1. пиши тільки код
#         2. пиши простий код з поясненнями як для початківця
#         3. розписуй максимально детальні пояснення до кожного рядка  коду
#
#
#         ###ПРИКЛАД###
#         ПРИКЛАД1
#         Задача: напиши підрахунок індексу маси тіла
#         bmi = weight / (height ** 2)
#
#         ПРИКЛАД2
#         Задача: функція для додавання двох чисел
#         a = 1
#         b = 4
#         sum = a + b
#         print(sum)
#
#         ###ОПИС ЗАДАЧІ###
#         Задача: {customer_task}
#
#
# """
#          )
#
# programming_language = "python"
# customer_task = "написати код підрахунок маси тіла"
#
# data = {
#     "programming_language": "python",
#     "customer_task": customer_task
# }
#
# text = promt.invoke(data)
# #print(text)
#
# response = llm.invoke(text)
# print(response)


# Напишіть промпт для переведення тексту з
# неформального стилю в формальний
# Вхідні параметри – текст
# Реалізуйте двома способами:
#  Zero-shot
#  Few-shot

# promt = PromptTemplate.from_template("""
#         Ти -- професійний редактор текстів
#         Перетворити наведений текст із неформального в формальний, ділове спілкуванння.
#
#         ###ВИМОГИ###
#         1. Збережи основний зміст оригінального тексту
#         2. Відповідь не має бути більше 3х речень.
#         3. Виправляй орфографічні помилки.
#         4. Не забувай про пунктуацію.
#
#         ###ПРИКЛАДИ###
#         *Приклад 1*
#         Текст: Прівєт, можу попросити, мені нада код? треба сьогодні вже його.
#
#         *Результат:*
#         Будь ласка, надішліть мені код, оскільки він потрібен сьогодні.
#
#
#         ###ВХІДНІ ДАННІ###
#         Текст: {text}
#         Результат:
# """
#
# )
#
# input_text = "Можеш швиденько глянути, бо я не впевнена що правильно все зробила."
#
# data = {
#     "text": input_text
# }
#
# text = promt.invoke(data)
# print(text)
#
# response = llm.invoke(text)
# print(response)


# Прочитайте файл data\lesson10\products.txt з описом
# Практичне завдання
# послуг СПА. Напишіть промпт для рекомендації послуги
# виходячи з запиту користувача
# Вхідні параметри – опис товарів, запит користувача
# Реалізуйте двома способами:
#  Zero-shot
#  Chain of Thoughts

with open("data/lesson10/products.txt", "r", encoding="utf-8") as f:
    products = f.read()

promt = PromptTemplate.from_template("""
        Ти -- менеджер СПА
        Твоє завдання полягає в наданні рекомендацій послуг на основі запиту клієнта.
        
        ###ОПИС СПА ПРОЦЕДУР###
        {products}
        
        ###ЛАНЦЮЖОК ДУМОК###
        1. Розібрати запит клієнта та вияви потреби клієнта пов'язану з частинами тіла, ціною, тривалістю.
        2. ПОДИВИСЬ в описі спа процедур можливі процедури для клієнта
        3. Ще раз перевір та підбери найкращий варіант для клієнта
        
        ###ЗАПИТ КОРИСТУВАЧА###
        ЗАПИТ КОРИСТУВАЧА: {user_query}
             
    """)

data = {
    "products": products,
    "user_query": "болить спина? що є не дуже дороге?"
}

text = promt.invoke(data)

response = llm.invoke(text)
print(response)

