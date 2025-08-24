from dotenv import load_dotenv
from telebot import types
import os
import telebot
from llm_module import RAGWithQdrant
from retrieval_module import DataRetrieval
from rag_system import RAGSystem

print("initing")
load_dotenv()  # Загружает переменные из .env
LLM_MODEL_NAME = "Qwen/Qwen-1_8B-Chat"
EMBEDDING_MODEL = "BAAI/bge-m3"
QDRANT_URL= "http://localhost:6333"
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

rag = RAGSystem(
    qdrant_url=QDRANT_URL,
    collection_name="starwars",
    embedding_model_name="BAAI/bge-m3",
    llm_model_name="Qwen/Qwen-1_8B-Chat"
)

# Задаём вопрос
response = rag.ask("Сколько планет в галактике Звёздных войн?")
print("Вопрос:", response["question"])
print("Ответ:", response["answer"])
print("\nИсточники:")
for i, src in enumerate(response["sources"], 1):
    print(f"[Источник {i}] {src['content'][:200]}...")

bot = telebot.TeleBot(os.getenv('BOT_TOKEN'))
# retrieval = DataRetrieval(EMBEDDING_MODEL, QDRANT_URL, COLLECTION_NAME)
# llm = RAGWithQdrant(LLM_MODEL_NAME, retrieval)

@bot.message_handler(commands = ['start'])
def url(message):
    markup = types.InlineKeyboardMarkup()
    btn1 = types.InlineKeyboardButton(text='Наш сайт', url='https://habr.com/ru/all/')
    markup.add(btn1)
    bot.send_message(message.from_user.id, "По кнопке ниже можно перейти на сайт хабра", reply_markup = markup)

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == '👋 Поздороваться':
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True) #создание новых кнопок
        btn1 = types.KeyboardButton('Как стать автором на Хабре?')
        btn2 = types.KeyboardButton('Правила сайта')
        btn3 = types.KeyboardButton('Советы по оформлению публикации')
        markup.add(btn1, btn2, btn3)
        bot.send_message(message.from_user.id, '❓ Задайте интересующий вопрос', reply_markup=markup) #ответ бота

bot.polling(none_stop=True, interval=0)