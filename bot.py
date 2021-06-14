import telepot
from telepot.loop import MessageLoop
from functions import get_data
import time
TOKEN = # EXCLUSIVE_online_bot
TelegramBot = telepot.Bot(TOKEN)


def on_chat_message(msg):
    content_type, chat_type, user = telepot.glance(msg)
    text = msg['text']
    if text == '/start':
        TelegramBot.sendMessage(user, 'введите данные резюме')
        return

    response = get_data(text)
    vacancies = response['vacancies']
    to_send = f"Бакет: {response['bucket']}\n\nваша зп: {response['salary']}"

    TelegramBot.sendMessage(user, to_send)

    for cluster in vacancies:
        to_send = ''
        for n, vacancy in enumerate(vacancies[cluster]):
            to_send += f"рекомендованный курс: {vacancies[cluster][n]['course_recommended']}\n\n"
            to_send += f"описание вакансии: {vacancies[cluster][n]['description'][:1000]}"
            TelegramBot.sendMessage(user, to_send)
            time.sleep(1)
            break  # one vacancy per cluster


print('started')
MessageLoop(TelegramBot, {'chat': on_chat_message}).run_forever()
