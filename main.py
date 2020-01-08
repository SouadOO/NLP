import telebot
import chatbot

TOKEN   = "952378975:AAGP7ZFzg6Mm6xu9G77GNj3l7CLQhTUZ5mk"
tebot   = telebot.TeleBot(TOKEN)
myBot   = chatbot.FAQ_Bot()

print("I am Ready to answer question on telegram NLP_FAQ_unige.  ")


def handle_messages(messages):
	for message in messages:	
		#print("question is :",str(message))
		answer = myBot.search_answer(str(message.text))
		tebot.reply_to(message, answer)

#tebot.send_message(text="Welcome to Zappos FAQ, how can I help you?")

tebot.set_update_listener(handle_messages)
tebot.polling()
