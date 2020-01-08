import pandas as pd
import similarity


#Create a corpus from data and pre built answer
data=pd.read_csv("./data/data_cln.csv")
CORPUS=data.to_dict('records')


class FAQ_Bot:

    def __init__(self):
        self.event_stack = []
        self.sim = similarity.Semantic_similarity()
        self.cos=similarity.Cosine_Tfidf_similarity()
        self.settings = {
            "min_score": 0.5,
            "help_email": "zappos_Help@zappos.com",
            "faq_page": "www.zappos.com/customer-service-center"
        }

    def search_answer(self,text):
        # Check for event stack
        potential_event = None

        if(len(self.event_stack)):
            potential_event = self.event_stack.pop()
        
        if potential_event:
            return potential_event.handle_response(text, self)

        else:
            answer = self.pre_built_responses_or_none(text)
            if not answer:
                answer = similarity.find_most_similar(text,self.sim,self.cos)
                return self.answer_question(answer, text)
            else:
                return answer
            
    def answer_question(self, answer, text):
        if answer['score'] > self.settings['min_score']:
            # set off event asking if the response question is what they were looking for
            answer = "\n %s :  %s\n " % (answer['score'],answer['answer'])
        else:
            answer = "Woops! I'm having trouble finding the answer to your question.\n Would you like to see the list of questions that I am able to answer?\n"
            # set off event for corpus dump
            self.event_stack.append(Event("corpus_dump", text))

        return answer

    def pre_built_responses_or_none(self, text):
        # only return answer if exact match is found
        pre_built = [
            {
                "Question": "Hi",
                "Answer": "Hi, how can I help you?\n"
            },
            {
                "Question": "Hello",
                "Answer": "Hi, how can I help you?\n"
            },
            {
                "Question": "ok",
                "Answer": "welcome\n"
            },
            {
                "Question": "Who made you?",
                "Answer": "I was created by Souad.\n"
            },
            {
                "Question": "When were you born?",
                "Answer": "I first opened my eyes in December 27th, 2019.\n"
            },
            {
                "Question": "What is your purpose?",
                "Answer": "I assist user experience by providing an interactive FAQ chat.\n"
            },
            {
                "Question": "Thanks",
                "Answer": "Glad I could help!\n"
            },
            {
                "Question": "Thank you",
                "Answer": "Glad I could help!\n"
            }
        ]
        for each_question in pre_built:
            if each_question['Question'].lower() in text.lower():
                return each_question['Answer']


    def dump_corpus(self):
        question_stack = []
        for each_item in CORPUS:
            question_stack.append(each_item['Question'])
        return question_stack


class Event:

    def __init__(self, kind, text):
        self.kind = kind
        self.CONFIRMATIONS = ["yes", "sure", "okay","ok", "that would be nice", "yep","of course"]
        self.NEGATIONS     = ["no", "don't", "dont", "nope","of course no"]
        self.original_text = text

    def handle_response(self, text, bot):
        if self.kind == "corpus_dump":
            return self.corpus_dump(text, bot)

    def corpus_dump(self, text, bot):
        for each_confirmation in self.CONFIRMATIONS:
            for each_word in text.split(" "):
                if each_confirmation.lower() == each_word.lower():
                    corpus = bot.dump_corpus()
                    corpus = ["-" + s for s in corpus]
                    answer= "%s%s%s" % ("\n", "\n".join(corpus), "\n")
        for each_negation in self.NEGATIONS:
            for each_word in text.split(" "):
                if each_negation.lower() == each_word.lower():
                    answer = "Feel free to ask another question or send an email to %s.\n" % (bot.settings['help_email'])
                    
        # base case, no confirmation or negation found
        if not answer:
            answer = "I'm having trouble understanding what you are saying. At the time, my ability is quite limited, " \
            "please refer to %s or email %s if I was not able to answer your question. " % (bot.settings['faq_page'],
                                                                                            bot.settings['help_email'])
        return answer
