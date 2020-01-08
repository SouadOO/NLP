import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS

#Convert text to lower case
def To_lower(txt):
    return txt.lower()

#Remove numbers
def Del_number(txt):
    return re.sub(r'\d+', '', txt)

#Remove punctuation
def Del_ponct(txt):
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" #+ '""“”’' + "®?•à-p‘?´°£€\×™—–&"
    return txt.translate(str.maketrans('', '', punct))

#Remove whitespace
def Del_whitespace(txt):
    return txt.strip()

#Tokenize the text and remove all stop words
def Tokenizer_del_StopWords(txt,stop_w=True):
    tokens    = word_tokenize(txt)
    if stop_w:
        stop_words= set(STOP_WORDS)
        return [i for i in tokens if not i in stop_words]
    else:
        return tokens

#Stemming
def stemming(list_words):
    stemmer= PorterStemmer()
    return [stemmer.stem(word) for word in list_words]

#Lemmatization
def lemming(list_words):
    lem=WordNetLemmatizer()
    return [lem.lemmatize(word) for word in list_words]



def preprocess(text,lower=True,c_num=True,c_pon=True,c_space=True,c_stop_w=True,stem=False,lemm=True):
    
    if lower:
        text  = To_lower  (text)                   # To lower case
    if c_num:
        text  = Del_number(text)                   # Delete  numbers
    if c_pon:
        text  = Del_ponct (text)                   # Delete ponctuations
    if c_space:
        text  = Del_whitespace(text)               # Delete white space 
    
    words = Tokenizer_del_StopWords(text,c_stop_w)      # Split the sentence to words and remove the stop words
    
    if lemm:
        words = lemming(words)                     # Reducing words to their word lemm
    else:
        words = stemming(words)                     # Reducing words to their word stem
        
    return words