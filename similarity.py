import spacy
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import gensim
import numpy as np
import re, math
from collections import Counter

from nlp_pre import preprocess


data=pd.read_csv("./data/data_cln.csv")
CORPUS=data.to_dict('records')

class Semantic_similarity:
    
    def __init__(self):
        # A larger model should be used for better accuracy
        self.nlp = spacy.load('en_core_web_sm') #median
                                           #small: en_core_web_sm
                                           #large:en_core_web_lg
        self.q_docs = [self.nlp(str(entry)) for entry in data.Question]     

    def get_cos_tfidf_simularity(self,question):
        """Returns (index, similarity) of argument q's most similar match in argument docs, all spaCy documents."""
        print("--------------------",question)
        q = self.nlp(question)
        max_i = 0
        max_s = 0
        ms = []
        for i, d in enumerate(self.q_docs):
            if d.similarity(q) > max_s:
                max_s = d.similarity(q)
                max_i = i

        return max_s,max_i    
		
		

class Cosine_Tfidf_similarity:
    
    def __init__(self):
        self.file_docs  = self.prepare_all()

        self.dictionary = gensim.corpora.Dictionary(self.file_docs)
        self.BOW       = [self.dictionary.doc2bow(gen_doc) for gen_doc in self.file_docs]
        self.tf_idf     = gensim.models.TfidfModel(self.BOW)
        self.sims       = gensim.similarities.Similarity('.',self.tf_idf[self.BOW], num_features=len(self.dictionary))
    
    def prepare_all(self):
        file_docs = []
        for q in data.Question:
            file_docs.append(preprocess(q))
        return file_docs
    
    def visualize(self):
        print("Dictionnary\n")
        print(self.dictionary.token2id)
        
        print("\nCorpus\n")
        print(self.BOW)
        
        print("\Tf-idf\n")
        for doc in self.tf_idf[self.BOW]:
            print([[self.dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])
    
    def get_cos_tfidf_simularity(self,word):
        query_doc_bow = self.dictionary.doc2bow(word)
        query_doc_tf_idf = self.tf_idf[query_doc_bow]
        simularity=self.sims[query_doc_tf_idf]
        #print('Comparing Result:', sims[query_doc_tf_idf]) 
        
        max_s=max(simularity)
        max_i=[i for i, j in enumerate(simularity) if j == max_s]
        
        return max_s,max_i[0]
        
    def cosine_similarity(self,vec1, vec2):
        intersection = set(vec1.keys()).union(set(vec2.keys()))
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        

        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator
        
        

def find_most_similar(word,sim,cos):
    cosine   = {"answer": None, "score": 0, "question": None}
    cosTfidf = {"answer": None, "score": 0, "question": None}
    semantic = {"answer": None, "score": 0, "question": None}

    #Semantic similarity
    semantic['score'],index =sim.get_cos_tfidf_simularity(word)
    semantic['question']= CORPUS[index]['Question']
    semantic['answer']= CORPUS[index]['Answer']

    word=preprocess(word)
    #Cosine similarity 
    for each in CORPUS:
        question= preprocess(each['Question'])
        score = cos.cosine_similarity(Counter(word), Counter(question))
        if score > cosine['score']:
            cosine['score'] = score
            cosine['answer'] = each['Answer']
            cosine['question'] = each['Question']
    
    #Cosine similarity with tf-idf
    cosTfidf['score'],index = cos.get_cos_tfidf_simularity(word)
    cosTfidf['question']= CORPUS[index]['Question']
    cosTfidf['answer']= CORPUS[index]['Answer']
    

        
    if (semantic['score']>cosine['score']) & (semantic['score']>cosTfidf['score']):
        return {"method":"Semantic","score": semantic['score'], "answer": semantic['answer'], "question": semantic['question']}
    elif (cosine['score']>semantic['score']) & (cosine['score']>cosTfidf['score']):
        return {"method":"cosine","score": cosine['score'], "answer": cosine['answer'], "question": cosine['question']}
    else:
        return {"method":"cosine-Tfidf","score": cosTfidf['score'], "answer": cosTfidf['answer'], "question": cosTfidf['question']}