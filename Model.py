import pandas as pd
import numpy as np
import spacy
from gensim.models import Word2Vec
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
import pickle
import os

class NLP_model():

    def __init__(self):

        df = pd.read_csv('https://media.githubusercontent.com/media/alonsomoncayo/Project-2/main/Restaurants_unique.csv')
        # df['color'] = df['mean'].astype(int).apply(lambda x: (255,255,255) if x == 5 else ((173,255,47) if x == 4 else ((255,255,0) if x == 3 else ((255,140,0) if x == 2 else (255,0,0)))))
        df['color'] = df['mean'].astype(int).apply(lambda x: '#648FFF' if x == 5 else ('#785EF0' if x == 4 else ('#DC267F' if x == 3 else ('#FE6100' if x == 2 else '#FFB000'))))
        
        self.df = df

        self.top10 = self.df.sort_values(by=['mean', 'count'], ascending=[False, False]).head(10)
        l = list(self.top10['categories'])
        self.list_cat = list(dict.fromkeys(l))
        self.top10words = pd.DataFrame([('food', 122), ('place', 87), ('delicious', 87), ('great', 86), ('good', 70), ('best', 60), ('tea', 57), ('coffee', 52), ('amazing', 49), ('new', 48), ('time', 47), ('beth', 47), ('try', 45), ('definitely', 45), ('love', 44), ('recommend', 44), ('friendly', 43), ('like', 40), ('chocolate', 37), ('fresh', 35), ('service', 35), ('pig', 35), ('got', 34), ('highly', 32), ('shop', 32), ('family', 32), ('little', 30), ('menu', 29), ('spot', 29), ('ordered', 28), ('looking', 27), ('nice', 26), ('perfect', 26), ('home', 26), ('lily', 26), ('work', 24), ('persian', 24), ('flavors', 24), ('eating', 24), ('going', 23), ('wonderful', 23), ('came', 22), ('right', 22), ('chicken', 22), ('grateful', 22), ('tacos', 22), ('eat', 21), ('owner', 21), ('day', 21), ('loved', 21)],columns=['Word','Freq'])

        self.topic_names = {0: u'general',
               1: u'customer service',
               2: u'italian',
               3: u'vegetarian',
               4: u'good service',
               5: u'ice cream',
               6: u'pizza',
               7: u'delivery service',
               8: u'various',
               9: u'bar',
               10: u'burger',
               11: u'dessert',
               12: u'chinese',
               13: u'family',
               14: u'greek',
               15: u'italian',
               16: u'salad',
               17: u'fresh',
               18: u'dessert',
               19: u'like/taste',
               20: u'sweet',
               21: u'birthday desserts',
               22: u'seafood',
               23: u'steak',
               24: u'donoughts',
               25: u'good',
               26: u'wings',
               27: u'noodles',
               28: u'locaton & time',
               29: u'flavour',
               30: u'wine & dine',
               31: u'breakfast',
               32: u'noodle',
               33: u'vietnamise',
               34: u'amazing',
               35: u'meals & apetizers',
               36: u'special events',
               37: u'price / rating',
               38: u'coffee',
               39: u'price',
               40: u'drink',
               41: u'mexican',
               42: u'indian',
               43: u'reservation',
               44: u'bathroom',
               45: u'bar',
               46: u'kids',
               47: u'italian',
               48: u'breakfast',
               49: u'various'}

        word2vec_file_path = 'word2vec_model_all'
        self.food2vec = Word2Vec.load(word2vec_file_path)
        self.food2vec.init_sims()

        self.nlp = spacy.load('en_core_web_sm')
        
        bigram_model_filepath = os.path.join('bigram_model_all')
        self.bigram_model = Phrases.load(bigram_model_filepath)

        trigram_model_filepath = os.path.join('trigram_model_all')
        self.trigram_model = Phrases.load(trigram_model_filepath)

        trigram_dictionary_filepath = os.path.join('trigram_dict_all.dict')
        self.trigram_dictionary = Dictionary.load(trigram_dictionary_filepath)

        lda_model_filepath = os.path.join('lda_model_all')
        self.lda = LdaMulticore.load(lda_model_filepath)

        sentiment_model_filepath = os.path.join('Sentiment_model.pkl')
        with open(sentiment_model_filepath, 'rb') as f:
            self.sentiment_model = pickle.load(f)
        f.close()

        TfidVectorizer_model_filepath = os.path.join('TfidVectorizer_model.pkl')
        with open(TfidVectorizer_model_filepath, 'rb') as f:
            self.verctorizer = pickle.load(f)
        f.close()
    
    def get_related_terms(self, token, topn=15,):
        similarities = []
        for word, similarity in self.food2vec.wv.most_similar(positive=[token],topn=topn):
            similarities.append([word,round(similarity,3)])
        return pd.DataFrame(similarities,columns=['similar word','Freq'])
    
    def punct_space(self,token):
        return token.is_punct or token.is_space
    
    def prepare_text(self,text):

        if isinstance(text,list):
            text = ' '.join(text)
        parsed_review = self.nlp(text)
        unigram_review = [token.lemma_ for token in parsed_review
                        if not self.punct_space(token)]
        bigram_review = self.bigram_model[unigram_review]
        trigram_review = self.trigram_model[bigram_review]
        trigram_review = [term for term in trigram_review
                        if not term in spacy.lang.en.stop_words.STOP_WORDS]
        return trigram_review
    
    def lda_description(self, text, min_topic_freq=0.05):
        trigram_review = self.prepare_text(text)
        review_bow = self.trigram_dictionary.doc2bow(trigram_review)
        review_lda = self.lda[review_bow]
        review_lda = sorted(review_lda, key=lambda x: -x[1])

        topics = []
        for topic_number, freq in review_lda:
            if freq < min_topic_freq:
                break
            topics.append([self.topic_names[topic_number], round(freq, 3)])
        weights = np.round([x[1] for x in topics]/sum([x[1] for x in topics]),2)
        for i in range(len(topics)):
            topics[i][1] = weights[i]
        return topics
    
    def convert_to_TfidfVectorizer(self,text):
        text_tfidf = self.verctorizer.transform(text)
        return text_tfidf
    
    def sentiment_analysis(self,opinions,obj_type):
        if obj_type == 'String':
            procesessd_text = self.prepare_text(opinions)
            processed_text_string = ' '.join(procesessd_text)
            text_tfidf = self.convert_to_TfidfVectorizer([processed_text_string])
            pred = self.sentiment_model.predict(text_tfidf)
            proba = self.sentiment_model.predict_proba(text_tfidf)
            return pred, proba
        else:
            prepared_df = opinions.map(self.prepare_text)
            prepared_df_string = prepared_df.map(lambda x: ' '.join(x))
            df_tfidf = self.convert_to_TfidfVectorizer(prepared_df_string['Reviews'])
            pred = self.sentiment_model.predict(df_tfidf)
            proba = self.sentiment_model.predict_proba(df_tfidf)
            return pred, proba
    def word_algebra(self,add=[], subtract=[], topn=2):
        answers = self.food2vec.wv.most_similar(positive=add, negative=subtract, topn=topn)
        terms = []
        for term, similarity in answers:
            terms.append(term)
        return terms