import re
import simplemma
from simplemma import simple_tokenizer

from .src.ua_corpus.UACorpus import StopWords
from .src.ua_stemmer.UAStemmer import UAStemmer 


ua_stop_words_list = StopWords().stopwords_list

stemmer = UAStemmer()
langdata = simplemma.load_data('uk', 'ru')


def preproc_text(text_str, steamm=False, lemm=False):
    review = re.sub('[^[аa-яіїєzАA-ЯІЇЄZ]', ' ', text_str)
    review = review.lower()

    tokens = simple_tokenizer(review)
    
    # without stopwords
    review_without_sw = []
    for word in tokens:
            if word not in ua_stop_words_list:
                review_without_sw.append(word)
    review = review_without_sw

    if lemm:
        review = [simplemma.lemmatize(word, langdata) for word in review]  

    if steamm:
        review = [stemmer.stem_word(word) for word in review]

    review = ' '.join(review)

    return review