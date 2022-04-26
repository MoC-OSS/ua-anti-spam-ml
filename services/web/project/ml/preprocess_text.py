import re
import simplemma
from simplemma import simple_tokenizer

from .src.translit_convertor import ru_translit as translit
from .src.translit_convertor import special_characters as spec_ch

from .src.ua_corpus.UACorpus import StopWords
from .src.ua_stemmer.UAStemmer import UAStemmer 


ua_stop_words_list = StopWords().stopwords_list

stemmer = UAStemmer()
langdata = simplemma.load_data('uk')


def skip_emails(text):
    # skip emails and part of emails (like: name@email.com, @email.ru .. )
    emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text)
    for email in emails:
        text = text.replace(email, '')
    return text

def skip_mentions(text):
    mentions = re.findall(r'\B@[._a-zA-Z0-9]{1,}', text)
    for m in mentions:
        text = text.replace(m, '')
    return text

def skip_custom_character(tokens):
    custom_character = ['«', '»', '_', '_', '[', ']']
    new_tokens = []
    for el in tokens:
        if el not in custom_character:
            new_tokens.append(el)
    return new_tokens

def skip_double_letters(tokens):
    new_tokens = []
    for token in tokens:
        prev_symb = ''
        new_token_part = ''
        for letter in token:
            if letter != prev_symb:
                new_token_part = new_token_part+letter
            prev_symb = letter
        new_tokens.append(new_token_part)
    return new_tokens

def skip_stopwords(tokens):
    # must be use before Tokenizer
    review_without_sw = []
    for word in tokens:
            if word not in ua_stop_words_list:
                review_without_sw.append(word)
    return review_without_sw

def only_charters(text):
    s = re.sub('[^[аa-яіїєzАA-ЯІЇЄZ]', ' ', text)
    s = re.sub("\s\s+" , " ", s)
    return s.rstrip().lstrip()

def skip_urls(text):
    s = re.sub(r'http\S+', '', text)
    return  s.rstrip().lstrip()


def preproc_text(text_str, steamm=False, lemm=False):
    '''Main func for preprocessing text'''
    review = text_str.lower()

    review = skip_emails(review)
    review = skip_mentions(review)
    review = skip_urls(review)

    # replacing special symbols (like ₽, @, 0, € etc..)
    review = spec_ch.replace_specsymb(review)
    # additional checking for eng symbols
    review = translit.cyrillic_to_ru(review)

    review = only_charters(review)

    tokens = simple_tokenizer(review)
    review = skip_stopwords(tokens)

    if lemm:
        review = [simplemma.lemmatize(word, langdata) for word in review]  

    if steamm:
        review = [stemmer.stem_word(word) for word in review]

    review = skip_custom_character(review)
    review = skip_double_letters(review)

    review = ' '.join(review)

    return review