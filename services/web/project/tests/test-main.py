# for runn 'python test-main.py -v'

import unittest
import simplemma

import sys 
sys.path.append('..')

from ml import preprocess_text as pt
from ml.src.ua_stemmer.UAStemmer import UAStemmer 
from ml.src.translit_convertor import ru_translit as rutrans
from ml.src.translit_convertor import special_characters as spec_char

class TestPreprocessTextMethods(unittest.TestCase):

    def test_skip_emails(self):
        self.assertEqual(pt.skip_emails('test@test.com 123'), ' 123')
        self.assertEqual(pt.skip_emails('123 some.name@email.mt.gov 123'), '123  123')
        self.assertEqual(pt.skip_emails('123 @mention 123'), '123 @mention 123')

    def test_skip_custom_character(self):
        test_tokens_1 = ['«', '»', 'ма', 'уваг', 'більшіст']
        self.assertEqual(pt.skip_custom_character(test_tokens_1), ['ма', 'уваг', 'більшіст'])
        test_tokens_2 = ['[', 'вебкадем', ']', 'реальн', 'проект']
        self.assertEqual(pt.skip_custom_character(test_tokens_2), ['вебкадем', 'реальн', 'проект'])

    def test_skip_double_letters(self):
        test_tokens_1 = ['якоб', 'пораженны', 'пкр', 'нннннептун']
        self.assertEqual(pt.skip_double_letters(test_tokens_1), ['якоб', 'поражены', 'пкр', 'нептун'])
        test_tokens_2 = ['ууурааа', 'теперь', 'хорошооооооооооо']
        self.assertEqual(pt.skip_double_letters(test_tokens_2), ['ура', 'теперь', 'хорошо'])


    def test_skip_mentions(self):
        self.assertEqual(pt.skip_mentions('@test 123'), ' 123')
        self.assertEqual(pt.skip_mentions('123 @mention 123'), '123  123')
        self.assertEqual(pt.skip_mentions('@PitonWarZ_bot У нашого бота'), ' У нашого бота')

    def test_skip_stopwords(self):
        test_tokens_1 = ['а', 'там', 'були', 'і', 'інших']
        self.assertEqual(pt.skip_stopwords(test_tokens_1), ['а', 'там', 'і'])
        test_tokens_2 = ['також', 'холодно', 'але', 'все', 'всередині', 'добре']
        self.assertEqual(pt.skip_stopwords(test_tokens_2), ['холодно', 'добре'])
        test_tokens_3 = ['аби', 'я', 'побачив', 'це', 'я', 'би', 'здивувався']
        self.assertEqual(pt.skip_stopwords(test_tokens_3), ['аби', 'я', 'побачив', 'це', 'я', 'здивувався'])

    def test_only_charters(self):
        self.assertEqual(pt.only_charters(u'аби я побачив це, я би здивувався!!!  12  3 test@ 😝'), 'аби я побачив це я би здивувався test')

    def test_skip_urls(self):
        self.assertEqual(pt.skip_urls('https://test.com/go-here please visit my site'), 'please visit my site')
        self.assertEqual(pt.skip_urls('тут є багато цікавого https://somedomain.gov.com/go-here'), 'тут є багато цікавого')

    def test_lemma(self):
        langdata = simplemma.load_data('uk')
        # прям за ними летить повітряна кулька
        # res: "прям за вони летіти повітряний кульок"
        w1 = 'прям'
        self.assertEqual(simplemma.lemmatize(w1, langdata), 'прям')
        w2 = 'за'
        self.assertEqual(simplemma.lemmatize(w2, langdata), 'за')
        w3 = 'ними'
        self.assertEqual(simplemma.lemmatize(w3, langdata), 'вони')
        w4 = 'летить'
        self.assertEqual(simplemma.lemmatize(w4, langdata), 'летіти')
        w5 = 'повітряна'
        self.assertEqual(simplemma.lemmatize(w5, langdata), 'повітряний')
        w6 = 'кулька'
        self.assertEqual(simplemma.lemmatize(w6, langdata), 'кульок')

    def test_setemmer(self):
        stemmer = UAStemmer()
        # прям за вони летіти повітряний кульок
        # res: "прям за вон летіт повітрян кульок"
        w1 = 'прям'
        self.assertEqual(stemmer.stem_word(w1), 'прям')
        w2 = 'за'
        self.assertEqual(stemmer.stem_word(w2), 'за')
        w3 = 'вони'
        self.assertEqual(stemmer.stem_word(w3), 'вон')
        w4 = 'летіти'
        self.assertEqual(stemmer.stem_word(w4), 'летіт')
        w5 = 'повітряний'
        self.assertEqual(stemmer.stem_word(w5), 'повітрян')
        w6 = 'кульок'
        self.assertEqual(stemmer.stem_word(w6), 'кульок')

    def test_ru_translit(self):
        s1 = 'Взрывается как мини ядерка'
        self.assertEqual(rutrans.ru_to_cyrillic(s1), 'Vzryvaetsja kak mini jaderka')
        # reverse
        s2 = 'Vzryvaetsja kak mini jaderka'
        self.assertEqual(rutrans.cyrillic_to_ru(s2), 'Взрываеця как мини ядерка')

        s3 = 'Ці хлопці перші беруть на себе удар'
        self.assertEqual(rutrans.ru_to_cyrillic(s3), "Tsі hloptsі pershі berut' na sebe udar")

    def test_special_characters(self):
        s1 = 'Если их наша оттечественная t0₽π€да отправила рыб кормить'
        self.assertEqual(spec_char.replace_specsymb(s1), 
                        'Если их наша оттечественная tорпеда отправила рыб кормить')
        s2 = 'запускаєм снaряд на М0скву'
        self.assertEqual(spec_char.replace_specsymb(s2), 
                        'запускаєм снaряд на Москву')
        s3 = 'Бабакает как мини яде₽ка'
        self.assertEqual(spec_char.replace_specsymb(s3), 
                        'Бабакает как мини ядерка')


if __name__ == '__main__':
    unittest.main()