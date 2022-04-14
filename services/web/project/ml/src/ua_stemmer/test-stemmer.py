import time
import re
from UAStemmer import UAStemmer


def main():
    stemmer = UAStemmer()
    test_string = 'Привіт,як твої  справи?  (це ж тест??) Зберігайте спокойствіе.Оплатіте ласка, зробіть погашення боргу/заборгованостей. ' + \
                  'Рефлексивного и тямущий'
    
    # test_string = 'Уже вдеся́те майорить Приморський бульвар синьо-жовтими барвами, і вже вкотре ми збираємось у самому серці Одеси, щоб помилуватися показом автентичного вбрання, написати диктант просто неба, концентруючи  нашу енергію й демонструючи всім як свою єдність, так і свою любов до рідного міста та своєї країни'

    start_time = time.time()
    prepare_test_string = test_string.lower()
    words = re.split(r'(\W)', prepare_test_string)
    words = [word for word in words if word != '']

    for i in range(len(words)):
        words[i] = stemmer.stem_word(words[i])

    stem_test_string = ''.join(words)
    work_time = time.time() - start_time

    print('Source: %s\nStemmed: %s\nTime: %.6f' % (test_string, stem_test_string, work_time))

    etalon_stem_test_string = 'привіт,як тво  справ?  (це ж тест??) зберігайт спокойств.оплатіт ласк, зробіт погашенн борг/заборгованост. ' + \
                              'рефлексивн и тямущ'
    if stem_test_string == etalon_stem_test_string:
        print('ALL OK')


if __name__ == '__main__':
    main()