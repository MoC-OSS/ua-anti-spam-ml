# https://pypi.org/project/transliterate/
from transliterate import translit, get_translit_function


def ru_to_cyrillic(string):
    translit_ru = get_translit_function('ru')
    return translit_ru(string, reversed=True)

def cyrillic_to_ru(string):
    return translit(string, 'ru')