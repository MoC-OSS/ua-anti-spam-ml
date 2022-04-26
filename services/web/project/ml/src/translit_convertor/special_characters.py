# https://pypi.org/project/transliterate/ with custom LanguagePack

from transliterate.discover import autodiscover
autodiscover()

from transliterate import translit, get_translit_function
from transliterate.base import TranslitLanguagePack, registry

class SpecSymbLanguagePack(TranslitLanguagePack):
    language_code = "specsymb"
    language_name = "SpecSymb"

    # Use this only in cases if a single character in source language 
    # shall be represented by more than one character 
    # in the target language
    pre_processor_mapping = {
        u"lj": u"љ",
        u"Lj": u"Љ",
        u"LJ": u"Љ",
        u"nj": u"њ",
        u"Nj": u"Њ",
        u"NJ": u"Њ",
        u"dž": u"џ",
        u"Dž": u"Џ",
        u"DŽ": u"Џ",
    }

    # ready to use in ru_translit.py
    #
    # mapping = (
    #     u"abcčćdđefghijklmnoprsštuvzžyABCČĆDĐEFGHIJKLMNOPRSŠTUVZŽY₽@0€$πωρθ",
    #     u"абцчћдђефгхијклмноррсштувзжуАБЦЧЋДЂЕФГХИЈКЛМНОПРСШТУВЗЖУраоеcпвро",
    # )

    # custom mapping for symbols and nums
    mapping = (
        u"0346₽@€$πωρθβηα",
        u"озчбраеcпвровна",
    )

registry.register(SpecSymbLanguagePack)


def replace_specsymb(text, rev=False):
    if rev:
        translit_spec = get_translit_function('specsymb')
        return translit_spec(text, reversed=True)
    return translit(text, 'specsymb')
