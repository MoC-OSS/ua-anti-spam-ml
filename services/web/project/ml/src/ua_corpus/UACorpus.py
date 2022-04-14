import os


class StopWords():
    stopwords_corpus_path = os.path.dirname(os.path.abspath(__file__)) + \
                            '/ua_dict/stopwords_list_v2.txt'
    stopwords_list = None

    def __init__(self,):
        self.stopwords_list = self.read_stopwords_corpus()

    def read_stopwords_corpus(self, ):
        with open(self.stopwords_corpus_path, encoding='utf8') as f:
            lines = f.read().splitlines()
            return lines

    