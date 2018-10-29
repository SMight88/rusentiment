import re

from nltk import tokenize, TweetTokenizer


class TextPreprocess():

    def tokenize_only(self, text):
        tokenizer = TweetTokenizer()
        text = re.sub(r'#', '# ', text)
        tokens = tokenizer.tokenize(text)
        return tokens

    def tokenize_with_lower(self, text):
        return [word.lower() for word in tokenize.WordPunctTokenizer().tokenize(text)]
