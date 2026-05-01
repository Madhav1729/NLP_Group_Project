from util import *

# Add your import statements here
import nltk
from nltk.corpus import stopwords
from collections import Counter


class StopwordRemoval():

    def __init__(self):
        try:
            self.nltk_stopwords = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.nltk_stopwords = set(stopwords.words('english'))
            
        self.custom_stopwords = set()

    def fromList(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence with stopwords removed
        """

        stopwordRemovedText = []

        #Fill in code here
        for sentence in text:
            filtered_sentence = [
                token for token in sentence 
                if token.lower() not in self.nltk_stopwords
            ]
            stopwordRemovedText.append(filtered_sentence)

        return stopwordRemovedText


    def buildCorpusStopwords(self, text, k=100):
        """
        Build a stopword list using corpus frequency

        Parameters
        ----------
        text : list
            A list of lists of tokens

        k : int
            Number of most frequent words to treat as stopwords

        Returns
        -------
        set
            Set of corpus-based stopwords
        """
        word_counts = Counter()

        for sentence in text:
            for token in sentence:
                word_counts[token.lower()] += 1

        # select top-k frequent words
        self.custom_stopwords = set(
            word for word, _ in word_counts.most_common(k)
        )

        return self.custom_stopwords