from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, spacy, re, etc.)
import string
import spacy
from nltk.tokenize import TreebankWordTokenizer

class Tokenization():
    
    def __init__(self):
        # Load the spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        # Initialize the Penn Treebank Tokenizer
        self.ptb = TreebankWordTokenizer()

    def naive(self, text):
        """
        Tokenization using a Naive Approach

        Parameters
        ----------
        arg1 : list
            A list of strings where each string is a single sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
        """

        tokenizedText = []

        # Fill in code here
        #First we can split by white spaces
        for sentence in text:
            raw_tokens = sentence.split()

        # Now we can remove leading and trailing punctuations from each token
            cleaned_tokens = [token.strip(string.punctuation) for token in raw_tokens]

        # remove any empty strings that might be left behind
            tokenizedText.append([token for token in cleaned_tokens if token])

        return tokenizedText


    def pennTreeBank(self, text):
        """
        Tokenization using the Penn Tree Bank Tokenizer

        Parameters
        ----------
        arg1 : list
            A list of strings where each string is a single sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
        """

        tokenizedText = []
        for sentence in text:
            tokenizedText.append(self.ptb.tokenize(sentence))
        return tokenizedText


    def spacyTokenizer(self, text):
        """
        Tokenization using spaCy

        Parameters
        ----------
        arg1 : list
            A list of strings where each string is a single sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
        """

        tokenizedText = []

        # Fill in code here
        for sentence in text:
            doc = self.nlp(sentence)
            tokenizedText.append([token.text for token in doc])

        return tokenizedText