from util import *

# Add your import statements here
import re
import nltk
import spacy
from nltk.tokenize import sent_tokenize


class SentenceSegmentation():

    def __init__(self):
        self.nlp = None

    def _get_spacy_pipeline(self):
        if self.nlp is not None:
            return self.nlp

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")

        return self.nlp

    def naive(self, text):
        """
        Sentence Segmentation using a Naive Approach

        Parameters
        ----------
        arg1 : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each string is a single sentence
        """
        segmentedText = []
        current_sentence = ""
        
        for index, char in enumerate(text):
            current_sentence += char
            
            if char in ['.', '!', '?']:   
                # Start looking ahead at the next character
                lookahead_index = index + 1
                count = 0
                # Keep moving forward as long as we are seeing spaces
                while lookahead_index < len(text) and text[lookahead_index] == ' ':
                    lookahead_index += 1
                    count = count + 1
                
                # Now check if the character we finally landed on is uppercase
                if count >= 1 and lookahead_index < len(text) and text[lookahead_index].isupper():
                    segmentedText.append(current_sentence.strip()) #strip so that any leading or trailing zeroes are removed.
                    current_sentence = ""
        
        if current_sentence.strip():
            segmentedText.append(current_sentence.strip())	#at the end if there is any sentence left, we need to add it
            
        return segmentedText

    def punkt(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each string is a single sentence
        """
        if not text or not text.strip():
            return []

        try:
            segmentedText = sent_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass
            segmentedText = sent_tokenize(text)

        segmentedText = [sentence.strip() for sentence in segmentedText if sentence.strip()]

        return segmentedText

    def spacySegmenter(self, text):
        """
        Sentence Segmentation using spaCy

        Parameters
        ----------
        arg1 : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each string is a single sentence
        """
        if not text or not text.strip():
            return []

        doc = self._get_spacy_pipeline()(text)
        segmentedText = [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]

        return segmentedText


