import json

from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
import nltk 

nltk.download('wordnet')
def run_assignment_analysis():
    print("Loading Cranfield dataset...")
    try:
        with open('cranfield/cran_docs.json', 'r') as f:
            docs = json.load(f)
    except FileNotFoundError:
        print("Error: Ensure 'cranfield/cran_docs.json' is in your directory.")
        return

    # Initialize modules
    segmenter = SentenceSegmentation()
    tokenizer = Tokenization()
    reducer = InflectionReduction()

    original_vocab = set()
    stemmed_vocab = set()
    lemmatized_vocab = set()
    
    stem_mappings = {}

    print("Processing documents")
    
    for i, doc in enumerate(docs):
        if i > 0 and i % 200 == 0:
            print(f"Processed {i}/{len(docs)} documents...")
            
        text = doc['body']
        
        # 1. Segment(returns list of sentences)
        sentences = segmenter.punkt(text)
        
        # 2. Tokenize
        tokenized_sentences = tokenizer.pennTreeBank(sentences)
            
        # 3. Reduce code
        # print(type(tokenized_sentences))
        # print(type(tokenized_sentences[0])) 
        # print(type(tokenized_sentences[0][0]))
        stemmed_sentences = reducer.porterStemmer(tokenized_sentences)
        lemmatized_sentences = reducer.wordnetLemmatizer(tokenized_sentences)
        
        # 4. Extract data
        for orig_sent, stem_sent, lem_sent in zip(tokenized_sentences, stemmed_sentences, lemmatized_sentences):
            for orig_word, stem_word, lem_word in zip(orig_sent, stem_sent, lem_sent):
                
                # Skip punctuation/numbers for the vocabulary count
                if not orig_word.isalpha():
                    continue
                
                orig_clean = orig_word.lower()
                stem_clean = stem_word.lower()
                lem_clean = lem_word.lower()
                
                original_vocab.add(orig_clean)
                stemmed_vocab.add(stem_clean)
                lemmatized_vocab.add(lem_clean)
                
                if stem_clean not in stem_mappings:
                    stem_mappings[stem_clean] = set()
                stem_mappings[stem_clean].add(orig_clean)

    print("\n" + "="*50)
    print("PART 1: CHANGES IN VOCABULARY SIZE")
    print("="*50)
    print(f"Original Tokens (Words): {len(original_vocab)}")
    print(f"Lemmatized Vocabulary:   {len(lemmatized_vocab)}")
    print(f"Stemmed Vocabulary:      {len(stemmed_vocab)}")

    print("\n" + "="*50)
    print("PART 2: EXAMPLES OF OVER-STEMMING IN DATA")
    print("="*50)
    target_stems = ['experi', 'univers', 'relat', 'gener', 'oper']
    for stem in target_stems:
        if stem in stem_mappings and len(stem_mappings[stem]) > 1:
            words = list(stem_mappings[stem])
            print(f"Stem '{stem}' incorrectly merges: {', '.join(words[:4])}")

    print("\n" + "="*50)
    print("PART 3: SEMANTIC PRESERVATION (Lemmatization)")
    print("="*50)
    
    # Passing these directly through your reducer to show the exact output
    target_words = ['matrices', 'vortices', 'analyses', 'axes']
    for word in target_words:
        # We wrap in [[word]] because your reducer expects a list of lists
        stemmed_output = reducer.porterStemmer([[word]])[0][0]
        lemmatized_output = reducer.wordnetLemmatizer([[word]])[0][0]
        print(f"Original: '{word.ljust(10)}' | Your Stemmer -> '{stemmed_output.ljust(8)}' | Your Lemmatizer -> '{lemmatized_output}'")

if __name__ == "__main__":
    run_assignment_analysis()