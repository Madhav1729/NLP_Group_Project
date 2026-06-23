This folder contains the template code for a search engine application.

Project Structure
-----------------
Cranfield Dataset NLP/
│
├── cranfield/
│   ├── cran_docs.json
│   ├── cran_queries.json
│   ├── cran_qrels.json
│   └── README.txt
│
└── template_code_part1/
    ├── main.py
    ├── sentenceSegmentation.py
    ├── tokenization.py
    ├── inflectionReduction.py
    ├── stopwordRemoval.py
    └── util.py


Files Description
-----------------

main.py
    The main module that contains the outline of the Search Engine.
    It handles preprocessing of queries and documents.
    The output folder is automatically created if it does not exist.

sentenceSegmentation.py
    Implement sentence segmentation methods:
        - naive
        - punkt
        - (optional) spacySegmenter

tokenization.py
    Implement tokenization methods:
        - naive
        - Penn Treebank (ptb)
        - (optional) spacyTokenizer

inflectionReduction.py
    Implement:
        - Porter Stemmer
        - WordNet Lemmatizer

stopwordRemoval.py
    Implement stopword removal using NLTK or corpus-based methods.

util.py
    Optional helper functions.


How to Run
----------

From inside template_code_part1:

If cranfield folder is inside the same parent directory:

    python main.py -dataset ../cranfield

If cranfield folder is inside the same directory:

    python main.py -dataset cranfield

Custom Query Mode:

    python main.py -dataset ../cranfield -custom

When -custom flag is passed, the system will prompt for a query:

    Enter query below
    Papers on Aerodynamics

Output
------

The following files will be generated inside the output folder:

    segmented_queries.txt
    tokenized_queries.txt
    reduced_queries.txt
    stopword_removed_queries.txt

    segmented_docs.txt
    tokenized_docs.txt
    reduced_docs.txt
    stopword_removed_docs.txt


Dependencies
------------

Install required libraries:

    pip install nltk spacy

Download required NLTK resources:

    python -m nltk.downloader punkt
    python -m nltk.downloader stopwords
    python -m nltk.downloader wordnet

Download spaCy model:

    python -m spacy download en_core_web_sm


Notes
-----

- Trailing slash in dataset path is NOT required.
- Output folder is created automatically.
- All preprocessing steps follow:
    Sentence Segmentation → Tokenization → Inflection Reduction → Stopword Removal
