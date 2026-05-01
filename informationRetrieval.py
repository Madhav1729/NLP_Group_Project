import math
from collections import defaultdict, Counter
from util import *

# Add your import statements here


class InformationRetrieval():

    def __init__(self):
        self.index = None
        self.idf = {}
        self.doc_vectors = [] 
        self.doc_norms = []
        self.doc_ids = []

    def buildIndex(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document[cite: 2]
        arg2 : list
            A list of integers denoting IDs of the documents[cite: 2]
        Returns
        -------
        None
        """

        index = None

        # Fill in code here
        N = len(docs)
        self.doc_ids = docIDs

        doc_term_freqs = []
        df = defaultdict(int)

        # 1. Calculate Term Frequencies (TF) and Document Frequencies (DF)
        for doc in docs:
            # Flatten the sentences into a single list of words for the document
            terms = [term for sentence in doc for term in sentence]
            tf = Counter(terms)
            doc_term_freqs.append(tf)
            
            # Increment document frequency for each unique term in this document
            for term in tf.keys():
                df[term] += 1

        # 2. Calculate Inverse Document Frequency (IDF)
        self.idf = {term: math.log10(N / df[term]) for term in df}

        self.doc_vectors = []
        self.doc_norms = []

        # 3. Construct TF-IDF representations and calculate document vector norms
        for tf in doc_term_freqs:
            vector = {}
            norm_sq = 0
            for term, count in tf.items():
                tfidf = count * self.idf[term]
                vector[term] = tfidf
                norm_sq += tfidf ** 2
            
            self.doc_vectors.append(vector)
            # Store the L2 norm for cosine similarity normalization later
            self.doc_norms.append(math.sqrt(norm_sq))

        # Store the computed structures in the index variable
        index = {
            "idf": self.idf,
            "doc_vectors": self.doc_vectors,
            "doc_norms": self.doc_norms,
            "doc_ids": self.doc_ids
        }

        self.index = index


    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query[cite: 2]
        

        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query[cite: 2]
        """

        doc_IDs_ordered = []

        # Fill in code here
        for query in queries:
            # Flatten the query sentences into a list of words
            terms = [term for sentence in query for term in sentence]
            query_tf = Counter(terms)

            query_vector = {}
            query_norm_sq = 0
            
            # 1. Construct the Query TF-IDF Vector
            for term, count in query_tf.items():
                if term in self.idf: # Ignore Out-Of-Vocabulary (OOV) terms
                    tfidf = count * self.idf[term]
                    query_vector[term] = tfidf
                    query_norm_sq += tfidf ** 2

            query_norm = math.sqrt(query_norm_sq)

            scores = []
            
            # 2. Compute Cosine Similarity between the query and each document
            for idx, doc_vector in enumerate(self.doc_vectors):
                doc_norm = self.doc_norms[idx]
                
                # Handle edge cases where vectors are empty
                if query_norm == 0 or doc_norm == 0:
                    score = 0.0
                else:
                    dot_product = sum(query_vector.get(term, 0) * doc_vector.get(term, 0) for term in query_vector)
                    score = dot_product / (query_norm * doc_norm)
                
                scores.append((score, self.doc_ids[idx]))

            # 3. Sort document IDs based on their cosine similarity score in descending order
            scores.sort(key=lambda x: x[0], reverse=True)
            ranked_ids = [doc_id for score, doc_id in scores]
            
            doc_IDs_ordered.append(ranked_ids)

        return doc_IDs_ordered