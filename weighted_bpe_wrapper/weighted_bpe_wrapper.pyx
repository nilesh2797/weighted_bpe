# distutils: language = c++
# distutils: sources = weighted_bpe.cpp

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map

# Define the C++ structures and functions
cdef extern from "weighted_bpe.h":
    # Define the C++ types for Cython
    ctypedef pair[int, int] Bigram
    ctypedef vector[vector[int]] Corpus
    ctypedef vector[vector[double]] Probs
    ctypedef unordered_map[Bigram, double] BigramWeights

    # Declare your C++ functions
    void cpp_weighted_bpe(const Corpus&, const Probs&, Corpus&, Probs&, vector[Bigram]&, int, int)

# Define a Python wrapper for the C++ function
def weighted_bpe(corpus_list, probs_list, int num_merges=1000, int new_tok_ctr=32000):
    """
    Performs weighted byte pair encoding (BPE) on a given corpus.

    This function takes a corpus and probabilities as input and performs BPE
    based on the given number of merges. It outputs the modified corpus, updated
    probabilities, and the new vocabulary generated through the BPE process.

    Parameters:
    corpus_list (list of list of int): The input corpus, represented as a list of sentences, 
                                       where each sentence is a list of token IDs.
    probs_list (list of list of float): Probabilities of the current token given the prefix
                                            tokens in the sentence.
    num_merges (int): The number of merge operations to perform in BPE.
    new_tok_ctr (int): Starting index for new token identifiers.

    Returns:
    tuple: A tuple containing three elements:
           - Updated corpus after BPE merges (list of list of int)
           - Updated probabilities (list of list of float)
           - New vocabulary as a result of the merges (list of tuples)
    """
    # Calculate total number of tokens in the corpus
    total_tokens = sum([len(sentence) for sentence in corpus_list])
    if num_merges >= total_tokens:
        raise ValueError("num_merges cannot be greater than the total number of tokens in the corpus")

    cdef Corpus corpus = Corpus()
    cdef Probs probs = Probs()
    cdef Corpus out_corpus
    cdef Probs out_probs
    cdef vector[Bigram] out_new_vocab

    # Convert Python lists to C++ vectors
    cdef vector[int] temp_sentence
    cdef vector[double] temp_probs

    for py_sentence in corpus_list:
        temp_sentence.clear()
        for word in py_sentence:
            temp_sentence.push_back(word)
        corpus.push_back(temp_sentence)
    
    for py_probs in probs_list:
        temp_probs.clear()
        for prob in py_probs:
            temp_probs.push_back(prob)
        probs.push_back(temp_probs)

    # Call the C++ function
    cpp_weighted_bpe(corpus, probs, out_corpus, out_probs, out_new_vocab, num_merges, new_tok_ctr)

    # Convert the results back to Python lists
    python_out_corpus = [[token for token in sentence] for sentence in out_corpus]
    python_out_probs = [[prob for prob in _probs] for _probs in out_probs]
    python_out_new_vocab = [(bigram.first, bigram.second) for bigram in out_new_vocab]

    return python_out_corpus, python_out_probs, python_out_new_vocab
