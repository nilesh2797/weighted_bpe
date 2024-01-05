# Weighted BPE

## Description

The `weighted_bpe` package provides an efficient implementation of the Weighted Byte Pair Encoding (BPE) algorithm in Python, with core functionalities written in C++ for performance optimization.

## Installation

You can install the `weighted_bpe` package directly from GitHub using pip:

```bash
pip install git+https://github.com/nilesh2797/weighted_bpe.git
```
Or, clone the repository and install it manually:

```bash
git clone https://github.com/nilesh2797/weighted_bpe.git
cd weighted_bpe
pip install .
```

## Usage
Here is a basic example of how to use the weighted_bpe package:

```python
from weighted_bpe.weighted_bpe_wrapper import weighted_bpe

# Example corpus and log probabilities
corpus = [[1, 2, 3], [4, 5, 6]]
probs = [[1, 0.2, 0.3], [1, 0.4, 0.6]] # Note this denotes the probability of the current token given the previous tokens

# Perform BPE
out_corpus, out_probs, out_new_vocab = weighted_bpe(corpus, probs, num_merges=3, new_tok_ctr=32000)

# Output
print("Output Corpus:", out_corpus)
print("Output Probabilities:", out_probs)
print("Output New Vocabulary:", out_new_vocab)
```

## `weighted_bpe` Documentation
Parameters:
- corpus_list (list of list of int): The input corpus, represented as a list of sentences, where each sentence is a list of token IDs.
- probs_list (list of list of float): Probabilities of the current token given the prefix tokens in the sentence.
- num_merges (int): The number of merge operations to perform in BPE.
- new_tok_ctr (int): Starting index for new token identifiers.

Returns: A tuple containing three elements:
- Updated corpus after BPE merges (list of list of int)
- Updated probabilities (list of list of float)
- New vocabulary as a result of the merges (list of tuples)
