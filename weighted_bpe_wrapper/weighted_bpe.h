#include <vector>
#include <tuple>

typedef std::pair<int, int> Bigram;
typedef std::vector<std::vector<int>> Corpus;
typedef std::vector<std::vector<double>> Probs;

void cpp_weighted_bpe(
    const Corpus& corpus, 
    const Probs& cur_token_probs, 
    Corpus& out_corpus, 
    Probs& out_cur_token_probs, 
    std::vector<Bigram>& out_new_vocab, int num_merges = 1, int new_tok_ctr = 32000
    );
