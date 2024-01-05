#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <tuple>
#include <utility>

using namespace std;

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;  
    }
};

typedef pair<int, int> Bigram;
typedef vector<vector<int>> Corpus;
typedef vector<vector<double>> Probs;
typedef unordered_map<Bigram, double, pair_hash> BigramWeights;

BigramWeights create_bigram_weights(const Corpus& corpus, const Probs& cur_token_probs) {
    BigramWeights bigram_weights;
    for (size_t s = 0; s < corpus.size(); ++s) {
        const auto& sentence = corpus[s];
        const auto& probs = cur_token_probs[s];
        for (size_t i = 0; i < sentence.size() - 1; ++i) {
            Bigram bigram(sentence[i], sentence[i + 1]);
            bigram_weights[bigram] += probs[i+1];
        }
    }
    return bigram_weights;
}

Bigram find_best_bigram(const BigramWeights& bigram_weights) {
    return max_element(bigram_weights.begin(), bigram_weights.end(),
                       [](const pair<Bigram, double>& a, const pair<Bigram, double>& b) {
                           return a.second < b.second;
                       })->first;
}

tuple<Corpus, Probs, int> merge_bigram_in_corpus(const Corpus& corpus, const Bigram& bigram, const Probs& cur_token_probs, int new_tok_ctr) {
    Corpus merged_corpus;
    Probs merged_cur_token_probs;

    for (size_t s = 0; s < corpus.size(); ++s) {
        vector<int> merged_sentence;
        vector<double> merged_probs;
        const auto& sentence = corpus[s];
        const auto& probs = cur_token_probs[s];

        for (size_t i = 0; i < sentence.size(); ) {
            if (i < sentence.size() - 1 && make_pair(sentence[i], sentence[i + 1]) == bigram) {
                merged_sentence.push_back(new_tok_ctr);                
                merged_probs.push_back(probs[i]*probs[i + 1]);
                i += 2;
            } else {
                merged_sentence.push_back(sentence[i]);
                merged_probs.push_back(probs[i]);
                i++;
            }
        }

        merged_corpus.push_back(merged_sentence);
        merged_cur_token_probs.push_back(merged_probs);
    }

    return make_tuple(merged_corpus, merged_cur_token_probs, new_tok_ctr + 1);
}

void cpp_weighted_bpe(
    const Corpus& corpus, 
    const Probs& cur_token_probs, 
    Corpus& out_corpus, 
    Probs& out_cur_token_probs, 
    vector<Bigram>& out_new_vocab, int num_merges = 1000, int new_tok_ctr = 32000) 
    {
    out_corpus = corpus;
    out_cur_token_probs = cur_token_probs;

    for (int i = 0; i < num_merges; ++i) {
        BigramWeights bigram_weights = create_bigram_weights(out_corpus, out_cur_token_probs);
        Bigram best_bigram = find_best_bigram(bigram_weights);
        tie(out_corpus, out_cur_token_probs, new_tok_ctr) = merge_bigram_in_corpus(out_corpus, best_bigram, out_cur_token_probs, new_tok_ctr);
        out_new_vocab.push_back(best_bigram);
    }
}

// int main() {
//     Corpus corpus = {{1, 2, 3, 4, 5, 6}};
//     Probs cur_token_probs = {{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}};
//     Corpus out_corpus;
//     Probs out_cur_token_probs;
//     vector<Bigram> out_new_vocab;
//     cpp_weighted_bpe(corpus, cur_token_probs, out_corpus, out_cur_token_probs, out_new_vocab, 2);
//     for (const auto& sentence : out_corpus) {
//         for (int tok : sentence) {
//             cout << tok << " ";
//         }
//         cout << endl;
//     }
//     for (const auto& probs : out_cur_token_probs) {
//         for (double prob : probs) {
//             cout << prob << " ";
//         }
//         cout << endl;
//     }
//     for (const auto& bigram : out_new_vocab) {
//         cout << bigram.first << " " << bigram.second << endl;
//     }
// }