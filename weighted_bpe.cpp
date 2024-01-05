#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <tuple>
#include <utility>
#include <cassert>

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
typedef tuple<size_t, size_t, double> UpdateLoc;
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

void update_bigram_weights(
    const Corpus& corpus, 
    const Probs& cur_token_probs, 
    BigramWeights& bigram_weights, 
    const Bigram& merged_bigram, 
    const vector<UpdateLoc>& merged_positions,
    const int& merged_bigram_index) {
    // First, remove the weight of the merged bigram
    bigram_weights.erase(merged_bigram);


    // Then, update the weights for bigrams affected by the merge
    for (const auto& position : merged_positions) {
        int row = get<0>(position);
        int col = get<1>(position);
        double original_prob = get<2>(position);

        // Adjust bigrams before and after the merged position if within bounds
        // assuming a symbol sequence "A B C", if "B C" is merged, reduce the frequency of "A B"
        if (col > 0) {
            Bigram before_bigram(corpus[row][col - 1], get<0>(merged_bigram));
            Bigram new_before_bigram(corpus[row][col - 1], merged_bigram_index);
            bigram_weights[before_bigram] -= original_prob;
            bigram_weights[new_before_bigram] += cur_token_probs[row][col];
        }

        // assuming a symbol sequence "A B C D", if "B C" is merged, reduce the frequency of "C D".
        // however, skip this if the sequence is A B C B C, because the frequency of "C B" will be reduced by the previous code block
        if ((col < corpus[row].size() - 1) && (corpus[row][col + 1] != merged_bigram_index)) {
            Bigram after_bigram(get<1>(merged_bigram), corpus[row][col + 1]);
            Bigram new_after_bigram(merged_bigram_index, corpus[row][col + 1]);
            bigram_weights[after_bigram] -= cur_token_probs[row][col + 1];
            bigram_weights[new_after_bigram] += cur_token_probs[row][col + 1];
        }
    }
}

Bigram find_best_bigram(const BigramWeights& bigram_weights) {
    return max_element(bigram_weights.begin(), bigram_weights.end(),
                       [](const pair<Bigram, double>& a, const pair<Bigram, double>& b) {
                           return a.second < b.second;
                       })->first;
}

vector<UpdateLoc> merge_bigram_in_corpus(Corpus& corpus, Probs& cur_token_probs, const Bigram& bigram, int new_tok_ctr) {
    vector<UpdateLoc> merged_positions;

    for (size_t s = 0; s < corpus.size(); ++s) {
        auto& sentence = corpus[s];
        auto& probs = cur_token_probs[s];
        size_t write_index = 0;

        for (size_t i = 0; i < sentence.size(); ) {
            if (i < sentence.size() - 1 && make_pair(sentence[i], sentence[i + 1]) == bigram) {
                sentence[write_index] = new_tok_ctr;
                probs[write_index] = probs[i] * probs[i + 1];

                // Store the position where the bigram was merged
                merged_positions.emplace_back(s, write_index, probs[i]);

                i += 2;
            } else {
                if (write_index != i) {
                    sentence[write_index] = sentence[i];
                    probs[write_index] = probs[i];
                }
                i++;
            }
            write_index++;
        }

        // Resize the vectors to remove unused elements
        sentence.resize(write_index);
        probs.resize(write_index);
    }

    return merged_positions;
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
    BigramWeights bigram_weights = create_bigram_weights(out_corpus, out_cur_token_probs);

    for (int i = 0; i < num_merges; ++i) {
        Bigram best_bigram = find_best_bigram(bigram_weights);
        auto merged_positions = merge_bigram_in_corpus(out_corpus, out_cur_token_probs, best_bigram, new_tok_ctr);
        out_new_vocab.push_back(best_bigram);
        update_bigram_weights(out_corpus, out_cur_token_probs, bigram_weights, best_bigram, merged_positions, new_tok_ctr);
        new_tok_ctr++;
        
        // assert bigram_weights is same as create_bigram_weights
        // auto test_bigram_weights = create_bigram_weights(out_corpus, out_cur_token_probs);
        // for (const auto& bigram_weight : test_bigram_weights) {
        //     assert(bigram_weights[bigram_weight.first] == bigram_weight.second);
        // }
    }
}

int main() {
    // Corpus corpus = {{1,2,3}, {1,2,3}};
    // Probs cur_token_probs = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.58}};

    Corpus corpus;
    Probs cur_token_probs;

    Corpus out_corpus;
    Probs out_cur_token_probs;
    vector<Bigram> out_new_vocab;

    for (int i = 0; i < 1000; ++i) {
        vector<int> sentence;
        vector<double> probs;
        for (int j = 0; j < 100; ++j) {
            int rand_int = rand() % 100;
            double rand_prob = (rand() % 100) / 100.0;
            sentence.push_back(rand_int);
            probs.push_back(rand_prob);
        }
        corpus.push_back(sentence);
        cur_token_probs.push_back(probs);
    }

    cpp_weighted_bpe(corpus, cur_token_probs, out_corpus, out_cur_token_probs, out_new_vocab, 1000);
    for (const auto& sentence : out_corpus) {
        for (int tok : sentence) {
            cout << tok << " ";
        }
        cout << endl;
    }
    for (const auto& probs : out_cur_token_probs) {
        for (double prob : probs) {
            cout << prob << " ";
        }
        cout << endl;
    }
    for (const auto& bigram : out_new_vocab) {
        cout << bigram.first << " " << bigram.second << endl;
    }
}