# Make a binary pair encoding (BPE) tokenizer
import sys
from typing import Optional
import regex as re
from collections import Counter

import pretokenization_example as pretokenization


def get_pair_counts(vocab: dict[tuple[bytes], int]) -> Counter:
    """Compute initial pair frequencies from vocab: {tuple: freq}"""
    pairs = Counter()
    for token, freq in vocab.items():
        for i in range(len(token) - 1):
            pairs[(token[i], token[i+1])] += freq
    return pairs


def bpe_train(
        pretokenized: dict[tuple[str], int],
        num_merges: int,
        vocab_size: int = 1000,
        special_tokens: Optional[list[bytes]] = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer.
    pretokenized: dict of pretokenized sequences to frequencies
    num_merges: number of BPE merges to perform
    vocab_size: desired vocabulary size
    special_tokens: list of special tokens to include in vocab
    Returns: final_vocab and list of merges (a, b) -> ab
    """
    # Current symbol sequences (mutable per merge)
    # We'll maintain a list of (symbols, freq) for easy update
    tokens = list(pretokenized.items())  # [(('l','o','w'), 5), ...]
    
    # Build initial pair frequencies
    pair_freq = get_pair_counts(pretokenized)
    
    merges = []
    
    for _ in range(num_merges):
        if not pair_freq:
            break
        # Get most frequent pair (handle ties: lexicographic order)
        # Since Counter.most_common() is stable, but we need lex tie-break
        best_pair = max(pair_freq.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(best_pair)
        a, b = best_pair
        
        # Remove the merged pair from counts
        del pair_freq[best_pair]
        
        # New symbol
        new_symbol = a + b  # if bytes/str; for tuples, could be (a, b) but usually string concat

        # Update each token that contains (a, b)
        new_tokens = []
        for symbols, freq in tokens:
            # Replace all non-overlapping (a, b) with new_symbol
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i+1] == b:
                    new_symbols.append(new_symbol)
                    i += 2  # skip both
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            # Only if merge happened, update pair_freq
            if len(new_symbols) != len(symbols):
                # Decrement old pairs that were removed
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i+1])
                    pair_freq[pair] -= freq
                    if pair_freq[pair] <= 0:
                        del pair_freq[pair]

                # Increment new pairs
                for i in range(len(new_symbols) - 1):
                    pair = (new_symbols[i], new_symbols[i+1])
                    pair_freq[pair] += freq
                
                new_tokens.append((tuple(new_symbols), freq))
            else:
                new_tokens.append((symbols, freq))
        tokens = new_tokens
    
    # Rebuild final vocab dict
    final_vocab = {k: token for k, token in enumerate(special_tokens)} if special_tokens else {}
    cur_len = len(final_vocab)
    for k in range(256):
        final_vocab[k + cur_len] = chr(k)
    cur_len = len(final_vocab)
    num_iters = min(vocab_size - cur_len, len(merges))
    for k in range(num_iters):
        final_vocab[k + cur_len] = "".join(merges[k]).encode('utf-8')

    return final_vocab, merges


if __name__ == "__main__":
    # Make args for filename, vocab size, special tokens
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        raise ValueError("Please provide a filename as argument.")
    if len(sys.argv) > 2:
        vocab_size = int(sys.argv[2])
    else:
        vocab_size = 1000
    if len(sys.argv) > 3:
        num_processes = int(sys.argv[3])
    else:
        num_processes = 4
    if len(sys.argv) > 4:
        special_tokens = sys.argv[4].encode('utf-8')
    else:
        special_tokens = '<|endoftext|>'.encode('utf-8')
    num_merges = vocab_size - 256 - 1 # assuming one special token
    pretokenized = pretokenization.pretokenize(
        filename, special_token=special_tokens, num_processes=num_processes
    )
    final_vocab, merges = bpe_train(
        pretokenized,
        num_merges=num_merges,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    print("Vocabulary size: ", len(final_vocab))
    print("First 10 merges: ", merges[:10])
