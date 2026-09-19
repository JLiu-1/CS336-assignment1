#import regex as re


from tests.parallel_tokenizer import parallel_pretokenize
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
path = 'data/TinyStoriesV2-GPT4-train.txt'
pretok = parallel_pretokenize(path, pattern=PAT, num_workers=8, special_tokens = ['<|endoftext|>'])
def to_tuple(token):
    return tuple(bytes([b]) for b in token)
counts = {}
for token, count in pretok.items():
    counts[to_tuple(token)] = count
mergecount = {}
vocab = {}
for i in range(256):
    vocab[i] = bytes([i])
vocab [256] = '<|endoftext|>'
vocab_size = len(vocab)


#for idx,word in vocab.items():
#    print(f"{idx}: {word}")
max_vocab_size = 10000
#mergecount = {}
while vocab_size < max_vocab_size:
    mergecount = {}
    for byte_tuple, count in counts.items():
        for i in range(1, len(byte_tuple)):
            pair = (byte_tuple[i-1], byte_tuple[i])
            if pair not in mergecount:
                mergecount[pair] = 0
            mergecount[pair] += count
    best_pair = max(mergecount, key=mergecount.get)
    
    vocab[vocab_size] = best_pair[0] + best_pair[1]
    vocab_size += 1

    for byte_tuple, count in list(counts.items()):
        to_upd = False
        for i in range(1, len(byte_tuple)):
            pair = (byte_tuple[i-1], byte_tuple[i])
            if pair == best_pair:
                to_upd = True
                break
        if to_upd:
            new_tuple = []
            i = 0
            while i < len(byte_tuple):
                if i < len(byte_tuple) - 1 and (byte_tuple[i], byte_tuple[i+1]) == best_pair:
                    new_tuple.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_tuple.append(byte_tuple[i])
                    i += 1
            counts[tuple(new_tuple)] = counts.get(tuple(new_tuple), 0) + count
            del counts[byte_tuple]

print(vocab)




    
