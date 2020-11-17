import torchtext
from torchtext import data
import spacy
from torchtext.datasets import WikiText2
from spacy.symbols import ORTH


my_tok = spacy.load('en')
def spacy_tok(x):
    return [tok.text for tok in my_tok.tokenizer(x)]
TEXT = data.Field(lower=True, tokenize=spacy_tok)

train, valid, test = WikiText2.splits(TEXT) # loading custom datasets requires passing in the field, but nothing else.
TEXT.build_vocab(train, vectors="glove.6B.200d")

train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
    (train, valid, test),
    batch_size=32,
    bptt_len=30, # this is where we specify the sequence length
    device=0,
    repeat=False)
