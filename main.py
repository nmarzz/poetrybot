import torch
import torchtext
from torchtext import data
import spacy
from torchtext.datasets import WikiText2,WMT14
from spacy.symbols import ORTH

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)

my_tok = spacy.load('en')
def spacy_tok(x):
    return [tok.text for tok in my_tok.tokenizer(x)]
SRC = data.Field(lower=True, tokenize=spacy_tok)
TRG = data.Field(lower=True, tokenize=spacy_tok)

train, valid, test = WMT14.splits(exts=('.de', '.en'),
                                           train='train.tok.clean.bpe.32000',
                                           validation='newstest2013.tok.bpe.32000',
                                           test='newstest2014.tok.bpe.32000',
                                           fields=(SRC, TRG))

# TEXT.build_vocab(train, vectors="glove.6B.200d")
#
# train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
#     (train, valid, test),
#     batch_size=32,
#     bptt_len=30, # this is where we specify the sequence length
#     device=device,
#     repeat=False)
