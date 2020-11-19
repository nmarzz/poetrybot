import torch
import torch.utils.data
import torchtext
from torchtext import data
import spacy
from torchtext.datasets import WikiText2
from spacy.symbols import ORTH

# Device: set up kwargs for future
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Define tokenizer and field (field will be deprecated soon)
my_tok = spacy.load('en')
def spacy_tok(x):
    return [tok.text for tok in my_tok.tokenizer(x)]
TEXT = data.Field(tokenize=spacy_tok)

# Define data and iterator
train = torchtext.datasets.LanguageModelingDataset(path = 'poems.txt',text_field=TEXT,newline_eos =False )
test = torchtext.datasets.LanguageModelingDataset(path = 'poems_test.txt',text_field=TEXT,newline_eos =False )

TEXT.build_vocab(train, vectors="glove.6B.200d")

train_iter, test_iter = data.BPTTIterator.splits((train, test),
    batch_size=32,
    bptt_len=30,
    device=device,
    repeat=False)
















    
