import torch
import torch.utils.data
import torchtext
from torchtext import data
import spacy
from torchtext.datasets import WikiText2,WMT14
from spacy.symbols import ORTH

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

my_tok = spacy.load('en')
def spacy_tok(x):
    return [tok.text for tok in my_tok.tokenizer(x)]
TEXT = data.Field(tokenize=spacy_tok)


Mydata = torchtext.datasets.LanguageModelingDataset(path = 'scpoems.txt',text_field=TEXT,newline_eos =False )
TEXT.build_vocab(Mydata)

loader = torchtext.data.BPTTIterator(dataset = Mydata,batch_size = 32,bptt_len = 30,shuffle = True,device = device)

batch = (next(iter(loader)))

print(getattr(batch,"text"))
print(getattr(batch,"target"))
