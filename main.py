import torch
import torch.utils.data
import torchtext
from torchtext import data
import spacy
from torchtext.datasets import WikiText2
from spacy.symbols import ORTH
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as V

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


# Define model taken from PyTorch examples
class RNNModel(nn.Module):
    def __init__(self, ntoken, ninp,
                 nhid, nlayers, bsz,
                 dropout=0.5, tie_weights=True):
        super(RNNModel, self).__init__()
        self.nhid, self.nlayers, self.bsz = nhid, nlayers, bsz
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()
        self.hidden = self.init_hidden(bsz) # the input is a batched consecutive corpus
                                            # therefore, we retain the hidden state across batches

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        emb = self.drop(self.encoder(input))
        output, self.hidden = self.rnn(emb, self.hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (V(weight.new(self.nlayers, bsz, self.nhid).zero_().cuda()),
                V(weight.new(self.nlayers, bsz, self.nhid).zero_()).cuda())

    def reset_history(self):
        self.hidden = tuple(V(v.data) for v in self.hidden)



weight_matrix = TEXT.vocab.vectors
model = RNNModel(weight_matrix.size(0), weight_matrix.size(1), 200, 1, BATCH_SIZE)

model.encoder.weight.data.copy_(weight_matrix)
model.to(device)
