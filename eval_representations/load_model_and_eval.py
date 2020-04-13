import torch
import argparse
import spacy
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
args = parser.parse_args()

checkpoint = torch.load('/Users/gallevshalev/Desktop/trained_models/{}/BEST.pt'.format(args.model), map_location='cpu')
representations = checkpoint['representations'].t()

spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]
def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)
SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            include_lengths=True)
train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

TRG.build_vocab(train_data, min_freq=2)

word_to_norm = {}
for word, index in TRG.vocab.stoi.items():
    norm = torch.norm(representations[index])
    word_to_norm[word] = norm
g = 0