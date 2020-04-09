import torch
from torchtext.data import Field, BucketIterator
import spacy
import numpy as np
from torchtext.datasets import Multi30k

random_embeddings = 0
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

glove_embeddings = torch.load('glove_embeddings')

# TRG.vocab.stoi
glove_representations = torch.zeros((5893, 300))

for word, index in TRG.vocab.stoi.items():
    if word in glove_embeddings:
        glove_representations[index] = torch.tensor(glove_embeddings[word])
    else:
        glove_representations[index] = torch.tensor(np.random.randint(low=-100, high=100, size=300))
        random_embeddings += 1

print('random_embeddings: {}'.format(random_embeddings))
torch.save(glove_representations, 'glove_representations_for_NMT')
