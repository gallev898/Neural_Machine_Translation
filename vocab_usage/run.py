import argparse
import os
import random
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import spacy
import torch
from nltk.translate.bleu_score import corpus_bleu
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
from dotproduct_models import Seq2Seq, Decoder, Encoder, Attention

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--cosine', default=False, action='store_true')
args = parser.parse_args()


def calculate_test_correlation(data, src_field, trg_field, model, device, max_len=50):
    if not os.path.exists('results/{}'.format(args.model)):
        os.mkdir('results/{}'.format(args.model))
    f = open('results/{}/{}_test_correlation.txt'.format(args.model, args.model), 'w')

    trgs = []
    pred_trgs = []

    for datum in data:
        src = vars(datum)['src']
        trg = vars(datum)['trg']

        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])

    f.write('{}\n'.format(pred_trgs[0]))
    f.write('{}\n'.format(pred_trgs[10]))
    f.write('{}\n'.format(pred_trgs[100]))

    test_translation_lst_str_words_counter = Counter([item for sublist in pred_trgs for item in sublist])

    word_to_index = trg_field.vocab.stoi
    words_representations = model.decoder.representations.t()

    freq_to_norm_to_word = []
    for word, freq in test_translation_lst_str_words_counter.items():
        if freq > 1000:
            continue
        word_rep_idx = word_to_index[word]
        rep = words_representations[word_rep_idx]
        rep_norm = torch.norm(rep).item()
        freq_to_norm_to_word.append((freq, rep_norm, word))

    sorted_lst = sorted(freq_to_norm_to_word, key=lambda x: x[0])

    freq_lst_by_model = [x[0] for x in sorted_lst]
    norm_lst_by_model = [x[1] for x in sorted_lst]

    plt.scatter(freq_lst_by_model, norm_lst_by_model, c='black', alpha=0.8)
    plt.title('reference')
    plt.xlabel('freq')
    plt.ylabel('norm')
    plt.savefig('results/{}/freq_vs_norm_by_model_at_test'.format(args.model))

    vocabulary_usage_model = set()
    vocab = len([vocabulary_usage_model.update(x) for x in pred_trgs])
    f.write('vocab: {}\n'.format(vocab))

    bleu = corpus_bleu([[x] for x in pred_trgs], [x[0] for x in trgs])
    f.write('bleu: {}\n'.format(bleu))

    test_spearmanr = scipy.stats.spearmanr(freq_lst_by_model, norm_lst_by_model)
    f.write('tset_spearmanr ref: {}\n'.format(test_spearmanr))


def calculate_train_correlation(data, src_field, trg_field, model, device, max_len=50):
    if not os.path.exists('results/{}'.format(args.model)):
        os.mkdir('results/{}'.format(args.model))
    f = open('results/{}/{}_train_correlation.txt'.format(args.model, args.model), 'w')

    all_train_trg_data = []
    for datum in data:
        trg = vars(datum)['trg']
        all_train_trg_data.append(trg)

    words_representations = model.decoder.representations.t()
    train_translation_lst_str_words_counter = Counter([item for sublist in all_train_trg_data for item in sublist])
    word_to_index = trg_field.vocab.stoi

    freq_to_norm_to_word = []
    for word, freq in train_translation_lst_str_words_counter.items():
        if freq > 1000:
            continue

        word_rep_idx = word_to_index[word]
        rep = words_representations[word_rep_idx]
        rep_norm = torch.norm(rep).item()
        freq_to_norm_to_word.append((freq, rep_norm, word))

    sorted_lst = sorted(freq_to_norm_to_word, key=lambda x: x[0])
    freq_lst = [x[0] for x in sorted_lst]
    norm_lst = [x[1] for x in sorted_lst]

    plt.clf()
    plt.scatter(freq_lst, norm_lst, c='pink', alpha=0.8)
    plt.title('reference')
    plt.xlabel('freq')
    plt.ylabel('norm')

    plt.savefig('results/{}/freq_vs_norm_train'.format(args.model))

    train_spearmanr = scipy.stats.spearmanr(freq_lst, norm_lst)
    f.write('train_spearmanr: {}\n'.format(train_spearmanr))
    f.close()


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(src_indexes)]).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)

    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            # output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask, args)

        attentions[i] = attention

        pred_token = output.argmax(1).item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attentions[:len(trg_tokens) - 1]


def get_embeddings(embedding_size, vocab_size, args):
    word2vec_dictionary = dict()
    for cls_idx in range(vocab_size):
        v = np.random.randint(low=-100, high=100, size=embedding_size)
        v = v / np.linalg.norm(v)

        word2vec_dictionary[cls_idx] = torch.from_numpy(v).float()

    w2v_matrix = torch.stack(list(word2vec_dictionary.values()), dim=1)
    return w2v_matrix


# SEED = 1234
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            include_lengths=True)

TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

print('before: train_data, valid_data, test_data')

data_path = '.data'
train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG), root=data_path)

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 1

train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
                                                                      batch_size=BATCH_SIZE,
                                                                      sort_within_batch=True,
                                                                      sort_key=lambda x: len(x.src),
                                                                      device=device)

INPUT_DIM = len(SRC.vocab) - 2
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

representations = get_embeddings(512, 5893, args)

requires_grad = False

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
# dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn, representations, requires_grad)

model = Seq2Seq(enc, dec, SRC_PAD_IDX, device)
model.to(device)

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]



checkpoint = torch.load('/Users/gallevshalev/Desktop/trained_models/{}/BEST.pt'.format(args.model), map_location='cpu')
model.load_state_dict(checkpoint['model'])
calculate_test_correlation(test_data, SRC, TRG, model, device)
calculate_train_correlation(train_data, SRC, TRG, model, device)
