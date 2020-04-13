import argparse
import math
import os
import random
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score
# from torchtext.datasets import WMT14
from nltk.translate.bleu_score import corpus_bleu

from torchtext.datasets import Multi30k

from dotproduct_models_no_attention import Seq2Seq, Decoder, Encoder, Attention, init_weights

parser = argparse.ArgumentParser()
parser.add_argument('--fixed', default=False, action='store_true')
parser.add_argument('--normalized', default=False, action='store_true')
parser.add_argument('--cosine', default=False, action='store_true')
parser.add_argument('--runname', type=str)

parser.add_argument('--run_local', default=False, action='store_true')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

if not args.run_local:
    import wandb

    wandb.init(project="Neural_Machine_Translation", name=args.runname, dir='/yoav_stg/gshalev/wandb')


def calculate_bleu(data, src_field, trg_field, model, device, max_len=50):
    trgs = []
    pred_trgs = []

    for datum in data:
        src = vars(datum)['src']
        trg = vars(datum)['trg']

        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)

        # cut off <eos> token
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])
    return corpus_bleu([[x] for x in pred_trgs], [x[0] for x in trgs])


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


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        # if i > 0:  # TODELETE
        #     break
        start = time.time()
        src, src_len = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, src_len, trg, args)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time:.3f}\t'
                  'Loss {loss:.4f})\t'
                  .format(epoch, i, len(iterator),
                          batch_time=time.time() - start,
                          loss=loss.item()
                          ))

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            # if i > 0:  # TODELETE
            #     break
            start = time.time()
            src, src_len = batch.src
            trg = batch.trg

            output = model(src, src_len, trg, args, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

            if i % print_freq == 0:
                print('Eval: Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time:.3f}\t'
                      'Loss {loss:.4f})\t'
                      .format(epoch, i, len(iterator),
                              batch_time=time.time() - start,
                              loss=loss.item()
                              ))

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


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
            output, hidden, _ = model.decoder(trg_tensor, hidden, encoder_outputs, mask, args)

        # attentions[i] = attention

        pred_token = output.argmax(1).item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], None


def display_attention(sentence, translation, attention):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(1).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
                       rotation=45)
    ax.set_yticklabels([''] + translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def get_embeddings(embedding_size, vocab_size, args):
    word2vec_dictionary = dict()
    for cls_idx in range(vocab_size):
        if not args.fixed: #dotproduct
            if cls_idx == 0:
                print('NOTICE: embeddings range +-100 and normalized')
            v = np.random.randint(low=-100, high=100, size=embedding_size)
            v = v / np.linalg.norm(v)
        else:
            if args.normalized:
                if cls_idx == 0:
                    print('NOTICE: embeddings range +-100 and normalized for fixed')
                v = np.random.randint(low=-100, high=100, size=embedding_size)
                v = v / np.linalg.norm(v)
            else:
                if cls_idx == 0:
                    print('NOTICE: embeddings range +-20 and UNnormalized for fixed')
                v = np.random.randint(low=-20, high=20, size=embedding_size)

        word2vec_dictionary[cls_idx] = torch.from_numpy(v).float()

    w2v_matrix = torch.stack(list(word2vec_dictionary.values()), dim=1)
    return w2v_matrix

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        lr_befor_update = param_group['lr']
        param_group['lr'] = param_group['lr'] * shrink_factor
        print('LR before: {} LR after: {}'.format(lr_befor_update, param_group['lr']))
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


print_freq = 30
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

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

data_path = '.data' if args.run_local else '/yoav_stg/gshalev/semantic_labeling/Multi30k'
train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG), root=data_path)
print('completed: train_data, valid_data, test_data')

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = args.batch_size

print('befor: train_iterator, valid_iterator, test_iterator')
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
                                                                      batch_size=BATCH_SIZE,
                                                                      sort_within_batch=True,
                                                                      sort_key=lambda x: len(x.src),
                                                                      device=device)
print('completed: train_iterator, valid_iterator, test_iterator')

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

representations = get_embeddings(512, 5893, args)
if args.fixed:
    requires_grad = False
else:
    requires_grad = True

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn, representations, requires_grad)

model = Seq2Seq(enc, dec, SRC_PAD_IDX, device)
model.to(device)

#model.apply(init_weights)

print('The model has {count_parameters(model):,} trainable parameters')

# optimizer.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

if not args.run_local:
    wandb.watch(dec)

if args.run_local:
    save_dir = args.runname
else:
    save_dir = os.path.join('/yoav_stg/gshalev/neural_machine_translation', args.runname)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    print('created dir : {}'.format(save_dir))

# sec: start epoch
N_EPOCHS = args.epochs
CLIP = 1
best_valid_loss = float('inf')
epochs_since_improvement = 0
best_bleu_score_value = 0
epoch = 0

while True:
# for epoch in range(N_EPOCHS):
    if epochs_since_improvement == 20:
        print('break after : epochs_since_improvement == 20')
        break
    if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
        print('!!!  ADJUST LR AFTER : epochs_since_improvement: {}'.format(epochs_since_improvement))
        adjust_learning_rate(optimizer, 0.8)
    start_time = time.time()

    print('start train')
    print('learning rate: {}'.format(args.lr))
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    print('start val')
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    bleu_score_value = calculate_bleu(test_data, SRC, TRG, model, device)
    print('-----------------EPOCH: {}, bleu_score: {}'.format(epoch, bleu_score_value))
    if bleu_score_value > best_bleu_score_value:
    # if valid_loss < best_valid_loss:
        epochs_since_improvement = 0
        best_bleu_score_value = bleu_score_value
        torch.save({
            'representations': representations,
            'model': model.state_dict(),
            'bleu_score': bleu_score_value,
            'encoder': enc.state_dict()

        }, os.path.join(save_dir, 'BEST.pt'))
    else:
        epochs_since_improvement += 1

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    if not args.run_local:
        wandb.log({"train_loss": train_loss,
                   "valid_loss": valid_loss,
                   "bleu-score": bleu_score_value})
    epoch += 1

# --------------------------------

# bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)
#
# example_idx = 12
#
# src = vars(train_data.examples[example_idx])['src']
# trg = vars(train_data.examples[example_idx])['trg']
#
# print(f'src = {src}')
# print(f'trg = {trg}')
#
# translation, attention = translate_sentence(src, SRC, TRG, model, device)
#
# print(f'predicted trg = {translation}')
# display_attention(src, translation, attention)


# dotproduct_train_no_attention.py
