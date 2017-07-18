# -*- encoding:utf-8 -*-
"""
模型1
"""
import pandas as pd
import numpy as np
import nltk
from collections import defaultdict
from util import read_lines
import pickle

from keras.preprocessing import sequence
from keras.layers import Input, Embedding, Activation, \
        LSTM, GRU, Convolution1D, Lambda, Dense, Dropout, merge
from keras.layers.advanced_activations import LeakyReLU, \
        PReLU, ELU, ParametricSoftplus, ThresholdedReLU, SReLU
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.optimizers import Adamax
from keras.constraints import maxnorm
from keras.regularizers import l2, activity_l2
from keras.utils import np_utils
from keras import backend as K
from keras.engine.topology import Layer

path = './train/lapt_2016.text'
test_path = './test/EN_LAPT_test.txt'

train_lines = read_lines(path)
test_lines = read_lines(test_path)

all_sentence = []
all_target = []
all_pol = []
all_id = []
all_type = []
train_data = []

pol_dict = {'positive': '1', 'negative': '0', 'neutral': '2'}

for line in train_lines:
    review_id = line.split('|')[0]
    sentence_id = line.split('|')[1]
    category = line.split('|')[2]
    pol = line.split('|')[3]
    if pol == 'neutral': # TODO......
        continue
    sentence = line.split('|')[4]
    # sentence = sentence.split()
    sentence = nltk.word_tokenize(sentence)
    all_id.append(sentence_id)
    all_target.append(category)
    all_pol.append(pol_dict[pol])
    all_sentence.append(sentence)
    all_type.append('train')


for line in test_lines:
    review_id = line.split('|')[0]
    sentence_id = line.split('|')[1]
    category = line.split('|')[2]
    pol = line.split('|')[3]
    if pol == 'neutral': # TODO......
        continue
    sentence = line.split('|')[4]
    # sentence = sentence.split()
    sentence = nltk.word_tokenize(sentence)
    all_id.append(sentence_id)
    all_target.append(category)
    all_pol.append(pol_dict[pol])
    all_sentence.append(sentence)
    all_type.append('test')

d = {'target': pd.Series(all_target,index=all_id), 'pol': pd.Series(all_pol, index=all_id), 'sentence': pd.Series(all_sentence, index=all_id), 'type': pd.Series(all_type, index=all_id)}
data = pd.DataFrame(d)

w = []
# print(len(data['sentence']))
for i in data['sentence']:
    w.extend(i)
# print(w)
# print(len(w))
word_dict = pd.DataFrame(pd.Series(w).value_counts())
# print(word_dict)
word_dict['id'] = list(range(1, len(word_dict)+1))
# print(len(word_dict))

get_words_id = lambda x: list(word_dict['id'][x])
data['words_id'] = data['sentence'].apply(get_words_id)
train_count = 0


# print(data.index[20:30])
# print(data['target'][20:30])
# print(data['pol'][20:30])

# load word_embedding
def load_word_embed(word_embed_dim=300):
    """

    """
    path = './train/EN.glove.840B.300d.pk'
    fp = open(path, 'rb')
    word_embed = pickle.load(fp)
    fp.close()
    return word_embed

def init_word_weights(word_embed_dim=300):
    """
    init_word_weights
    """
    word_embed = load_word_embed(300)
    word_size = len(word_dict) + 2
    word_weights = np.zeros((word_size, word_embed_dim), dtype='float32')
    random_vec = np.random.uniform(-0.1, 0.1, size=(word_embed_dim,))
    exit_count = 0
    for word in word_dict.index:
        word_id = word_dict['id'][word]
        # print(word_id)
        flag = 0
        if '-' in word:
            flag = 1
        if flag:
            word_list = word.split('-')
            list_weigt = np.zeros(word_embed_dim, dtype='float32')
            for w in word_list:
                # print(list_weigt)
                if w in word_embed:
                    exit_count += 1
                    list_weigt += word_embed[w]
                else:
                    list_weigt += random_vec
            list_weigt /= float(len(word_list))
            word_weights[word_id, :] = list_weigt
        else:
            if word in word_embed:
                word_weights[word_id, :] = word_embed[word]
                exit_count += 1
            else:
                word_weights[word_id, :] = random_vec                
        # print(word)
        # exit_count += 1
    # print(exit_count)
    # print(len(word_dict))
    return word_weights

word_weights = init_word_weights()


print(len(list(data['words_id'][data.type=='train'])))
print(len(list(data['words_id'][data.type=='test'])))



word_embed_dim = 300
max_len = 200
batch_size = 32
nb_epoch = 8
nb_class = 2 # TODO......

k_fold = 5
f_scores = []
data_count = len(data)
per_fold_count = int(data_count / k_fold)
print('Begin......')
for step in range(1):
    """
    start = per_fold_count * step
    end = per_fold_count * (step + 1)
    test_sentence = np.array(list(data['words_id'][start:end]))
    # print(test_sentence[0:5])
    test_label = np.array(list(data['pol'][start:end]))
    test_id = data.index[start:end]
    test_target = data['target'][start:end]
    # print(test_sentence)

    train_sentence = np.concatenate((np.array(list(data['words_id'][:start])), np.array(list(data['words_id'][end:]))))
    train_label = np.concatenate((np.array(list(data['pol'][:start])), np.array(list(data['pol'][end:]))))
    train_id = np.concatenate((data.index[:start], data.index[end:]))
    train_target = np.concatenate((data['target'][:start], data['target'][end:]))
    """

    train_sentence = np.array(list(data['words_id'][data.type=='train']))
    test_sentence = np.array(list(data['words_id'][data.type=='test']))
    train_label = np.array(list(data['pol'][data.type=='train']))
    test_label = np.array(list(data['pol'][data.type=='test']))

    train_label = np_utils.to_categorical(train_label, nb_class)
    test_label = np_utils.to_categorical(test_label, nb_class)

    train_sentence = sequence.pad_sequences(train_sentence, maxlen=max_len)
    test_sentence = sequence.pad_sequences(test_sentence, maxlen=max_len)
    # exit(0)
    
    print('Building model......')
    input_sentence = Input(shape=(max_len,), dtype='int32', name='input_sentence')
    embed_sentence = Embedding(output_dim=word_embed_dim, input_dim=word_weights.shape[0],
            input_length=max_len, weights=[word_weights], name='embed_sentence')(input_sentence)
    X_sentence = embed_sentence
    lstm_out = LSTM(128)(X_sentence)
    X_dropout = Dropout(0.5)(lstm_out)
    X_output = Dense(nb_class,
            W_constraint=maxnorm(3),
            W_regularizer=l2(0.01),
            activity_regularizer=activity_l2(0.01),
            activation='softmax')(X_dropout)
    model = Model(input=[input_sentence], output=[X_output])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print('Training...')
    modelCheckpoint = ModelCheckpoint('./model/best_model_kf.hdf5', verbose=1, save_best_only=True, mode='min')
    model.fit([train_sentence],
            [train_label],
            nb_epoch=nb_epoch,
            batch_size=batch_size,
            callbacks=[modelCheckpoint],
            validation_data=(test_sentence, test_label))
    score, acc = model.evaluate(test_sentence, test_label)
    # print('accuracy is:', acc)
    f_scores.append(acc)
print('\n**************************\nfscores:\n', f_scores)
print('mean of f_scores:', sum(f_scores) / k_fold)



