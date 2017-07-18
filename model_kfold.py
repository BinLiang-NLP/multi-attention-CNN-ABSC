# -*- encoding:utf-8 -*-

import os
import re
from collections import defaultdict
from util import read_lines
from time import time
import numpy as np
from dataset import load_data
#from dataset_mark import load_data
from keras.layers import Input, Embedding, Activation, \
    LSTM, GRU, Convolution1D, Lambda, Dense, Dropout, merge
from keras.layers.advanced_activations import LeakyReLU, \
    PReLU, ELU, ParametricSoftplus, ThresholdedReLU, SReLU
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adamax
from keras.constraints import maxnorm
from keras.regularizers import l2, activity_l2
from keras.utils import np_utils
from keras import backend as K
from evaluate import sim_compute, compute
import theano
import theano.tensor as T
from theano import scan
from theano import pp
from keras.engine.topology import Layer

# for data augmentation
# from imblearn.over_sampling import SMOTE

# 参数设置
batch_size = 32
nb_epoch = 8
nb_filter = 200
nb_hiddens = 200
word_embed_dim = 100  # 50 or 300
position_embed_dim = 10   # v0.2新增
tag_embed_dim = 50  # v0.4新增
nb_classes = 3
max_sent_len = 100
class_embbed_dim = 300
k_max_size = 3

# 加载数据
print('Loading data...')
train_data, test_data, word_weights, position_weights, tag_weights, label_voc_rev = \
    load_data(max_sent_len, word_embed_dim, position_embed_dim, tag_embed_dim, sentence_len=False)
td_target_indices, td_sentence, td_tag, td_position, td_target, td_label, td_num, td_targets_str = train_data[:]
# 测试 da
sm = SMOTE(random_state=42)
# end
seed = 12345
np.random.seed(seed)
np.random.shuffle(td_target_indices)
np.random.seed(seed)
np.random.shuffle(td_sentence)
np.random.seed(seed)
np.random.shuffle(td_tag)
np.random.seed(seed)
np.random.shuffle(td_position)
np.random.seed(seed)
np.random.shuffle(td_target) 
np.random.seed(seed)
np.random.shuffle(td_label)
np.random.seed(seed)
np.random.shuffle(td_num)
np.random.seed(seed)
np.random.shuffle(td_targets_str)
#test_nums, test_targets_str, test_target_indices, test_sentence, test_tag, test_position, test_target = test_data[:]
#td_label = np_utils.to_categorical(td_label, nb_classes)

def max_1d(X):
    return K.max(X, axis=1)

def min_1d(X):
    return K.min(X, axis=1)

def gather_k_max(x, k):
    """
    :param x: shape=[n, nb_filter]
    """
    indices = T.argsort(x.transpose())[:, -k:]
    results, updates = theano.scan(lambda d, index: d[index], sequences=[x.transpose(), indices])
    return results.flatten()

def k_max(X, k):
    """
    K-max pooling
    :param k: shape=[batch_size, n, nb_filter]
    """
    results, updates = theano.scan(gather_k_max, sequences=X, non_sequences=k)
    return results

def get_att(X, index):
    result, update = theano.scan(lambda v, u: T.dot(v, T.transpose(u)), sequences=X, non_sequences=X[index])
    result_soft = T.nnet.softmax(result)  # T.exp(result) / T.sum(T.exp(result))
    A = T.diag(T.flatten(result_soft))  # 对角阵，n×n
    return T.dot(A, X)

def get_input_att(Xs, target_indices):
    """
    :param X: 输入
    :param target_index: target在句子中的下标
    :return: xx
    """
    result, update = theano.scan(get_att, sequences=[Xs, target_indices])
    return result 


class AttPoolLayer(Layer):

    ID = 1

    def __init__(self, output_dim, nb_classes, **kwargs):
        #self.input_dim = input_dim
        self.output_dim_test = output_dim
        self.nb_classes = nb_classes
        super(AttPoolLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        #U_value = np.random.normal(size=(input_shape[-1], self.output_dim)).astype('float32')
        #self.U = theano.shared(U_value, name='param_U'+str(AttPoolLayer.ID))  # params
        #WL_value = np.random.normal(size=(self.output_dim, self.nb_classes)).astype('float32')
        #self.WL = theano.shared(WL_value, name='param_WL'+str(AttPoolLayer.ID))  # params
        self.U = self.add_weight(shape=(input_shape[-1], self.output_dim_test), initializer='normal',
            trainable=True)
        self.WL = self.add_weight(shape=(self.output_dim_test, self.nb_classes), initializer='normal',
            trainable=True)
        AttPoolLayer.ID += 1
        super(AttPoolLayer, self).build(input_shape)
    
    def call(self, x, mask=None):
        """
        :param x: shape=[batch_size, n, nb_filter](若上一层是conv)
        """
        return self.att_based_pooling(x)

    def get_output_shape_for(self, input_shape):
        #return (input_shape[0], self.output_dim)
        return (input_shape[0], input_shape[-1])

    def pool_one(self, R):
        """
        Attention-based pooling
        :param R: sentence representation, shape=[n, nb_filter]
        :return W_max: shape=[class_embbed_dim,]
        """
        G = theano.dot(theano.dot(R, self.U), self.WL)  # shape=[n, nb_classes]
        A = T.nnet.softmax(G.transpose()).transpose()  # shape=[n, nb_classes]
        WO = T.dot(R.transpose(), A)  # shape=[nb_filter, nb_classes]
        W_max = T.max(WO, axis=1)  # shape=[nb_filter,]
        return T.tanh(W_max)

    def att_based_pooling(self, Rs):
        """
        Attention-based pooling
        :param Rs: 卷积层输出，shape=[batch_size, n, nb_filter]
        """
        results, updates = theano.scan(self.pool_one, sequences=Rs)
        return results  # shape=[batch_size, class_embed_dim]


k_fold = 5  # k-fold
f_scores = []  # 存放5次的f值
data_count = len(td_label)  # 所有数据量
per_fold_count = int(data_count / k_fold)  # 每一份数据数量
error_num = defaultdict(dict)  # 预测错误的实例，编号：{错误次数, 错误标签集合}
#max_sent_len += 2
for step in range(k_fold):
    # 划分测试集、开发集
    boundry = int(len(td_label) * 0.8)
    start, end = per_fold_count*step, per_fold_count*(step+1)
    test_target_indices, test_sentence, test_position, test_tag, test_label = \
            td_target_indices[start:end], td_sentence[start:end], td_position[start:end], td_tag[start:end], td_label[start:end]
    train_target_indices, train_sentence, train_position, train_tag, train_label = \
        np.concatenate((td_target_indices[0:start], td_target_indices[end:])), \
        np.concatenate((td_sentence[0:start], td_sentence[end:])), \
        np.concatenate((td_position[0:start], td_position[end:])), \
        np.concatenate((td_tag[0:start], td_tag[end:])), \
        np.concatenate((td_label[0:start], td_label[end:]))
    #train_sentence, train_label = sm.fit_sample(train_sentence, train_label)
    train_label = np_utils.to_categorical(train_label, nb_classes)
    test_label = np_utils.to_categorical(test_label, nb_classes)
    test_num = td_num[start:end]  # 待预测实例的编号
    targets_str = td_targets_str[start:end]

    # 构建模型
    print('Buildind model...')
    #input_target_indices = Input(shape=(1,), dtype='int32', name='input_target')
    input_sentence = Input(shape=(max_sent_len,), dtype='int32', name='input_sentence')
    embed_sentence = Embedding(output_dim=word_embed_dim, input_dim=word_weights.shape[0],
                               input_length=max_sent_len, weights=[word_weights],
                               dropout=0.12, name='embed_sentence')(input_sentence)

    input_tag = Input(shape=(max_sent_len,), dtype='int32', name='input_sentiment')
    embed_tag = Embedding(output_dim=tag_embed_dim, input_dim=tag_weights.shape[0],
                               input_length=max_sent_len, weights=[tag_weights],
                               dropout=0.12, name='embed_tag')(input_tag)

    #input_position = Input(shape=(max_sent_len,), dtype='int32', name='input_position')
    #embed_position = Embedding(output_dim=position_embed_dim, input_dim=position_weights.shape[0],
    #                           input_length=max_sent_len, weights=[position_weights],
    #                           dropout=0.12, name='embed_position')(input_position)

    X_sentence = merge([embed_sentence, embed_tag],
                       mode='concat', concat_axis=2)

    cnn_layer_2 = Convolution1D(nb_filter=nb_filter, filter_length=2,  # 窗口2
        border_mode='same', activation='relu', name='conv_window_2')(X_sentence)
    pool_output_2 = Lambda(max_1d, output_shape=(nb_filter,))(cnn_layer_2)
    #pool_output_2 = AttPoolLayer(class_embbed_dim, 3)(cnn_layer_2)

    cnn_layer_3 = Convolution1D(nb_filter=nb_filter, filter_length=3,  # 窗口3
        border_mode='same', activation='relu', name='conv_window_3')(X_sentence)
    pool_output_3 = Lambda(max_1d, output_shape=(nb_filter,))(cnn_layer_3)
    #pool_output_3 = AttPoolLayer(class_embbed_dim, 3)(cnn_layer_3)

    cnn_layer_4 = Convolution1D(nb_filter=nb_filter, filter_length=4,  # 窗口4
        border_mode='same', activation='relu', name='conv_window_4')(X_sentence)
    pool_output_4 = Lambda(max_1d, output_shape=(nb_filter,))(cnn_layer_4)
    #pool_output_4 = AttPoolLayer(class_embbed_dim, 3)(cnn_layer_4)

    cnn_layer_5= Convolution1D(nb_filter=nb_filter, filter_length=5,  # 窗口5
        border_mode='same', activation='relu', name='conv_window_5')(X_sentence)
    pool_output_5 = Lambda(max_1d, output_shape=(nb_filter,))(cnn_layer_5)
    #pool_output_5 = AttPoolLayer(class_embbed_dim, 3)(cnn_layer_5)

    pool_output = merge([pool_output_2, pool_output_3, pool_output_4, pool_output_5],
        mode='concat', name='pool_output')
    #X_dropout = GaussianDropout(0.5)(pool_output)
    X_dropout = Dropout(0.5)(pool_output)
    X_output = Dense(nb_classes,
        W_constraint=maxnorm(3),
        W_regularizer=l2(0.01), 
        activity_regularizer=activity_l2(0.01),
        activation='softmax')(X_dropout)
    model = Model(input=[input_sentence, input_tag], output=[X_output])
    model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.summary()
    #exit()
    print('Train...')
    modelcheckpoint = ModelCheckpoint('./model/best_model_kf.hdf5', verbose=1, save_best_only=True, mode='min')
    model.fit([train_sentence, train_tag],
              [train_label],
              nb_epoch=nb_epoch, batch_size=batch_size,
              callbacks=[modelcheckpoint],
              validation_data=([test_sentence, test_tag], [test_label]))
    # 评价
    score = model.evaluate([test_sentence, test_tag], [test_label])
    pre = model.predict([test_sentence, test_tag]) 
    pre_labels, data_labels = [], []
    for i in range(pre.shape[0]):
        num = test_num[i]
        target = targets_str[i]
        p_1 = pre[i].argmax()
        pre_labels.append(p_1)
        p_2 = test_label[i].argmax()
        data_labels.append(p_2)
        if p_1 != p_2:  # 预测错误
            num_target = num + '|' + target
            if 'err_labels' not in error_num[num_target]:
                error_num[num_target]['err_labels'] = [p_1]
            else:
                error_num[num_target]['err_labels'].append(p_1)
    r_path = './result/result_kfold_att.txt'
    score = compute(pre_labels, data_labels, ignore_label=None, classes_dict_rev=label_voc_rev, result_path=r_path)
    f_scores.append(score[1])
print('\n\nf_scores:', f_scores)
print('mean of f_scores: %f' % (sum(f_scores) / k_fold))
# 预测错误的标签写入文件
path = './error'
if not os.path.exists(path):
    os.mkdir(path)
# 保存模型
#print('Save model...')
model_path = './model/'
if not os.path.exists(model_path):
    os.mkdir(model_path)
#model.save(model_path+'multi-window_cnn.h5')
#del model

# 重新加载模型
model = load_model(model_path+'best_model.hdf5')
# 加载训练数据
train_dict = dict()
lines = read_lines('./com_data/data_h/Train.csv')
for line in lines:
    items = line.split('|')
    num_target = '|'.join(items[:2])
    train_dict[num_target] = '|'.join(items[2:])
file = open(path+'/error.csv','w',encoding='utf-8')
error_num = sorted(error_num.items(), key=lambda d:d[0])
for item in error_num:
    num_target = item[0]
    inner_dict = item[1]
    err_labels = inner_dict['err_labels'] \
        if 'err_labels' in inner_dict  else 0
    e_labels = set()
    for err_label in err_labels:
        e_labels.add(label_voc_rev[err_label])
    file.write('%s|error_labels=[%s]|%s\n' % \
        (num_target, ','.join(e_labels), train_dict[num_target]))
file.close()
print('Done!')

