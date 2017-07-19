#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import

import click as ck
import numpy as np
import pandas as pd

from keras.models import Model, load_model
from keras.layers import (
    Input, Dense, Dropout, merge, MaxoutDense,
    BatchNormalization, GaussianDropout, Reshape, Flatten)
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop, Adam, Adagrad, Adadelta, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import regularizers
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

@ck.command()
@ck.option('--test', default=False, is_flag=True)
def main(test):
    dataset = load_data()
    
    train_model(dataset, test=test)
    

def load_data():
    df = pd.read_csv('data/bowel.tsv', sep='\t')
    print(df.columns.values)
    features = { 
        'Gender': [0, 2],
        'Age': [2, 100],
        'BMI': [2, 100],
        'ASA grade': [1, 6],
        'IHD/Stroke': [1, 4],
        'Diabetes': [1, 5],
        'Anticoagulant use': [1, 4],
        'Abnormal SCr': [1, 4],
        'Smoking': [1, 8],
        # 'Smoking pack years',
        'Statin': [1, 4],
        'Preop nutrition': [1, 5],
        'Albumin': [1, 4],
        # 'Albumin value': ,
        'Hb': [2, 100],
        'operating surgeon': [1, 5],
        'Indication': [1, 12],
        'Urgency': [1, 3],
        'Preop abscess': [1, 4],
        'Preop abscess drainage': [1, 5],
        'Interval from drainage to surgery': [2, 100],
        'Preop fistula': [1, 4],
        'Defunctioning stoma': [1, 7],
        # 'op_dur': [2, 100],
        'Access': [1, 11],
        'Bowel prep': [1, 4],
        'RHEMI prox cut': [1, 7],
        'RHEMI distal cut': [1, 9],
        'STOMA type reversed': [1, 4],
        'Diverting stoma': [1, 3],
        # 'STOMA end_loop': [1, 4],
        # 'LEFTR prox cut': [],
        # 'LEFTR distal cut': [],
        # 'Anastomosis_none_handsewn_stapled': [1, 4],
        # 'Anastomosis_intra_extra corporeal': [1, 5],
        # 'Anastomosis_SS_ES': [1, 7],
        # 'Handsewn anastomosis_continuous_interrupted': [1, 4],
        # 'Handsewn anastomosis_suture size': [1, 7],
        # 'Handsewn anastomosis_suture used': [1, 20],
        # 'Handsewn anastomosis_number of layers': [1, 5],
        # 'Handsewn anastomosis_layers taken': [1, 5],
        # 'Primary device L or C': [1, 4],
        # 'Apical device L or C': [1, 4],
        # 'Stapled oversewn': [1, 6],
        # 'Fascia method': [1, 4],
        # 'Fascia material': [1, 18],
        # 'Skin material': [1, 5],
        # 'Skin method': [1, 7],
        # 'IOC - any': [0, 2],
        # 'IOC - vascular': [0, 2],
        # 'IOC - bowel': [0, 2],
        # 'IOC - organ': [0, 2],
        # 'Revise anastomosis': [1, 4],
        # 'CRP': [2, 100],
        # 'criticial_care': [1, 6],
        # 'LOS': [2, 30],
        # 'Clavien Dindo': [1, 11],
        'Prev surgery': [1, 5],
        'CD preop steroids': [1, 4],
        'CD preop 5ASA': [1, 4],
        'CD preop immunosuppressants': [1, 4],
        'CD preop biologics': [1, 4],
        'unexp_abscess': [1, 4],
        'fistula': [1, 4],
        'fistula_bowel': [1, 4],
        'fistula_colon': [1, 4],
        'fistula_bladder': [1, 4],
        'fistula_skin': [1, 4],
        'bowel_obstruct': [1, 4],
        'stricture_plasties': [1, 4],
        'Days since biopsy': [2, 100],
        'chemo': [1, 4]
        
    }
    n = len(df)
    data = np.empty((n, 0), dtype=np.int32)
    for key, info in features.iteritems():
        fs = df[key].values
        if info[0] == 0:
            fs = fs.astype(np.int32) - 1
            fs = fs.reshape(n, 1)
        elif info[0] == 1:
            fs[fs == 'na'] = '0'
            fs = fs.astype(np.int32)
            fs[fs < 0] = 0
            fd = np.zeros((n, info[1]), dtype=np.int32)
            for i in xrange(n):
                fd[i, fs[i]] = 1
            fs = fd
        else:
            continue
            fs[fs == 'Missing'] = '0'
            fs[fs == 'missing'] = '0'
            fs[fs == 'na'] = '0'
            fs = np.array(map(
                lambda x: x.replace(',','.'), fs), dtype=np.float32)
            # print(key, np.max(fs), np.min(fs))
            mx = np.max(fs)
            mn = np.min(fs)
            fs = (fs / mx).reshape(n, 1)
        data = np.concatenate((data, fs), axis=1)
    print(np.sum(data))
    po = df['Primary outcome'].values.astype(np.int32) - 1
    po[po == 2] = 1
    # One Hot Encoding
    labels = np.zeros((n, 2), dtype=np.int32)
    for i in xrange(n):
        labels[i, po[i]] = 1
    print(np.sum(labels))
    index = np.arange(n)
    np.random.seed(seed=0)
    np.random.shuffle(index)
    return data[index], labels[index]
    

def train_model(dataset, split=0.8, batch_size=512, epochs=128, test=False):
    data, labels = dataset
    n = data.shape[0]
    valid_n = int(split * n)
    train_n = int(split * valid_n) 
    train_data, train_labels = data[:train_n], labels[:train_n]
    valid_data, valid_labels = data[train_n:valid_n], labels[train_n:valid_n]
    test_data, test_labels = data[valid_n:], labels[valid_n:]

    print('Training data size:', train_data.shape)
    print('Valid data size:', valid_data.shape)
    print('Test data size:', test_data.shape)

    # PCA
    # pca = PCA(n_components=24)
    # train_data = pca.fit_transform(train_data)
    # print(pca.explained_variance_ratio_)
    # valid_data = pca.transform(valid_data)
    # test_data = pca.transform(test_data)
    # return
    if not test:
        input_length = train_data.shape[1]
        print('Input length:', input_length)

        inputs = Input(shape=(input_length, ))
        net = inputs
        net = Dense(256, activation='relu')(net)
        net = Dense(2, activation='softmax')(net)
        model = Model(inputs=inputs, outputs=net)

        model.summary()
        optimizer = RMSprop(lr=0.0001)
        model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

        checkpointer = ModelCheckpoint(
            filepath='data/model.h5',
            save_best_only=True,
            verbose=1)

        history = model.fit(
            train_data, train_labels,
            batch_size=batch_size, epochs=epochs,
            verbose=1, validation_data=(valid_data, valid_labels),
            callbacks=[checkpointer])

    model = load_model('data/model.h5')

    score = model.evaluate(test_data, test_labels, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    preds = model.predict(test_data)
    predictions = (preds >= 0.3).astype(np.int32)
    print(classification_report(test_labels, predictions))
    roc_auc = compute_roc(preds, test_labels)
    print('ROC AUC:', roc_auc)

def compute_roc(preds, labels):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


if __name__ == '__main__':
    main()
