import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Bidirectional
from keras.layers.convolutional import Conv1D, AveragePooling1D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, roc_auc_score,accuracy_score,recall_score,f1_score,precision_score
import numpy as np
import argparse
from sklearn.metrics import plot_roc_curve,precision_recall_curve,average_precision_score
from dealwithdata import *
from keras_self_attention import SeqSelfAttention
from utils import *



#Configure GPU and TensorFlow sessions:
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


def CRIP(parser):
    protein = parser.protein
    batch_size = parser.batch_size
    hiddensize = parser.hiddensize
    n_epochs = parser.n_epochs

    trainX, testX, trainY, testY = dealwithdata(protein)

    testY = testY[:,1]

    kf = KFold(n_splits=5)

    fpr_list = []
    tpr_list = []
    aucs = []
    Acc = []
    precision1 = []
    recall1 = []
    fscore1 = []

    for train_index,eval_index in kf.split(trainY):
        train_X = trainX[train_index]
        train_Y = trainY[train_index]
        test_X = trainX[eval_index]
        test_Y = trainY[eval_index]

        model = Sequential()
        model.add(
          MS_CAM(input_dim=81, input_length=101))

        model.add(AveragePooling1D(pool_size=5))

        model.add(Dropout(0.5))
        #Add a bidirectional GRU layer to the model, with a hidden layer size of hidden, and set to return the complete sequence.
        model.add(Bidirectional(GRU(hiddensize, return_sequences=True)))
        model.add(SeqSelfAttention(attention_activation='sigmoid'))
        model.add(Flatten())
        #Add a fully connected layer Dense to the model, with nbfilter neurons and ReLU activation function.
        model.add(Dense(nbfilter, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        #Add an activation layer and use the softmax function for multi classification.
        model.add(Activation('softmax'))
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4))  # 'rmsprop'
        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
        print("model start train---------------")
        model.fit(train_X, train_Y, batch_size=batch_size, epochs=n_epochs, verbose=0,
                  validation_data=(test_X, test_Y),
                  callbacks=[earlystopper])
        predictions = model.predict_proba(test_X)[:, 1]
        pre = np.argmax(model.predict_proba(test_X), axis=-1)
        fpr, tpr, _ = roc_curve(test_Y, predictions)
        # precision, recall, thresholds = precision_recall_curve(test_Y, predictions)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc = roc_auc_score(test_Y, predictions)
        precision = precision_score(test_Y, pre)
        recall = recall_score(test_Y, pre)
        fscore = f1_score(test_Y, pre)
        acc = accuracy_score(test_Y, pre)
        aucs.append(auc)
        Acc.append(acc)
        precision1.append(precision)
        recall1.append(recall)
        fscore1.append(fscore)






if __name__ == "__main__":
    parser.add_argument('--protein', type=str, metavar='<data_file>', required=True)
    parser.add_argument('--hiddensize', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--n_epochs', type=int, default=30)
    args = parser.parse_args()
    CRIP(args)
