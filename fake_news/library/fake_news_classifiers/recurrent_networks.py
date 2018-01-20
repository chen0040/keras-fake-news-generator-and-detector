from __future__ import print_function

from keras.models import Sequential
from keras.layers import Embedding, Dense, SpatialDropout1D
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import numpy as np

EMBEDDING_SIZE = 100
BATCH_SIZE = 64
VERBOSE = 1
EPOCHS = 10


def generate_batch(x_samples, y_samples):
    num_batches = len(x_samples) // BATCH_SIZE

    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            yield x_samples[start:end], y_samples[start:end]


class LstmClassifier(object):
    model_name = 'lstm'

    def __init__(self, config):
        self.num_input_tokens = config['num_input_tokens']
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.word2idx = config['word2idx']
        self.idx2word = config['idx2word']
        self.config = config

        model = Sequential()
        model.add(Embedding(input_dim=self.num_input_tokens, output_dim=EMBEDDING_SIZE,
                            input_length=self.max_input_seq_length))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(self.num_target_tokens, activation='softmax'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def load_weights(self, weight_file_path):
        self.model.load_weights(weight_file_path)

    def transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = []
            for word in line.lower().split(' '):
                wid = 1
                if word in self.word2idx:
                    wid = self.word2idx[word]
                x.append(wid)
                if len(x) >= self.max_input_seq_length:
                    break
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_input_seq_length)

        print(temp.shape)
        return temp

    def transform_target_encoding(self, targets):
        return np_utils.to_categorical(targets, num_classes=self.num_target_tokens)

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, model_dir_path=None):
        if epochs is None:
            epochs = EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'

        config_file_path = model_dir_path + '/' + self.model_name + '-config.npy'
        weight_file_path = model_dir_path + '/' + self.model_name + '-weights.h5'
        checkpoint = ModelCheckpoint(weight_file_path)
        np.save(config_file_path, self.config)
        architecture_file_path = model_dir_path + '/' + self.model_name + '-architecture.json'
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.transform_target_encoding(Ytrain)
        Ytest = self.transform_target_encoding(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)

        train_gen = generate_batch(Xtrain, Ytrain)
        test_gen = generate_batch(Xtest, Ytest)

        train_num_batches = len(Xtrain) // BATCH_SIZE
        test_num_batches = len(Xtest) // BATCH_SIZE

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)
        return history

    def predict(self, x):
        is_str = False
        if type(x) is str:
            is_str = True
            x = [x]

        Xtest = self.transform_input_text(x)

        preds = self.model.predict(Xtest)
        if is_str:
            preds = preds[0]
            return np.argmax(preds)
        else:
            return np.argmax(preds, axis=1)

