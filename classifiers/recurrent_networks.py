from keras.models import Sequential
from keras.layers import Embedding, Dense
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from collections import Counter
import numpy as np

EMBEDDING_SIZE = 100
BATCH_SIZE = 64
VERBOSE = 1
EPOCHS = 20
MAX_VOCAB_SIZE = 10000


def fit_input_text(X):
    input_counter = Counter()
    max_seq_length = 0
    for line in X:
        max_seq_length = max(max_seq_length, len(line))
        for word in line:
            input_counter[word.lower()] += 1
    word2idx = dict()
    for idx, word in enumerate(input_counter.most_common(MAX_VOCAB_SIZE)):
        word2idx[word[0]] = idx + 2
    word2idx['PAD'] = 0
    word2idx['UNK'] = 1
    idx2word = dict([(idx, word) for word, idx in word2idx.items()])
    num_input_tokens = len(word2idx)
    config = dict()
    config['word2idx'] = word2idx
    config['idx2word'] = idx2word
    config['num_input_tokens'] = num_input_tokens
    config['max_input_seq_length'] = max_seq_length
    return config


class LstmClassifier(object):

    num_input_tokens = None
    max_input_seq_length = None
    num_target_tokens = None
    word2idx = None
    idx2word = None
    model = None

    def __init__(self, config):
        self.num_input_tokens = config['max_input_tokens']
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.word2idx = config['word2idx']
        self.idx2word = config['idx2word']

        model = Sequential()
        model.add(Embedding(input_dim=self.num_input_tokens, output_dim=EMBEDDING_SIZE, input_length=self.max_input_seq_length))
        model.add(LSTM(256, return_sequences=False, return_state=False, dropout=0.2))
        model.add(Dense(self.num_target_tokens))
        self.model = model

    def transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = []
            for word in line:
                wid = 1
                if word in self.word2idx:
                    wid = self.word2idx[word]
                x.append(wid)
            temp.append(x)
        texts = pad_sequences(temp, maxlen=self.max_input_seq_length)
        return texts

    def transform_target_encoding(self, targets):
        return np_utils.to_categorical(targets, num_classes=self.num_target_tokens)

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None):
        if epochs is None:
            epochs = EPOCHS

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)
        Ytrain = self.transform_target_encoding(Ytrain)
        Ytest = self.transform_target_encoding(Ytest)

        self.model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, verbose=VERBOSE, epochs=epochs, validation_data=(Xtest, Ytest))

    def predict(self, x):
        Xtest = self.transform_input_text([x])
        preds = self.model.predict(Xtest)[0]
        return np.argmax(preds)