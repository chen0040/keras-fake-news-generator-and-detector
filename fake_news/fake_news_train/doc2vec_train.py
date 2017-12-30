from __future__ import print_function

from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from fake_news.fake_news_utility.fake_news_loader import fit_input_text
from fake_news.fake_new_encoders.doc2vec import Doc2Vec
import numpy as np

MAX_INPUT_SEQ_LENGTH = 50
MAX_VOCAB_SIZE = 2000


def main():
    np.random.seed(42)
    data_dir_path = './data'
    very_large_data_dir_path = './very_large_data'

    print('loading csv file ...')

    # Import `fake_or_real_news.csv`
    df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")
    X = df['text']

    config = fit_input_text(X, max_input_seq_length=MAX_INPUT_SEQ_LENGTH, max_vocab_size=MAX_VOCAB_SIZE)

    classifier = Doc2Vec(config)
    classifier.load_glove(very_large_data_dir_path)

    Xtrain, Xtest = train_test_split(X, test_size=0.2, random_state=42)

    print('testing size: ', len(Xtest))

    print('start fitting ...')
    history = classifier.fit(Xtrain, Xtest)


if __name__ == '__main__':
    main()