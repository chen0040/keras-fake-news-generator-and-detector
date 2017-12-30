from __future__ import print_function

import pandas as pd
from sklearn.model_selection import train_test_split
from fake_news.fake_news_utility.fake_news_loader import fit_input_text
from fake_news.fake_new_encoders.doc2vec import Doc2Vec
import numpy as np

MAX_INPUT_SEQ_LENGTH = 200
MAX_VOCAB_SIZE = 2000


def main():
    np.random.seed(42)
    data_dir_path = './data'
    model_dir_path = './models'
    very_large_data_dir_path = './very_large_data'

    print('loading csv file ...')

    # Import `fake_or_real_news.csv`
    df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")
    X = df['text']

    config = np.load(Doc2Vec.get_config_file_path(model_dir_path=model_dir_path)).item()

    classifier = Doc2Vec(config)
    classifier.load_glove(very_large_data_dir_path)

    print('start predicting ...')

    for x in X[0:10]:
        encoded = classifier.predict(x)
        print(encoded)


if __name__ == '__main__':
    main()