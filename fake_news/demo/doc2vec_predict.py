from __future__ import print_function

import pandas as pd
from fake_news.library.encoders.doc2vec import Doc2Vec
import numpy as np

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

    doc_encoder = Doc2Vec(config)
    doc_encoder.load_glove(very_large_data_dir_path)

    print('start predicting ...')

    print('output dimension: ', doc_encoder.get_doc_vec_length())

    for x in X[0:10]:
        encoded = doc_encoder.predict(x)
        print('encoded doc length: ', len(encoded))


if __name__ == '__main__':
    main()