from __future__ import print_function

import pandas as pd
from sklearn.model_selection import train_test_split
from fake_news.library.fake_news_utility.plot_utils import plot_and_save_history
from fake_news.library.fake_news_classifiers.feedforward_networks import GloveFeedforwardNet, MAX_SEQ_LENGTH
from fake_news.library.fake_news_utility.fake_news_loader import fit_input_text
import numpy as np


def main():
    np.random.seed(42)
    data_dir_path = './data'
    very_large_data_dir_path = './very_large_data'
    report_dir_path = './reports'

    print('loading csv file ...')

    # Import `fake_or_real_news.csv`
    df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")

    # Set `y`
    Y = [1 if label == 'REAL' else 0 for label in df.label]

    # Drop the `label` column
    df.drop("label", axis=1)

    print('extract configuration from input texts ...')

    X = df['text']

    config = fit_input_text(X, max_input_seq_length=MAX_SEQ_LENGTH)
    config['num_target_tokens'] = 2

    print('configuration extracted from input texts ...')

    classifier = GloveFeedforwardNet(config)
    classifier.load_glove(very_large_data_dir_path)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    print('training size: ', len(Xtrain))
    print('testing size: ', len(Xtest))

    print('start fitting ...')
    history = classifier.fit(Xtrain, Ytrain, Xtest, Ytest)

    history_plot_file_path = report_dir_path + '/' + GloveFeedforwardNet.model_name + '-history.png'
    plot_and_save_history(history, classifier.model_name, history_plot_file_path)


if __name__ == '__main__':
    main()