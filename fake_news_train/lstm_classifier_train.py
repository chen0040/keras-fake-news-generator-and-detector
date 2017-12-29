from __future__ import print_function

from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from fake_news_train.utils import plot_confusion_matrix
from fake_news_classifiers.recurrent_networks import LstmClassifier, fit_input_text
import numpy as np


def main():
    np.random.seed(42)
    data_dir_path = './data'

    print('loading csv file ...')

    # Import `fake_or_real_news.csv`
    df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")

    # Set `y`
    Y = [1 if label == 'REAL' else 0 for label in df.label]

    # Drop the `label` column
    df.drop("label", axis=1)

    print('extract configuration from input texts ...')

    X = df['text']

    config = fit_input_text(X)
    config['num_target_tokens'] = 2

    print('configuration extracted from input texts ...')

    classifier = LstmClassifier(config)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    print('training size: ', len(Xtrain))
    print('testing size: ', len(Xtest))

    print('start fitting ...')
    classifier.fit(Xtrain, Ytrain, Xtest, Ytest)

    print('start predicting ...')
    pred = classifier.predict(Xtest)
    score = metrics.accuracy_score(Ytest, pred)
    print("accuracy:   %0.3f" % score)
    cm = metrics.confusion_matrix(Ytest, pred, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


if __name__ == '__main__':
    main()