from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from fake_news_train.utils import plot_confusion_matrix
from classifiers.recurrent_networks import LstmClassifier, fit_input_text
import numpy as np


def main():
    np.random.seed(42)
    data_dir_path = './data'

    # Import `fake_or_real_news.csv`
    df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")

    # Set `y`
    y = [1 if label == 'REAL' else 0 for label in df.label]

    # Drop the `label` column
    df.drop("label", axis=1)

    config = fit_input_text(df['text'])
    config['num_target_tokens'] = 2

    print(config)

    classifier = LstmClassifier(config)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(df['text'], y, test_size=0.2, random_state=42)

    classifier.fit(Xtrain, Ytrain, Xtest, Ytest)

    pred = classifier.predict(Xtest)
    score = metrics.accuracy_score(Ytest, pred)
    print("accuracy:   %0.3f" % score)
    cm = metrics.confusion_matrix(Ytest, pred, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


if __name__ == '__main__':
    main()