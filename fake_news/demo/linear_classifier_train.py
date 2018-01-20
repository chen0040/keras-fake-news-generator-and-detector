from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from fake_news.library.utility.plot_utils import plot_confusion_matrix, most_informative_feature_for_binary_classification

def main():
    data_dir_path = './data'

    # Import `fake_or_real_news.csv`
    df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")

    # Set `y`
    y = df.label

    # Drop the `label` column
    df.drop("label", axis=1)

    # Make training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

    # Initialize the `tfidf_vectorizer`
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # Fit and transform the training data
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)

    # Transform the test set
    tfidf_test = tfidf_vectorizer.transform(X_test)

    linear_clf = PassiveAggressiveClassifier(n_iter=50)

    linear_clf.fit(tfidf_train, y_train)
    pred = linear_clf.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
    most_informative_feature_for_binary_classification(tfidf_vectorizer, linear_clf, n=30)


if __name__ == '__main__':
    main()