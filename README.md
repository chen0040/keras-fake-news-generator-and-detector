# keras-fake-news-generator-and-detector

Fake news generator and detector using keras

The fake news data is from [https://github.com/GeorgeMcIntire/fake_real_news_dataset](https://github.com/GeorgeMcIntire/fake_real_news_dataset)

# Fake News Detector

The following neural network models are implemented for fake new detector:

* LSTM recurrent network with embedding layer
    * training: run fake_news_train/lstm_classifier_train.py to train and save the model
    * predicting: run fake_news_train/lstm_classifier_predict.py to load the trained model for prediction
 
* Feed-forward network with GloVe embedding layer
    * training: run fake_news_train/glove_ffn_classifier_train.py to train and save the model
    * predicting: run fake_news_train/glove_ffn_classifier_predict.py to load the trained model for prediction
    
* Feed-forward network with Doc2Vec that encode the new article and pass as input to the feedforward network
    * training: run fake_news_train/glove_ffn_classifier_train.py to train and save the model
    * predicting: run fake_news_train/glove_ffn_classifier_predict.py to load the trained model for prediction
    
Below are the training history in terms of loss and accuracy for a number of neural network implemented in keras:

![lstm-history.png](/fake_news/fake_news_train/reports/lstm-history.png)

![glove-feed-forward-history.png](/fake_news/fake_news_train/reports/glove-feed-forward-history.png)

![doc2vec-feed-forward-history.png](/fake_news/fake_news_train/reports/doc2vec-feed-forward-history.png)
