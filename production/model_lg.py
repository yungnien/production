import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import numpy as np
from joblib import dump, load
import my_utils
import my_data

num_words = my_utils.num_words 
path = my_utils.path

def train_lg(X_train, y_train):
  logreg = Pipeline([('vect', CountVectorizer(max_features=num_words, min_df=2, lowercase=False)),
                   ('tfidf', TfidfTransformer()),
                   ('clf', LogisticRegressionCV(class_weight='balanced', cv=5, scoring='roc_auc', max_iter=1000,n_jobs=-1)),
                  ])
  logreg.fit(X_train, y_train)
  dump(logreg, path+'logreg_ref.pkl')
  logging.info('complete the training')
  return logreg

def predict_lg(sentence):
  logreg = load(path +'logreg_ref.pkl')
  sentence = str_clean(sentence)
  y_pred = logreg.predict([sentence])
  logging.info('return prediction')
  return y_pred
  
def test_lg(X_dev, y_dev):
  logreg = load(path +'logreg_ref.pkl')
  y_pred = logreg.predict(X_dev)
  return f1_score(y_dev, y_pred, average='weighted') > 0.90

def run_train_lg(filename==""):
  if filename = None:
    filename = my_utils.datafile

  X_train, X_dev, X_test, y_train, y_dev, y_test = my_data.generate_lg(filename)
  train_lg(X_train, y_train)

