import logging
import re
import contractions
import pandas as pd
import unicodedata
from joblib import dump, load
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import my_utils

num_words = my_utils.num_words
max_tokens = my_utils.max_tokens
pad = my_utils.pad
path = my_utils.path

def remove_white_space(text):
    return text.strip().strip('\t\n')

def remove_special_character(text):
    return re.sub('[^A-Za-z0-9\s]+', '', text)

def data_clean(train_data, filename):
  # simple text clean up
  train_data['question_text'] = train_data['question_text']\
  .str.normalize('NFKD').apply(contractions.fix).apply(remove_white_space)\
  .str.lower().apply(remove_special_character)
  train_data['word_count'] = train_data['question_text'].apply(lambda x: len(str(x).split()))
  #remove empty text
  train_data = train_data.loc[(train_data.word_count > 0)]
  train_data= train_data.reset_index()
  dump(train_data, path+filename)
  return train_data

def str_clean(question):
  return remove_special_character(remove_white_space(contractions.fix(unicodedata.normalize('NFKD', question))).lower())

def threeway_split(X, y):
    X_train, X_hold, y_train, y_hold  = train_test_split(X, y, 
                                                     train_size = 0.8, test_size = 0.2, 
                                                     random_state = 42, stratify = y)
    X_dev, X_test, y_dev, y_test  = train_test_split(X_hold, y_hold, 
                                                     train_size = 0.5, test_size = 0.5,  
                                                     random_state = 42, stratify = y_hold)
    del X_hold, y_hold
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def generate(filename):
  train_data = pd.read_csv(path+filename)
  train_data = data_clean(train_data, 'train_ref.pkl')
  train_data_s = pd.concat([train_data.loc[(train_data['target'] == 0) & (train_data['question_text'].str.len() > 10)].sample(n=90000, random_state=42),\
                            train_data.loc[(train_data['target'] == 1) & (train_data['question_text'].str.len() > 10)].sample(n=80000, random_state=42)], ignore_index=True)
  train_data_s = train_data_s.sample(frac=1).reset_index(drop=True)
  
  X_train, X_dev, X_test, y_train, y_dev, y_test = threeway_split(train_data['question_text'], train_data['target'])
  X_train_s, X_dev_s, X_test_s, y_train_s, y_dev_s, y_test_s = threeway_split(train_data_s['question_text'], train_data_s['target'])
  logging.info('Training data set (regular): ' + str(len(train_data)))
  logging.info('Training data set (small): ' + str(len(train_data_s)))
  logging.info(train_data_s.head())
  
  #for model 1
  dump(X_train, path+'X_train_ref.pkl')
  dump(y_train, path+'y_train_ref.pkl')
  dump(X_dev, path+'X_dev_ref.pkl')
  dump(y_dev, path+'y_dev_ref.pkl')
  dump(X_test, path+'X_test_ref.pkl')
  dump(y_test, path+'y_test_ref.pkl')

  #for model 2
  tokenizer = Tokenizer(num_words=num_words, lower=False, char_level=False)
  tokenizer.fit_on_texts(train_data['question_text'])
  # need tokenizer and padding for predict
  X_train_token  = tokenizer.texts_to_sequences(X_train)
  X_dev_token  = tokenizer.texts_to_sequences(X_dev)
  X_test_token  = tokenizer.texts_to_sequences(X_test)
  X_train_token = pad_sequences(X_train_token, maxlen=max_tokens, padding=pad, truncating=pad).tolist()
  X_dev_token = pad_sequences(X_dev_token, maxlen=max_tokens, padding=pad, truncating=pad).tolist()
  X_test_token = pad_sequences(X_test_token, maxlen=max_tokens, padding=pad, truncating=pad).tolist()
  dump(tokenizer, path+'tokenizer_ref.pkl')
  dump(X_train_token, path+'X_train_token_ref.pkl')
  dump(X_dev_token, path+'X_dev_token_ref.pkl')
  dump(X_test_token, path+'X_test_token_ref.pkl')

  #for model 3
  dump(X_train_s, path+'X_train_s_ref.pkl')
  dump(y_train_s, path+'y_train_s_ref.pkl')
  dump(X_dev_s, path+'X_dev_s_ref.pkl')
  dump(y_dev_s, path+'y_dev_s_ref.pkl')
  dump(X_test_s, path+'X_test_s_ref.pkl')
  dump(y_test_s, path+'y_test_s_ref.pkl')
  logging.info("generate complete")

def test():
  if len(load(path+'X_train_ref.pkl')) != len(load(path+'y_train_ref.pkl')):
    return False
  if len(load(path+'X_dev_ref.pkl')) != len(load(path+'y_dev_ref.pkl')):
    return False
  if len(load(path+'X_test_ref.pkl')) != len(load(path+'y_test_ref.pkl')):
    return False
  if len(load(path+'X_train_token_ref.pkl')) != len(load(path+'y_train_ref.pkl')):
    return False
  if len(load(path+'X_dev_token_ref.pkl')) != len(load(path+'y_dev_ref.pkl')):
    return False
  if len(load(path+'X_test_token_ref.pkl')) != len(load(path+'y_test_ref.pkl')):
    return False  
  if len(load(path+'X_train_s_ref.pkl')) != len(load(path+'y_train_s_ref.pkl')):
    return False
  if len(load(path+'X_dev_s_ref.pkl')) != len(load(path+'y_dev_s_ref.pkl')):
    return False
  if len(load(path+'X_test_s_ref.pkl')) != len(load(path+'y_test_s_ref.pkl')):
    return False  
  if len(load(path+'X_train_ref.pkl')) < 1000000:
    return False  
  if len(load(path+'X_train_s_ref.pkl')) < 100000:
    return False  
  logging.info("test complete")
  return True
 
def lg_generate(filename):
  train_data = pd.read_csv(path+filename)
  train_data = data_clean(train_data, 'train_ref.pkl')
  train_data_s = pd.concat([train_data.loc[(train_data['target'] == 0) & (train_data['question_text'].str.len() > 10)].sample(n=90000, random_state=42),\
                            train_data.loc[(train_data['target'] == 1) & (train_data['question_text'].str.len() > 10)].sample(n=80000, random_state=42)], ignore_index=True)
  train_data_s = train_data_s.sample(frac=1).reset_index(drop=True)
  
  X_train, X_dev, X_test, y_train, y_dev, y_test = threeway_split(train_data['question_text'], train_data['target'])
  logging.info("lg_generate complete")
  return X_train, X_dev, X_test, y_train, y_dev, y_test

def rnn_generate(filename):
  train_data = pd.read_csv(path+filename)
  train_data = data_clean(train_data, 'train_ref.pkl')
  train_data_s = pd.concat([train_data.loc[(train_data['target'] == 0) & (train_data['question_text'].str.len() > 10)].sample(n=90000, random_state=42),\
                            train_data.loc[(train_data['target'] == 1) & (train_data['question_text'].str.len() > 10)].sample(n=80000, random_state=42)], ignore_index=True)
  train_data_s = train_data_s.sample(frac=1).reset_index(drop=True)
  
  X_train, X_dev, X_test, y_train, y_dev, y_test = threeway_split(train_data['question_text'], train_data['target'])
  
  tokenizer = Tokenizer(num_words=num_words, lower=False, char_level=False)
  tokenizer.fit_on_texts(train_data['question_text'])
  X_train_token  = tokenizer.texts_to_sequences(X_train)
  X_dev_token  = tokenizer.texts_to_sequences(X_dev)
  X_test_token  = tokenizer.texts_to_sequences(X_test)
  X_train_token = pad_sequences(X_train_token, maxlen=max_tokens, padding=pad, truncating=pad).tolist()
  X_dev_token = pad_sequences(X_dev_token, maxlen=max_tokens, padding=pad, truncating=pad).tolist()
  X_test_token = pad_sequences(X_test_token, maxlen=max_tokens, padding=pad, truncating=pad).tolist()
  dump(tokenizer, path+'tokenizer_ref.pkl')
  logging.info("lg_generate complete and tokenizer dumpped to "+path+'/tokenizer_ref.pkl")
  return X_train_token, X_dev_token, X_test_token, y_train, y_dev, y_test, tokenizer
  
def lm_generate(filename):
  train_data = pd.read_csv(path+filename)
  train_data = data_clean(train_data, 'train_ref.pkl')
  train_data_s = pd.concat([train_data.loc[(train_data['target'] == 0) & (train_data['question_text'].str.len() > 10)].sample(n=90000, random_state=42),\
                            train_data.loc[(train_data['target'] == 1) & (train_data['question_text'].str.len() > 10)].sample(n=80000, random_state=42)], ignore_index=True)
  train_data_s = train_data_s.sample(frac=1).reset_index(drop=True)
  X_train_s, X_dev_s, X_test_s, y_train_s, y_dev_s, y_test_s = threeway_split(train_data_s['question_text'], train_data_s['target'])
  logging.info("lm_generate complete")
  return X_train_s, X_dev_s, X_test_s, y_train_s, y_dev_s, y_test_s


