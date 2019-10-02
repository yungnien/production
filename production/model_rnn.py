from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential, save_model, load_model
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from sklearn.metrics import f1_score
from joblib import dump, load
import numpy as np
import my_data
import my_utils
import logging

num_words = my_utils.num_words
embedding_size = my_utils.embedding_size
max_tokens = my_utils.max_tokens
pad = my_utils.pad
path = my_utils.path

def load_para(word_index):
    EMBEDDING_FILE = path+'../embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.0053247833,0.49346462
    embed_size = all_embs.shape[1]

    nb_words = min(num_words, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= num_words: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

def train_rnn():
    tokenizer= load(path+'tokenizer_ref.pkl')
    X_train_token = load(path +'X_train_token_ref.sav')
    X_dev_token = load(path +'X_dev_token_ref.sav')
    y_train = load(path +'y_train_ref.sav')
    y_dev = load(path +'y_dev_ref.sav')
    paragram_embeddings = load_para(tokenizer.word_index)
    
    model = Sequential()
    optimizer = Adam(lr=1e-3)
    model.add(Embedding(weights=[paragram_embeddings], trainable=False, input_dim=num_words, output_dim=embedding_size, input_length=max_tokens))
    model.add(GRU(units=32, return_sequences=True))
    model.add(GRU(units=16, dropout=0.5, return_sequences=True))
    model.add(GRU(units=8, return_sequences=True))
    model.add(GRU(units=4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['AUC', 'accuracy'])
    model.summary()
    history = model.fit(np.array(X_train_token), y_train, validation_data=(np.array(X_dev_token),y_dev), epochs=4, batch_size=500)
    save_model(model,path+'rnn_model_ref.h5')
    logging.info('train complete')

def train_rnn(filename):
    X_train_token, X_dev_token, X_test_token, y_train, y_dev, y_test, tokenizer = my_data.generate_rnn(filename)
    paragram_embeddings = load_para(tokenizer.word_index)
    
    model = Sequential()
    optimizer = Adam(lr=1e-3)
    model.add(Embedding(weights=[paragram_embeddings], trainable=False, input_dim=num_words, output_dim=embedding_size, input_length=max_tokens))
    model.add(GRU(units=32, return_sequences=True))
    model.add(GRU(units=16, dropout=0.5, return_sequences=True))
    model.add(GRU(units=8, return_sequences=True))
    model.add(GRU(units=4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['AUC', 'accuracy'])
    model.summary()
    history = model.fit(np.array(X_train_token), y_train, validation_data=(np.array(X_dev_token),y_dev), epochs=4, batch_size=500)
    save_model(model,path+'rnn_model_ref.h5')
    logging.info('train complete')
    return model

def predict_rnn_token(X_token):
    model = load_model(path+'rnn_model_ref.h5')
    predicted = model.predict(np.array(X_token))
    predicted = predicted.T[0]
    cls_pred = np.array([1.0 if p>0.5 else 0.0 for p in predicted])
    logging.info('return prediction')
    return cls_pred

def test_rnn_token(X_token, y_value):
    model = load_model(path+'rnn_model_ref.h5')
    predicted = model.predict(np.array(X_token))
    predicted = predicted.T[0]
    cls_pred = np.array([1.0 if p>0.5 else 0.0 for p in predicted])  
    return f1_score(y_value, cls_pred, average='weighted') > 0.90

def predict_rnn(question):
    tokenizer= load(path+'tokenizer_ref.pkl')
    X_token = tokenizer.texts_to_sequences([my_data.str_clean(question)])
    X_token = pad_sequences(X_token, maxlen=max_tokens, padding=pad, truncating=pad).tolist()
    result = predict_rnn_token(X_token)
    logging.info('predict rnn: ' + question + ' result' )
    return result

