import tensorflow_hub as hub
import tensorflow as tf
from joblib import dump, load
import pandas as pd
import numpy
import logging
import my_data
import my_utils

TOTAL_STEPS = 4000
STEP_SIZE = 500
path = my_utils.path

def train_lm(filename):
    X_train_s, X_dev_s, X_test_s, y_train_s, y_dev_s, y_test_s = my_data.generate_lm(filename)
    embedding_feature = hub.text_embedding_column(
        key='sentence', module_spec="https://tfhub.dev/google/universal-sentence-encoder/2", trainable=True)
  
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {'sentence': X_train_s.values}, y_train_s.values, 
        batch_size=256, num_epochs=None, shuffle=True)
    predict_train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {'sentence': X_train_s.values}, y_train_s.values, shuffle=False)
    predict_val_input_fn = tf.estimator.inputs.numpy_input_fn(
        {'sentence': X_dev_s.values},  y_dev_s.values, shuffle=False)
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        {'sentence': tf.io.FixedLenFeature([], tf.string)})
  
    dnn = tf.estimator.DNNClassifier(
        hidden_units=[512, 128],
        feature_columns=[embedding_feature],
        n_classes=2,
        activation_fn=tf.nn.relu,
        dropout=0.1,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.005),
        model_dir=path+'../exploration/models/',
        config=my_checkpointing_config)
    for step in range(0, TOTAL_STEPS+1, STEP_SIZE):
        logging.info('Training for step =', step)
        dnn.train(input_fn=train_input_fn, steps=STEP_SIZE)
        logging.info('Eval Metrics (Train):', dnn.evaluate(input_fn=predict_train_input_fn))
        logging.info('Eval Metrics (Validation):', dnn.evaluate(input_fn=predict_val_input_fn))
        logging.info('\n')

    export_dir = dnn.export_savedmodel(path+'../exploration/models/', serving_input_receiver_fn)        
    logging.info('export model to '+ export_dir)
    return dnn, export_dir

def predict_lm(sentence, export_dir):
    export_dir = b'../exploration/data/models/1569699519'
    # retrive model 
    predict_fn = tf.contrib.predictor.from_saved_model(export_dir)
    inputs = pd.DataFrame({'sentence': [my_data.str_clean(sentence)],})
    examples = []
    for index, row in inputs.iterrows():
        feature = {}
        for col, value in row.iteritems():
            value = str.encode(value)
            feature[col] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        examples.append(example.SerializeToString())

    predictions = predict_fn({'inputs': examples})
    for score in predictions['scores']:
        if score[0] > score[1]:
            return 'sincere'
        else :
            return 'insincere'

def get_predictions(estimator, input_fn):
    return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]

def test_lm(dnn):
    x = ['documents required at the time of interview in sbi']
    input_fn = tf.estimator.inputs.numpy_input_fn(
        {'sentence': numpy.array(x)},shuffle=False)
    logging.info(get_predictions(estimator=dnn, input_fn=input_fn))
    x = ['white felame is the best human']
    input_fn = tf.estimator.inputs.numpy_input_fn(
        {'sentence': numpy.array(x)},shuffle=False)
    logging.info(get_predictions(estimator=dnn, input_fn=input_fn))

