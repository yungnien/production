import tensorflow_hub as hub
import tensorflow as tf
from joblib import dump, load
import numpy

TOTAL_STEPS = 4000
STEP_SIZE = 500
path = '/content/drive/My Drive/data/'

# def lm_train():
# def lm_predict(X_token):
# def lm_test(X_token, y_value):

X_train_s = load(path + 'X_train_s_ref.pkl')
X_test_s = load(path + 'X_test_s_ref.pkl')
X_dev_s = load(path + 'X_dev_s_ref.pkl')
y_train_s = load(path + 'y_train_s_ref.pkl')
y_test_s = load(path + 'y_test_s_ref.pkl')
y_dev_s = load(path + 'y_dev_s_ref.pkl')

X_dev = load(path + 'X_dev_ref.pkl')
y_dev = load(path + 'y_dev_ref.pkl')

# Retain the 2 most recent checkpoints.
my_checkpointing_config = tf.estimator.RunConfig(
    keep_checkpoint_max=2,
)
# Training input on the whole training set with no limit on training epochs.
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'sentence': X_train_s.values}, y_train_s.values,
    batch_size=256, num_epochs=None, shuffle=True)
# Prediction on the whole training set.
predict_train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'sentence': X_train_s.values}, y_train_s.values, shuffle=False)
# Prediction on the whole validation set.
predict_val_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'sentence': X_dev_s.values}, y_dev_s.values, shuffle=False)
# Prediction on the test set.
predict_test_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'sentence': X_dev.values}, y_dev.values, shuffle=False)


def get_predictions(estimator, input_fn):
    return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]


def train_and_evaluate_with_sentence_encoder(hub_module, train_module=False, path=''):
    embedding_feature = hub.text_embedding_column(
        key='sentence', module_spec=hub_module, trainable=train_module)

    print('Training with', hub_module)
    print('Trainable is:', train_module)

    dnn = tf.estimator.DNNClassifier(
        hidden_units=[512, 128],
        feature_columns=[embedding_feature],
        n_classes=2,
        activation_fn=tf.nn.relu,
        dropout=0.1,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.005),
        model_dir=path,
        config=my_checkpointing_config)

    for step in range(0, TOTAL_STEPS + 1, STEP_SIZE):
        print('Training for step =', step)
        dnn.train(input_fn=train_input_fn, steps=STEP_SIZE)
        print('Eval Metrics (Train):', dnn.evaluate(input_fn=predict_train_input_fn))
        print('Eval Metrics (Validation):', dnn.evaluate(input_fn=predict_val_input_fn))
        print('\n')

    predictions_train = get_predictions(estimator=dnn, input_fn=predict_train_input_fn)
    predictions_dev = get_predictions(estimator=dnn, input_fn=predict_test_input_fn)
    return predictions_train, predictions_dev, dnn


tf.logging.set_verbosity(tf.logging.ERROR)

predictions_test, predictions_dev, dnn = train_and_evaluate_with_sentence_encoder(
    "https://tfhub.dev/google/universal-sentence-encoder/2", train_module=True, path=path + 'storage/models/refact/')

# report(y_dev.values, predictions_dev)
# plot_roc(y_dev.values, predictions_dev)
# store_matrix("use-512-with-training (dev)", y_dev.values, predictions_dev)
# store_matrix("use-512-with-training (train)", y_train_s.values, predictions_train)


x = ['documents required at the time of interview in sbi']
input_fn = tf.estimator.inputs.numpy_input_fn(
    {'sentence': numpy.array(x)}, shuffle=False)
print(get_predictions(estimator=dnn, input_fn=input_fn))
x = ['white felame is the best human']
input_fn = tf.estimator.inputs.numpy_input_fn(
    {'sentence': numpy.array(x)}, shuffle=False)
print(get_predictions(estimator=dnn, input_fn=input_fn))


# save model
feature_spec = {
    'sentence': tf.io.FixedLenFeature([], tf.string)
}
serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
export_dir = dnn.export_savedmodel(path+'storage/models/export', serving_input_receiver_fn)
export_dir


#import tensorflow_hub as hub
#import tensorflow as tf
#from joblib import dump, load
#import pandas as pd

export_dir = b'/content/drive/My Drive/data/storage/models/export/1569699519'
# retrive model
predict_fn = tf.contrib.predictor.from_saved_model(export_dir)

inputs = pd.DataFrame({
    'sentence': ['documents required at the time of interview in sbi','white felame is the best human'],
})

examples = []
for index, row in inputs.iterrows():
    feature = {}
    for col, value in row.iteritems():
        value = str.encode(value)
        feature[col] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    example = tf.train.Example(
        features=tf.train.Features(
            feature=feature
        )
    )
    examples.append(example.SerializeToString())

predictions = predict_fn({'inputs': examples})
for score in predictions['scores']:
    if score[0] > score[1]:
        print('sincere')
    else :
        print('insincere')
