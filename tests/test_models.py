from production import my_utils, my_data
import pandas as pd
import os
from sklearn.metrics import accuracy_score

validation_data = os.path.join(os.path.dirname(__file__), "validation_data/train.csv")

def test_model():
    X_train, X_dev, X_test, y_train, y_dev, y_test = my_data.generate_lg_test(validation_data)
    return test_lg(X_dev, y_dev)

