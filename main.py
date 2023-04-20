from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neural_network import BernoulliRBM
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np
import json
# Load and preprocess the data


class PcapData:
    def __init__(self, data):
        self.__dict = json.loads(data)


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
        pcap_data = PcapData(data)
        return pcap_data


def load_data():
    flow = []
    for line in open('data/flow.json', 'r'):
        flow.append(json.loads(line))
    x = []
    for num, line in enumerate(flow):
        src_port = line.get("src_port", None)
        dst_port = line.get("dst_port", None)
        dst4_addr = line.get("dst4_addr", None)
        temp = [src_port, dst_port, dst4_addr]
        if None not in temp:
            x.append(temp)

    return x

print(len(load_data()))

#
# X_train, X_test, y_train, y_test = load_data()
#
#
# def preprocess_data(X_train, X_test):
#     pass
#
#
# X_train = preprocess_data(X_train)
# X_test = preprocess_data(X_test)
#
# # Train the DBN
# dbn = BernoulliRBM(n_components=100, learning_rate=0.01, n_iter=10, verbose=0)
# dbn.fit(X_train)
#
# # Extract features
# X_train_features = dbn.transform(X_train)
# X_test_features = dbn.transform(X_test)
#
# # Train the SVM
# svm = SVC(kernel='rbf', C=1, gamma=0.1)
# svm.fit(X_train_features, y_train)
#
# # Create a pipeline for DBN-SVM
# dbn_svm = Pipeline(steps=[('dbn', dbn), ('svm', svm)])
#
# # Evaluate the model
# y_pred = dbn_svm.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)
#
# print('Accuracy:', accuracy)
# print('Precision:', precision)
# xprint('F1-score:', f1_score)
