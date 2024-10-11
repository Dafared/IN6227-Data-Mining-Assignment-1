import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time
import argparse

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):  
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        
        k_indices = np.argsort(distances)[:self.k]
        
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

parser = argparse.ArgumentParser(description='KNN Classifier')
parser.add_argument('--k', type=int, default=5, help='Number of neighbors for KNN (default: 5)')
args = parser.parse_args()

k = args.k

train_data = pd.read_csv("Census Income Data Set/adult.data", header=None, na_values=' ?', skipinitialspace=True)
test_data = pd.read_csv("Census Income Data Set/adult.test", header=None, na_values=' ?', skipinitialspace=True, skiprows=1)

train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]
y_test = y_test.str.replace('.', '', regex=False)

for col in X_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

y_train = LabelEncoder().fit_transform(y_train)
y_test = LabelEncoder().fit_transform(y_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNNClassifier(k)
knn.fit(X_train, y_train)

print("knn fit finish")

correct_predictions = 0
total_samples = len(y_test)
total_time = 0

start_time = time.time()
for i, sample in enumerate(tqdm(X_test, desc="Processing", unit="sample")):
    y_pred = knn.predict([sample])
    actual = y_test[i]
    
    if y_pred[0] == actual:
        correct_predictions += 1
end_time = time.time()
total_time = end_time - start_time

accuracy = correct_predictions / total_samples
print(f"\nTotal Accuracy: {accuracy * 100:.2f}%")

average_prediction_time = total_time / total_samples
print(f"Average prediction time per sample: {average_prediction_time:.6f} seconds")