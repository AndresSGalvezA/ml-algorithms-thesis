import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

green_data = pd.read_csv('C:/Users/asgas/Desktop/Prototipo tesis/Data/Training and validation data/green_banana_data.csv')
light_data = pd.read_csv('C:/Users/asgas/Desktop/Prototipo tesis/Data/Training and validation data/light_banana_data.csv')
yellow_data = pd.read_csv('C:/Users/asgas/Desktop/Prototipo tesis/Data/Training and validation data/yellow_banana_data.csv')
black_data = pd.read_csv('C:/Users/asgas/Desktop/Prototipo tesis/Data/Training and validation data/black_banana_data.csv')

green_data['label'] = 'green'
light_data['label'] = 'light'
yellow_data['label'] = 'yellow'
black_data['label'] = 'black'

data = pd.concat([green_data, light_data, yellow_data, black_data], ignore_index=True)

X = data.iloc[:, :-1].values
y = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

best_params = {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'}
clf = SVC(**best_params, random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy score:\n", accuracy_score(y_test, y_pred))

sample = np.array([[185, 1264, 758, 893, 1122, 1063, 838, 473, 3022, 176]])
sample_scaled = scaler.transform(sample)
print("Prediction: ", clf.predict(sample_scaled))