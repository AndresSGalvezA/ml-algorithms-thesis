import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from os import chdir

chdir('C:/Users/asgas/Desktop')

df_green = pd.read_csv('C:/Users/asgas/Desktop/Prototipo tesis/Data/Training and validation data/green_banana_data.csv')
df_light = pd.read_csv('C:/Users/asgas/Desktop/Prototipo tesis/Data/Training and validation data/light_banana_data.csv')
df_yellow = pd.read_csv('C:/Users/asgas/Desktop/Prototipo tesis/Data/Training and validation data/yellow_banana_data.csv')
df_black = pd.read_csv('C:/Users/asgas/Desktop/Prototipo tesis/Data/Training and validation data/black_banana_data.csv')

df_green['label'] = 'green'
df_light['label'] = 'light'
df_yellow['label'] = 'yellow'
df_black['label'] = 'black'

data = pd.concat([df_green, df_light, df_yellow, df_black])

X = data.iloc[:, :-1].values
y = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

best_params = {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
clf = RandomForestClassifier(**best_params, random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy score:\n", accuracy_score(y_test, y_pred))

sample = np.array([[185, 1264, 758, 893, 1122, 1063, 838, 473, 3022, 176]])
print("Prediction:", clf.predict(sample))