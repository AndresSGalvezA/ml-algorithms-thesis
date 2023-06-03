from pandas import read_csv, concat
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

green_data = read_csv('C:/Users/asgas/Desktop/Prototipo tesis/Data/TrainingValidation/green_banana_data.csv')
light_data = read_csv('C:/Users/asgas/Desktop/Prototipo tesis/Data/TrainingValidation/light_banana_data.csv')
yellow_data = read_csv('C:/Users/asgas/Desktop/Prototipo tesis/Data/TrainingValidation/yellow_banana_data.csv')
black_data = read_csv('C:/Users/asgas/Desktop/Prototipo tesis/Data/TrainingValidation/black_banana_data.csv')

green_data['label'] = 'green'
light_data['label'] = 'light'
yellow_data['label'] = 'yellow'
black_data['label'] = 'black'

data = concat([green_data, light_data, yellow_data, black_data], ignore_index=True)

X = data.iloc[:, :-1].values
y = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.22, random_state = 0)

best_params = {'max_depth': 6, 'min_samples_leaf': 5, 'min_samples_split': 2}
clf = DecisionTreeClassifier(**best_params, random_state = 0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy score:\n", accuracy_score(y_test, y_pred))

sample = array([[185, 1264, 758, 893, 1122, 1063, 838, 473, 176]])
print("Prediction:", clf.predict(sample))
