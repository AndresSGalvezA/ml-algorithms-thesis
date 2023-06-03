from pandas import read_csv, concat
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

green_data = read_csv('C:/Users/asgas/Desktop/Prototipo tesis/Data/TrainingValidation/green_banana_data.csv')
light_data = read_csv('C:/Users/asgas/Desktop/Prototipo tesis/Data/TrainingValidation/light_banana_data.csv')
yellow_data = read_csv('C:/Users/asgas/Desktop/Prototipo tesis/Data/TrainingValidation/yellow_banana_data.csv')
black_data = read_csv('C:/Users/asgas/Desktop/Prototipo tesis/Data/TrainingValidation/black_banana_data.csv')

green_data['label'] = 'green'
light_data['label'] = 'light'
yellow_data['label'] = 'yellow'
black_data['label'] = 'black'

data = concat([green_data, light_data, yellow_data, black_data], ignore_index = True)

X = data.iloc[:, :-1].values
y = data['label'].values

label_encoder = LabelEncoder()
label_encoder.fit(y)

y_encoded = label_encoder.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.22, random_state = 0)

best_params = {'colsample_bytree': 1.0, 'gamma': 0, 'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 300, 'subsample': 0.5}
clf = XGBClassifier(**best_params, use_label_encoder = False, eval_metric = 'mlogloss', random_state = 0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

print("Classification Report:\n", classification_report(y_test_labels, y_pred_labels))
print("Confusion Matrix:\n", confusion_matrix(y_test_labels, y_pred_labels))
print("Accuracy score:\n", accuracy_score(y_test_labels, y_pred_labels))

sample = array([[185, 1264, 758, 893, 1122, 1063, 838, 473, 176]])

predicted_label_encoded = clf.predict(sample)
predicted_label = label_encoder.inverse_transform(predicted_label_encoded)
print("Prediction: ", predicted_label)