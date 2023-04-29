import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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

param_grid = {
    'n_neighbors': list(range(1, 31)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

clf = KNeighborsClassifier()

grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)
print("Best accuracy score:", grid_search.best_score_)

best_clf = grid_search.best_estimator_
best_clf.fit(X_train, y_train)
accuracy = best_clf.score(X_test, y_test)
print("Accuracy on test set:", accuracy)