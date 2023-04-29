import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import cycle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

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

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_binarized = label_binarize(y_encoded, classes=list(range(4)))

#y_binarized = label_binarize(y, classes=['black', 'green', 'light', 'yellow'])
n_classes = y_binarized.shape[1]

X_train, X_test, y_train_binarized, y_test_binarized = train_test_split(X, y_binarized, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

best_rf_params = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
best_svm_params = {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'}
best_knn_params = {'metric': 'euclidean', 'n_neighbors': 1, 'weights': 'uniform'}
best_dt_params = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
best_xgb_params = {'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 200, 'subsample': 1.0}

rf_clf = RandomForestClassifier(**best_rf_params, random_state=0)
svm_clf = SVC(**best_svm_params, probability=True, random_state=0)
knn_clf = KNeighborsClassifier(n_neighbors=best_knn_params['n_neighbors'], weights=best_knn_params['weights'], metric=best_knn_params['metric'])
dt_clf = DecisionTreeClassifier(**best_dt_params, random_state=0)
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', **best_xgb_params, random_state=0)

rf_clf.fit(X_train, y_train)
svm_clf.fit(X_train, y_train)
knn_clf.fit(X_train, y_train)
dt_clf.fit(X_train, y_train)
xgb_clf.fit(X_train, y_train)

models = [
    ('Random Forest', rf_clf),
    ('Support Vector Machine', svm_clf),
    ('K Neighbors', knn_clf),
    ('Decision Tree', dt_clf),
    ('XGBoost', xgb_clf)
]

def plot_multiclass_roc(clf, X_test, y_test_binarized, n_classes, title):
    y_score = clf.predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]))
    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

for name, model in models:
    plot_multiclass_roc(model, X_test, y_test_binarized, 4, f"{name} Multi-class ROC")