import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def plot_multiclass_roc(clf, X_test, y_test, n_classes, title):
    y_score = clf.predict_proba(X_test)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    plt.show()

filenames = ['C:/Users/asgas/Desktop/Prototipo tesis/Data/Training and validation data/green_banana_data.csv', 
             'C:/Users/asgas/Desktop/Prototipo tesis/Data/Training and validation data/light_banana_data.csv', 
             'C:/Users/asgas/Desktop/Prototipo tesis/Data/Training and validation data/yellow_banana_data.csv', 
             'C:/Users/asgas/Desktop/Prototipo tesis/Data/Training and validation data/black_banana_data.csv'
            ]
colors = ['green', 'light', 'yellow', 'black']

dataframes = []
for color, filename in zip(colors, filenames):
    df = pd.read_csv(filename)
    df['label'] = color
    dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)
X = data.iloc[:, :-1].values
y = data['label'].values

y_binarized = label_binarize(y, classes=['green', 'light', 'yellow', 'black'])

X_train, X_test, y_train, y_test_binarized = train_test_split(X, y_binarized, test_size=0.2, random_state=0)

rf_clf = RandomForestClassifier()
svm_clf = OneVsRestClassifier(SVC(kernel='rbf', probability=True))
knn_clf = KNeighborsClassifier()
dt_clf = DecisionTreeClassifier()
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

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

for name, model in models:
    plot_multiclass_roc(model, X_test, y_test_binarized, 4, f"{name} Multi-class ROC")