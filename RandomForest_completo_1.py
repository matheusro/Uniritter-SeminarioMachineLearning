import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot

dados_dengue = pd.read_csv('dados/caso-dengue2018_C.csv', delimiter=';',  low_memory=False)

X = dados_dengue.drop(['tp_sexo','tp_classificacao_final','tp_criterio_confirmacao', 'resultado'], axis=1)
y = dados_dengue['resultado']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=2, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
rfc.fit(X_train, y_train)
rfc_predict = rfc.predict(X_test)
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')
rfc_cv_score_AC = cross_val_score(rfc, X, y, cv=10, scoring='accuracy')

# calcula AUC e curva ROC
rfc_probs = rfc.predict_proba(X_test)
rfc_probs = rfc_probs[:, 1]
rfc_auc_score = roc_auc_score(y_test, rfc_probs)
fpr, tpr, tresholds = roc_curve(y_test, rfc_probs)

pyplot.plot([0, 1], [0, 1], linestyle='--', color='darkblue')
pyplot.plot(fpr, tpr, color='orange', label='ROC')
pyplot.title('Curva ROC')
pyplot.xlabel('Taxa de falsos positivos')
pyplot.ylabel('Taxa de verdadeiros positivos')
pyplot.legend()
pyplot.show()

print("=== Matriz de confusão ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Relatório de Classificação ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Média AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
print('\n')
print("=== Acuracia ===")
print(rfc_cv_score_AC.mean())