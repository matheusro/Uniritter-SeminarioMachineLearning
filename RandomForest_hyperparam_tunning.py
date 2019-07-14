#import libraries for pre-processing
import warnings

from sklearn.neighbors import KNeighborsClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np

from dateutil.parser import parse
from datetime import datetime
from scipy.stats import norm

# import all what you need for machine learning
import sklearn
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV

dados_dengue = pd.read_csv('dados/caso-dengue2018_C.csv', delimiter=';',  low_memory=False)

X = dados_dengue.drop(['tp_sexo','tp_classificacao_final','tp_criterio_confirmacao', 'resultado'], axis=1)
y = dados_dengue['resultado']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                             max_depth=10, max_features='auto', max_leaf_nodes=None,
                             min_impurity_decrease=0.0, min_impurity_split=None,
                             min_samples_leaf=1, min_samples_split=2,
                             min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=None,
                             oob_score=False, random_state=0, verbose=0, warm_start=False)
rfc.fit(X_train, y_train)
rfc_predict = rfc.predict(X_test)



param_grid = [
{'n_estimators': [100, 250, 500], 'max_features': [5, 10, 'auto'],
 'max_depth': [10, 50, None], 'bootstrap': [True, False]}
]

grid_search_forest = GridSearchCV(rfc, param_grid, cv=10, scoring='roc_auc')
grid_search_forest.fit(X_train, y_train)