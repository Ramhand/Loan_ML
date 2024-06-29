import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import pickle
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn

try:
    with open('loan.dat', 'rb') as file:
        data = pickle.load(file)
except FileNotFoundError:
    data = 'https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv'
    data = pd.read_csv(data, sep=';')
    with open('./data/raw/loan.dat', 'wb') as file:
        pickle.dump(data, file)
finally:
    data.drop_duplicates(inplace=True)


def data_flattener(datum):
    if datum == 'yes' or datum == 'success':
        return 1
    elif datum == 'no' or datum == 'failure':
        return 0
    else:
        return np.nan


# for i in data.select_dtypes(include=['object']).columns.to_list():
#     if data[i].nunique() == 2:
#         data[i] = data[i].factorize()[0]
#     elif data[i].nunique() == 3:
#         data[i] = data[i].apply(data_flattener)
categorical = data.select_dtypes(include=['object']).columns.to_list()
num_num = len(data.columns) - len(categorical)
# fig, axs = plt.subplots(num_num + 1, 1, figsize=(5, 7))
# x = 0
# while x != num_num:
#     for j in data.columns:
#         if j not in categorical:
#             sns.boxplot(ax=axs[x], data=data[j])
#             x += 1
# sns.heatmap(data.drop(categorical, axis=1).corr(), ax=axs[num-num], fmt='.2f', cbar=True, annot=True)
# plt.show()
# imp = SimpleImputer(strategy='most_frequent')
# imp.fit_transform(data)
data.drop(['age', 'default', 'housing', 'loan', 'campaign', 'cons.conf.idx'], axis=1, inplace=True)
one_hot = OneHotEncoder(sparse_output=False)
categorical = data.select_dtypes(include=['object']).columns.to_list()
categorical.pop(categorical.index('y'))
data_hot = one_hot.fit_transform(data[categorical])
one_hot_data = pd.concat([data, pd.DataFrame(data_hot, columns=one_hot.get_feature_names_out(categorical))], axis=1)
one_hot_data.drop(categorical, axis=1, inplace=True)
one_hot_data.dropna(axis=0, inplace=True)

X = one_hot_data.drop('y', axis=1)
Y = one_hot_data['y']
scaler = MinMaxScaler()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_test_scaled = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

try:
    with open('./models/model.dat', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, Y_train)
    with open('./models/model.dat', 'wb') as file:
        pickle.dump(model, file)
finally:
    acc_check = model.predict(X_test_scaled)
    score = accuracy_score(Y_test, acc_check)
    print('Base Accuracy:', f'{score * 100: .2f}%')

hyperparams = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "penalty": ["l1", "l2", "elasticnet", None],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}

grid = GridSearchCV(model, hyperparams, scoring="accuracy", cv=5)
grid.fit(X_train_scaled, Y_train)
print('Grid Search Results:', grid.best_params_, '\nGrid Search Score:', f'{grid.best_score_ * 100: .2f}%')

hyperparams = {
    "C": np.logspace(-4, 4, 100),
    "penalty": ["l1", "l2", "elasticnet", None],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}

randome = RandomizedSearchCV(model, hyperparams, n_iter=100, scoring='accuracy', cv=5, random_state=42)
randome.fit(X_train_scaled, Y_train)
print('Random Search Results:', randome.best_params_, '\nRandom Search Score:', f'{randome.best_score_ * 100: .2f}%')
