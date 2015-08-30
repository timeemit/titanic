import pandas
from numpy import nan
from sklearn import tree

NAME = 'Tree with Mean Age'
PICKLE = 'learn/tree_with_mean_age/tree.pkl'

FEATURES = [
    'age',
    'pclass',
    'sibsp',
    'parch',
    'fare',
    'gender',
    'cabin_class',
    'elder_travelers', 
    'younger_travelers',
    'peer_travelers',
    'familial_travelers'
]

GRID = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [8, 16, 32, 64, 128],
    'min_samples_split': [1, 2, 4, 8, 16, 32],
    'min_samples_leaf': [1, 2, 4, 8, 16],
    'max_leaf_nodes': [None, 16, 32, 64, 128, 256]
}

def data(imputed_data):
    X = pandas.DataFrame()
    for feature in FEATURES:
        X[feature] = imputed_data[feature]

    # Impute average age instead of -1
    for i in range(len(X)):
        if X.iloc[i, 0] == -1.0:
            print('False age found!')
            X[i, 0] = nan
    X = X.fillna(X.mean()['age'])

    return X

def classifier():
    classifier = tree.DecisionTreeClassifier(random_state=0)
