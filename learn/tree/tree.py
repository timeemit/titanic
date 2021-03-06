import pandas
from sklearn import tree

NAME = 'Tree'
PICKLE = 'learn/tree/tree.pkl'

FEATURES = [
    # Ignoring age, since it is not always known
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
    
    return X

def classifier():
    return tree.DecisionTreeClassifier(random_state=0)
