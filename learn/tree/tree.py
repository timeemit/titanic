import os
import pickle
import pandas
from sklearn import tree
from sklearn.grid_search import GridSearchCV

DATA = pandas.read_csv('train/2.train.cross.csv')

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

if not os.path.isfile('learn/tree/tree.pkl'):
    print('No pickle file found, training a new classifier')

    X = pandas.DataFrame()
    for feature in FEATURES:
        X[feature] = DATA[feature]

    Y = DATA['survived']

    classifier = tree.DecisionTreeClassifier(random_state=0)
    classifier = GridSearchCV(
        estimator=classifier,
        param_grid=GRID,
        verbose=100,
        n_jobs=-1
    )

    classifier.fit(X, Y)

    with open(PICKLE, 'wb') as output:
        pickle.dump(classifier, output)

else:
    print('Loading pickle file!')
    classifier = None
    with open(PICKLE, 'rb') as file_stream:
        classifier = pickle.load(file_stream)


print(classifier.best_score_)
print(classifier.best_params_)
