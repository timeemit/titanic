import pandas
from sklearn import tree
from sklearn.grid_search import GridSearchCV

DATA        = pandas.read_csv('train/1.train.csv')
VALIDATION  = pandas.read_csv('train/2.cross_validation.csv')

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

# 1. Train

X = pandas.DataFrame()
for feature in FEATURES:
    X[feature] = DATA[feature]

Y = DATA['survived']

VALIDATION['survived']

classifier = tree.DecisionTreeClassifier(random_state=0)
classifier = GridSearchCV(
    estimator=classifier,
    param_grid=GRID,
)

print(classifier)

