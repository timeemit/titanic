import os
import pickle
import pandas
from sklearn.grid_search import GridSearchCV
import tree.tree as A
import tree.tree_with_age as B
import tree.tree_with_mean_age as C
import forest.forest as D


DATA = pandas.read_csv('train/2.train.cross.csv')
TEST = pandas.read_csv('train/3.test.csv')

MODELS = [ A, B, C, D ]

def build_new_classifier(model):
    print('No pickle file found, training a new classifier')

    X = model.data(DATA)
    Y = DATA['survived']
    classifier = model.classifier() 

    classifier = GridSearchCV(
        estimator=classifier,
        param_grid=model.GRID,
        verbose=100,
    )

    classifier.fit(X, Y)

    with open(model.PICKLE, 'wb') as output:
        pickle.dump(classifier, output)

    return classifier

def retrieve_classifier(model):
    print('Loading pickle file!')

    classifier = None
    with open(model.PICKLE, 'rb') as file_stream:
        classifier = pickle.load(file_stream)

    return classifier

def get_classifier(model):
    if not os.path.isfile(model.PICKLE):
        classifier = build_new_classifier(model)
    else:
        classifier = retrieve_classifier(model)

    return classifier

def compare(x, y):
    df = pandas.DataFrame()
    for model in MODELS:
        print('Considering model: {}'.format(model.NAME))

        classifier = get_classifier(model)
        
        series = pandas.Series([
           classifier.best_score_,
           classifier.score(model.data(x), y)
        ])
        df[model.NAME] = series
    return df

if __name__ == '__main__':
    x, y = DATA, DATA['survived']
    comparison = compare(x, y)
    print(comparison)
