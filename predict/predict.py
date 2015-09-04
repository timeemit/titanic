import pandas
from learn.trainer import get_classifier
from learn.trainer import D as winner
 
DATA = pandas.read_csv('predict/input.csv')

data = winner.data(DATA)
classifier = get_classifier(winner)

output = pandas.DataFrame({'PassengerId': DATA['passenger_id']})

survived = list()
for row in range(len(data)):
    survival = classifier.predict(data.iloc[row ,:])
    survived.append(survival[0])

output['Survived'] = pandas.Series(survived)
output.to_csv('predict/output.csv', index=False)
