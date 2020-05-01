from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
import glob
import os
import random
import re
import string

import joblib

from utils.plot import plot_cm

WEIGHTS_DIR = 'weights/'


def latest_modified_weight():
    """
    returns latest trained weight
    :return: model weight trained the last time
    """
    weight_files = glob.glob(WEIGHTS_DIR + '*')
    latest = max(weight_files, key=os.path.getctime)
    return latest


def load_model():
    """
    :param path: weight path
    :return: load model based on the path
    """
    path = latest_modified_weight()

    with open(path, 'rb') as f:
        return joblib.load(filename=f)


def dump_model(model, name):
    model_name = WEIGHTS_DIR + name + generate_model_name(5) + '.pkl'
    with open(model_name, 'wb') as f:
        joblib.dump(value=model, filename=f, compress=3)
        print(f'Model saved at {model_name}')


def generate_model_name(size=5):
    """
    :param size: name length
    :return: random lowercase and digits of length size
    """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))


def train_model(data, args):
    if args.gs:
        params = {
            'gamma': [0.1, 1, 10, 100],
            'C': [0.1, 1, 10, 100, 200],
            'degree': [2, 3, 4, 5, 6],
            'kernel':['linear','rbf','poly']
        }
        gs = GridSearchCV(estimator=SVC(), param_grid=params, cv=KFold(10, True, 42))
        gs.fit(data['train']['features'], data['train']['labels'])

        model = gs.best_estimator_
        preds = model.predict(data['test']['features'])

        cm = confusion_matrix(data['test']['labels'], preds)
        score = accuracy_score(data['test']['labels'], preds)
        best_params = re.sub("[{}' ,()]", '', str(gs.best_params_))

        plot_cm(cm, f'cm-accuracy:{score:.2f}-{best_params}-SVM-gs')

        ans = input('Do you want to save the model weight? ')
        if ans in ('yes', '1'):
            dump_model(model, 'SVM-gs')

        return model

    else:

        model = SVC()
        model.fit(data['train']['features'], data['train']['labels'])
        preds = model.predict(data['test']['features'])

        cm = confusion_matrix(data['test']['labels'], preds)
        score = accuracy_score(data['test']['labels'], preds)

        plot_cm(cm, f'cm-accuracy:{score:.2f}SVM')

        ans = input('Do you want to save the model weight? ')
        if ans in ('yes', '1'):
            dump_model(model, 'SVM')

        return model
