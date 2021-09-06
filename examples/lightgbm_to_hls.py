import numpy as np
from scipy.special import expit
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, Dataset, Booster, plot_tree, train
import conifer
import datetime
from pprint import pprint
import matplotlib.pyplot as plt

# Make a random dataset from sklearn 'hastie'
dataset = load_boston()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(y_train[:5])

train_set = Dataset(X_train, label=y_train)

params = {
    'objective': 'regression',
    'n_estimators': 100,
    'max_depth': 4
}
# Train a BDT
reg = train(params, train_set=train_set)

# plot_tree(clf, dpi=400)
plt.show()

booster: Booster = reg
print(booster.trees_to_dataframe())
print(booster.num_trees())
# pprint(booster.dump_model())

# Create a conifer config
cfg = conifer.backends.vivadohls.auto_config()
# Set the output directory to something unique
cfg['OutputDir'] = 'prj_{}'.format(int(datetime.datetime.now().timestamp()))

# Create and compile the model
model = conifer.model(reg, conifer.converters.lightgbm, conifer.backends.vivadohls, cfg)
model.compile()

# Run HLS C Simulation and get the output
y_hls = expit(model.decision_function(X))
y_skl = reg.predict(X)

assert np.allclose(y_hls, y_skl), f'{y_hls} != {y_skl}'

# Synthesize the model
model.build()
