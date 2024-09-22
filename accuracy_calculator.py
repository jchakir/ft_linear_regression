import numpy as np
import pandas as pd
from os.path import isfile
from sys import exit

# Local Varibles
DATASET_PATH = './data/data.csv'
MODEL_PARAMS_PATH = './model/params.npy'

# Basic Checks
if not isfile(DATASET_PATH):
  print('data file not found, make sure is exist')
  exit(1)

if not isfile(MODEL_PARAMS_PATH):
  print('model parameters file is not found, try train the model before')
  exit(1)

# Load Dataset from file
data = pd.read_csv(DATASET_PATH)
X = data.values[:, 0:1]
y = data.values[:, 1:2]

# Accuracy Calculator
def accuracy() -> float:
  th0, th1 = np.load(MODEL_PARAMS_PATH)
  yHat = th0 + th1 * X
  rss = np.sum(np.power(y - yHat, 2))
  tss = np.sum(np.power(y - np.mean(y), 2))
  r2 = 1 - rss / tss
  return r2 * 100

if __name__ == '__main__':
  acc = accuracy()
  print('Accuracy:', acc)
