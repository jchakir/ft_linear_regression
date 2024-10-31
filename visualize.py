import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import isfile
from sys import exit

plt.style.use('ggplot')

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

# Visulizer
def visualize() -> None:
  t0, t1 = np.load(MODEL_PARAMS_PATH)
  _, ax = plt.subplots(figsize=(10, 6))
  ax.scatter(x=X, y=y)
  ax.plot(X, np.mean(y) + X*0, color="b", label="Mean")
  ax.plot(X, t0 + t1 * X, color="g", label="Best Fit")
  # Set title and labels
  ax.set_title("Mileage/Price")
  ax.set_xlabel("Mileage")
  ax.set_ylabel("Price")
  ax.legend()
  plt.show()

if __name__ == '__main__':
  visualize()

