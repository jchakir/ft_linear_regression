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

# Load Dataset from file
data = pd.read_csv(DATASET_PATH)
X = data.values[:, 0:1]
y = data.values[:, 1:2]


# Training Class for Linear Regression
class LinearRegression:
  def __init__(self) -> None:
    """
    Initialize learnable paramettres (theta0 and theta1)
    """
    self.__theta0: float = 0
    self.__theta1: float = 0
    self.__standardization: tuple[float, float, float, float]
    self.__X: np.ndarray
    self.__y: np.ndarray
    self.__data_size: int

  def __forward(self) -> np.ndarray:
    """
    Forward Propagation: takes X as input and return yHat as output
    Respecting this rule yHat = theta0 + theta1 * X
    """
    return self.__theta0 + self.__theta1 * self.__X

  def __backward(self, yHat: np.ndarray, alpha: float) -> float:
    """
    Backward Propagation throught derivite of (mse) mean-squared-error
    yHat: predicted label, alpha: learning_rate
    return mean-squared-error value
    """
    yHat_y = yHat - self.__y
    self.__theta0 -= alpha * (1/self.__data_size) * np.sum(yHat_y)
    self.__theta1 -= alpha * (1/self.__data_size) * np.sum(yHat_y.T.dot(self.__X))
    return (1 / self.__data_size) * np.sum(np.power(yHat_y, 2))

  def __save(self) -> None:
    """
    save theta0 and theta1 into binnary file for later use with numpy.save
    first: re-transform theta0 and theta1 to real dimmention
    this operation done using this formula https://datascience.stackexchange.com/questions/49765/
    """
    t0, t1 = self.__theta0, self.__theta1
    meanx, stdx, meany, stdy = self.__standardization

    theta0 = t0*stdy + meany - meanx*t1*stdy/stdx
    theta1 = stdy*t1/stdx
    np.save(MODEL_PARAMS_PATH, [theta0, theta1])

  def __check_shapes(self, X: np.ndarray, y: np.ndarray) -> None:
    if X.shape[0] == 0 or X.shape[0] != y.shape[0]:
      raise Exception('Error in data shape: empty or none symetrique, Please double check !')

  def __transform_by_standardization(self, X: np.ndarray, y: np.ndarray) -> None:
    meanx, stdx, meany, stdy = np.mean(X), np.std(X), np.mean(y), np.std(y)
    self.__standardization = meanx, stdx, meany, stdy # type: ignore
    self.__X = (X - meanx) / stdx
    self.__y = (y - meany) / stdy

  def __accuracy(self, yHat: np.ndarray) -> float:
    rss = np.sum(np.power(self.__y - yHat, 2))
    tss = np.sum(np.power(self.__y - np.mean(self.__y), 2))
    return (1 - rss / tss)

  def fit(self, X: np.ndarray, y: np.ndarray, alpha: float = 0.1, depth: int = 200) -> None:
    self.__check_shapes(X, y)
    self.__data_size = X.shape[0]
    self.__transform_by_standardization(X, y)

    for i in range(1, depth + 1):
      yHat = self.__forward()
      acc = self.__backward(yHat, alpha)

      if not i % int(depth//10):
        acc = self.__accuracy(yHat)
        print(f'epoche {i:3}: accuracy: {acc}')

    self.__save()


# Trainig
def train() -> None:
    try:
      model = LinearRegression()
      model.fit(X, y)
    except Exception as e:
      print(e)


if __name__ == '__main__':
  train()

