import numpy as np
from os.path import isfile
from sys import exit

# Local Varibles
model_params_path = './model/params.npy'

# Basic Checks
if not isfile(model_params_path):
  print('model parameters file is not found, try train the model before')
  exit(1)

# Predictor
def predict() -> None:
  """ Load thetas """
  theta0, theta1 = np.load(model_params_path)
  """Start Prediction"""
  try:
    while True:
      mileage = input('Please, fill a mileage positive number for price prediction: ')
      if not mileage.isnumeric():
        continue

      mileage = int(mileage)
      if mileage < 0:
        continue

      predicted_price = theta0 + theta1 * mileage
      print(f'the Predicted Price of given Mileage({mileage}) is: {predicted_price}')
      break

  except (KeyboardInterrupt, EOFError):
    print('Exiting ...')
  except:
    print('500 Internal Error ...')

if __name__ == '__main__':
  predict()
