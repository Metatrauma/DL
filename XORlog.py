import numpy as np

def unitStep(v):
  if v >= 0:
    return 1
  else:
    return 0

def perceptronModel(x, w, b):
  v = np.dot(w, x) + b
  y = unitStep(v)
  return y

def train_perceptron(X, y, w, b, learning_rate=0.01, epochs=10):
  weights = w
  bias = b
  for epoch in range(2):
    for i in range(len(X)):
      y_pred = perceptronModel(X[i], weights, bias)
      if y_pred == y[i]:
        continue
      else:
        error = y[i] - y_pred
        weights += learning_rate * error * X[i]
        bias += learning_rate * error
  return weights, bias

def NOT_logicFunction(x, wNOT, bNOT):
  return perceptronModel(x, wNOT, bNOT)

def AND_logicFunction(x, w1, bAND):
  return perceptronModel(x, w1, bAND)

def OR_logicFunction(x, w2, bOR):
  return perceptronModel(x, w2, bOR)

def XOR_logicFunction(x):
  wNOT = -1
  bNOT = 0.5
  x1 = np.array([0, 1])
  y1 = np.array([1, 0])
  wNOT, bNOT = train_perceptron(x1, y1, wNOT, bNOT)

  w1 = np.array([1, 1])
  bAND = -1.5
  y2 = np.array([0, 0, 0, 1])
  w1, bAND = train_perceptron(X, y2, w1, bAND)

  w2 = np.array([1, 1])
  bOR = -0.5
  y3 = np.array([0, 1, 1, 1])
  w2, bOR = train_perceptron(X, y3, w2, bOR)

  y11 = AND_logicFunction(x, w1, bAND)
  y12 = OR_logicFunction(x, w2, bOR)
  y13 = NOT_logicFunction(y11, wNOT, bNOT)
  final_x = np.array([y12, y13])
  finalOutput = AND_logicFunction(final_x, w1, bAND)
  return finalOutput

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])

# Write output to a file
with open("xor_output.txt", "w") as file:
  for i in range(len(X)):
    result = XOR_logicFunction(X[i])
    file.write(f"XOR({X[i][0]}, {X[i][1]}) = {result}\n")