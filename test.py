import numpy as np
import matplotlib.pyplot as plt
import itertools
from functools import partial

grid_size = 20
x = np.linspace(-1, 1, num=20)
y = np.linspace(-1, 1, num=20)
orig_pts = [e for e in itertools.product(x, y)]

def transform(pts, t, residual=False):
  a = []
  for (x,y) in pts:
    tx, ty = t(x, y)
    if residual:
      #a.append(((tx*0.5 + x*0.5) , (ty*0.5 + y*0.5) ))
      a.append(((tx + x), (ty + y)))
    else:
      a.append((tx, ty))
  return a

def activation(pts):
  return [(np.max((0, x)), np.max((0, y))) for (x, y) in pts]


def get_mean_and_variance(pts):
  arr = np.array(pts)
  mean = arr.mean(axis=0)
  var = arr.var(axis=0)
  return mean, var


# operations
def batch_norm(mean, var, x, y):
  x, y = translate(-mean[0], -mean[1], x, y)

  scale_x = 1.0 / np.sqrt(var[0])
  scale_y = 1.0 / np.sqrt(var[1])
  x, y = scale(scale_x, scale_y, x, y)
  return x, y


def translate(nx, ny, x, y):
  v_in = np.array([x, y, 1])
  T = np.matrix([[1, 0, nx],
                 [0, 1, ny],
                 [0, 0, 1]])
  v_out = np.array(np.matmul(T,v_in))[0]
  return v_out[0], v_out[1]


def rand(stddev, x, y):
  R = np.matrix(np.random.normal(scale=stddev, size=(2, 2)))
  v_out = np.array(np.matmul(R, np.array([x, y])))[0]
  return v_out[0], v_out[1]


def rot(rot, x, y):
  theta = np.radians(rot)
  R = np.matrix([[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta), np.cos(theta)]])
  v_out = np.array(np.matmul(R, np.array([x,y])))[0]
  return v_out[0], v_out[1]


def scale(tx, ty, x, y):
  return tx * x, ty * y


def spez(x,y):
  x, y = rot(70, x, y)
  x, y = translate(0.5, 0.5, x, y)
  return x, y

def id(x,y):
  return x, y

def softmax(x, y):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(), y


f, axarr = plt.subplots(3, 2, figsize=(10, 14))

all_pts = []

print()
# top left
pts = orig_pts
mean, var = get_mean_and_variance(pts)
print("Mean: ", mean," Var: ", var)
all_pts.append(orig_pts)

pts = transform(pts, partial(rand, 0.01), residual=True)
mean, var = get_mean_and_variance(pts)
print("Mean: ", mean, " Var: ", var)
pts = transform(pts, partial(batch_norm, mean, var), residual=False)
# pts = activation(pts)
all_pts.append(pts)

pts = transform(pts, partial(rand, 0.01), residual=True)
mean, var = get_mean_and_variance(pts)
print("Mean: ", mean, " Var: ", var)
pts = transform(pts, partial(batch_norm, mean, var), residual=False)
# pts = activation(pts)
all_pts.append(pts)

pts = transform(pts, partial(rand, 0.01), residual=False)
mean, var = get_mean_and_variance(pts)
print("Mean: ", mean, " Var: ", var)
pts = transform(pts, partial(batch_norm, mean, var), residual=False)
# pts = activation(pts)
all_pts.append(pts)

pts = transform(pts, partial(rand, 0.01), residual=True)
mean, var = get_mean_and_variance(pts)
print("Mean: ", mean, " Var: ", var)
pts = transform(pts, partial(batch_norm, mean, var), residual=False)
# pts = activation(pts)
all_pts.append(pts)

pts = transform(pts, partial(rand, 0.01), residual=True)
mean, var = get_mean_and_variance(pts)
print("Mean: ", mean, " Var: ", var)
pts = transform(pts, partial(batch_norm, mean, var), residual=False)
# pts = activation(pts)
all_pts.append(pts)

"""
for i in range(5):
  # top right
  pts = transform(pts, partial(rand, 0.01), residual=False)
  mean, var = get_mean_and_variance(pts)
  print("Mean: ", mean," Var: ", var)
  pts = transform(pts, partial(batch_norm, mean, var), residual=False)
  #pts = activation(pts)
  all_pts.append(pts)
"""


"""
# mid left
#pts = transform(pts, partial(rot, 20))
mean, var = get_mean_and_variance(pts)
print("Mean: ", mean," Var: ", var)
pts = transform(pts, partial(batch_norm, mean, var), residual=False)
pts = activation(pts)
all_pts.append(pts)

# mid right
pts = transform(pts, partial(rot, 20))
mean, var = get_mean_and_variance(pts)
print("Mean: ", mean," Var: ", var)
pts = transform(pts, partial(batch_norm, mean, var), residual=False)
pts = activation(pts)
all_pts.append(pts)

# bot left BN
pts = transform(pts, partial(rot, 20))
mean, var = get_mean_and_variance(pts)
print("Mean: ", mean," Var: ", var)
pts = transform(pts, partial(batch_norm, mean, var), residual=False)
pts = activation(pts)
all_pts.append(pts)

# bot right
pts = transform(pts, partial(rot, 20))
mean, var = get_mean_and_variance(pts)
print("Mean: ", mean," Var: ", var)
pts = transform(pts, partial(batch_norm, mean, var), residual=False)
pts = activation(pts)
all_pts.append(pts)
"""

for (ax, pts) in zip(axarr.flatten().tolist(), all_pts):
  ax.scatter(*zip(*pts), marker='o', s=1, color='blue')
  ax.axis([-4, 4, -4, 4])
  ax.grid(True)


plt.show()