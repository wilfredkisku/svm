import numpy as np

x = np.arrange(1, 200, 20)

N = x.size()
a,b = np.meshgrid(x,x)

it = np.array(b.ravel(),a.ravel(),np.ones(N)])
