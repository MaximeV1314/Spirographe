import random as rd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import glob
import matplotlib.cm as cm
import time
import seaborn as sns
from numba import jit

N_c = 14                                                    # # of circles
#Rc = 3                                                   #radius expo
Wc = 11                                                      #angular speed expo
#R = np.array([Rc**(-n) for n in range(N_c)])                #radius
W = np.array([Wc**(n) * (-1)**n for n in range(N_c - 1)])   #angular speed

a = 3
R = np.array([1, a**(-1), a**(-2), a**(-3), a**(-5), a**(-7), a**(-11), a**(-13), a**(-17), a**(-19), a**(-23), a**(-29), a**(-31), a**(-37)])

N = len(R)
D = np.sum(2 * R)
pas = 0.0001

theta = np.arange(0, 2 * np.pi, pas)

@jit(nopython = True)
def circle():

    X_tabl = [np.zeros((N-1, ))]
    Y_tabl = [np.zeros((N-1, ))]
    for i in range(len(theta)):

        Xi = np.zeros((N-1, ))
        Yi = np.zeros((N-1, ))
        for k in range(1, N):

            Xi[k-1] = Xi[k-2] + (R[k] + R[k-1]) * np.cos(W[k-1] * theta[i])
            Yi[k-1] = Yi[k-2] + (R[k] + R[k-1]) * np.sin(W[k-1] * theta[i])

        X_tabl.append(Xi)
        Y_tabl.append(Yi)

    return X_tabl, Y_tabl

X, Y = circle()

L = theta.shape[0]

fig = plt.figure(figsize = (9, 9))
ax = plt.gca()

colormap = sns.color_palette("hls", as_cmap=True)
plt.scatter(np.transpose(X)[-1][1:], np.transpose(Y)[-1][1:], s = 1, c = colormap(np.arange(0, L, 1)/L), zorder = -1)
#plt.plot(np.transpose(X)[-1][1:], np.transpose(Y)[-1][1:], "-", color = "black")

ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-D, D)
ax.set_ylim(-D, D)

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.axis('off')

name_pic = 'img/'+ str(N_c)+"PR" + str(a) + "_" + str(Wc)+'.png'
plt.savefig(name_pic, bbox_inches='tight', dpi=300, transparent=True)
#plt.show()

ax.clear()
plt.close(fig)

print("img sucessed")