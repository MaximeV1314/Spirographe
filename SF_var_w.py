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

mpl.use('Agg')

N_c = 14                                                    # # of circles
Rc = 2                                                    #radius expo
#R = np.array([1, a**(-1), a**(-2), a**(-3), a**(-5), a**(-7), a**(-11), a**(-13), a**(-17), a**(-19), a**(-23), a**(-29), a**(-31), a**(-37)])

R = np.array([Rc * 1/n for n in range(1, N_c+1)])                #radius
N = len(R)
D = np.sum(1.75 * R)

Wc_b = 1
Wc_f = 10                                                    #angular speed expo
pas_Wc_f = 1
Wc = np.arange(Wc_b, Wc_f, pas_Wc_f)

pas_theta = 0.00001
theta = np.arange(0, 2 * np.pi, pas_theta)

pas_img = 1
digit = 4


@jit(nopython = True)
def name(i,digit):

    i = str(i)

    while len(i)<digit:
        i = '0'+i

    i = 'img/'+i+'.png'

    return(i)


@jit(nopython = True)
def circle(W):

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

L_theta = theta.shape[0]
X_t = []
Y_t = []
for i in range(len(Wc)):
    W = np.array([Wc[i]**(n) * (-1)**n for n in range(N_c - 1)])   #angular speed
    X, Y = circle(W)
    X_t.append(np.transpose(X)[-1][1:L_theta])
    Y_t.append(np.transpose(Y)[-1][1:L_theta])
    #print(i/len(Wc))

print("calculation successed")

extension="img/*.png"
for f in glob.glob(extension):
  os.remove(f)

colormap = sns.color_palette("hls", as_cmap=True)
for i in range(0, len(Wc), pas_img):

    fig = plt.figure(figsize = (9, 9))
    ax = plt.gca()

    ax.scatter(X_t[i], Y_t[i], s = 1, c = colormap(np.arange(1, L_theta, 1)/L_theta))
    #ax.plot(X_t[i], Y_t[i], "-", color = "black")

    #ax.plot(X_t[i], Y_t[i], "-", color = colormap(k/L_theta))

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-D, D)
    ax.set_ylim(-D, D)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')

    name_pic = 'img/'+ str(N_c)+"PolyR" + str(Rc) + "_" + str(Wc[i])+'.png'
    plt.savefig(name_pic, bbox_inches='tight', dpi=300, transparent=True)

    ax.clear()
    plt.close(fig)
    print(i/len(Wc))

print("img successed")
# ffmpeg -i img/%05d.png -r 30 -pix_fmt yuv420p hexagon.mp4
# ffmpeg -r 10 -i img/%04d.png -vcodec libx264 -y -an test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" (if 'width not divisible by two' error)