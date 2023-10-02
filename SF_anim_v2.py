import random as rd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import glob
import matplotlib.cm as cm
import time

mpl.use('Agg')

N_c = 6                                                    # # of circles
Rc = 2                                                    #radius expo
Wc = 3                                                      #angular speed expo
R = np.array([Rc**(-n) for n in range(N_c)])                #radius
W = np.array([Wc**(n) * (-1)**n for n in range(N_c - 1)])   #angular speed

N = len(R)
D = np.sum(2 * R)
pas = 0.0001
pas_img = 50

theta = np.arange(0, 2 * np.pi, pas)
digit = 4

def name(i,digit):

    i = str(i)

    while len(i)<digit:
        i = '0'+i

    i = 'img/'+i+'.png'

    return(i)

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

extension="img/*.png"
for f in glob.glob(extension):
  os.remove(f)

for i in range(0, theta.shape[0] - 1, pas_img):

    fig = plt.figure(figsize = (9, 9))
    ax = plt.gca()

    circle1 = plt.Circle((0, 0), R[0], ec = "red", fc = "none")    #cercle
    ax.add_patch(circle1)

    for k in range(1, N):
        circle1 = plt.Circle((X[i][k-1], Y[i][k-1]), R[k], ec = "red", fc = "none")
        ax.add_patch(circle1)

    plt.plot(np.transpose(X)[-1][:i], np.transpose(Y)[-1][:i], ".", color = "black", markersize = 1, zorder = -1)

    #print(X[i], Y[i])

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-D, D)
    ax.set_ylim(-D, D)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')

    name_pic = name(int(i/pas_img), digit)
    plt.savefig(name_pic, bbox_inches='tight', dpi=300)

    ax.clear()
    plt.close(fig)
    print(i/theta.shape[0])

print("img sucessed")

# ffmpeg -i img/%05d.png -r 30 -pix_fmt yuv420p hexagon.mp4
# ffmpeg -r 10 -i img/%04d.png -vcodec libx264 -y -an test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" (if 'width not divisible by two' error)
