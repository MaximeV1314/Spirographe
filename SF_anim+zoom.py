import random as rd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import glob
import matplotlib.cm as cm
import time
import seaborn as sns

mpl.use('Agg')

N_c = 10                                                    # # of circles
Rc = 2                                                      #radius ratio
Wc = 3                                                      #angular velocity ratio
R = np.array([Rc**(-n) for n in range(N_c)])                #list of radius
W = np.array([Wc**(n) * (-1)**n for n in range(N_c - 1)])   #list of angular velocity

N = len(R)
D = np.sum(R) + 1.5 * R[0]
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

colormap = sns.color_palette("hls", as_cmap=True)
f = theta.shape[0] - 1
for i in range(0, f, pas_img):

    fig = plt.figure(figsize = (9, 9))
    ax = fig.add_subplot(111)

    circle1 = plt.Circle((0, 0), R[0], ec = colormap(0), fc = "none", linewidth=2, zorder = -2)    #cercle
    ax.add_patch(circle1)

    for k in range(1, N):
        circle1 = plt.Circle((X[i][k-1], Y[i][k-1]), R[k], ec = colormap(k/N), fc = "none", linewidth=2)
        ax.add_patch(circle1)

    ax.plot(np.transpose(X)[-1][:i], np.transpose(Y)[-1][:i], "-", color = "black", markersize = 1, zorder = -1)

    #print(X[i], Y[i])

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-D, D)
    ax.set_ylim(-D, D)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')


    ax2 = fig.add_subplot(333)

    circle1 = plt.Circle((0, 0), R[0], ec = colormap(0), fc = "none", linewidth=2, zorder = -2)    #cercle
    ax2.add_patch(circle1)

    for k in range(1, N):
        circle1 = plt.Circle((X[i][k-1], Y[i][k-1]), R[k], ec = colormap(k/N), fc = "none", linewidth=2)
        ax2.add_patch(circle1)

    ax2.plot(np.transpose(X)[-1][:i], np.transpose(Y)[-1][:i], "-", color = "black", markersize = 1, zorder = -1)

    #print(X[i], Y[i])


    #d = D - 9 * i *D /( 10 * theta.shape[0])
    d = D - 15 * D /( 16 )
    d = 1.1 * R[3] + 2 * np.sum(R[4:])
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlim(np.transpose(X)[2][i] - d, np.transpose(X)[2][i] + d)
    ax2.set_ylim(np.transpose(Y)[2][i] - d, np.transpose(Y)[2][i] + d)

    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    #ax2.axis('off')

    name_pic = name(int(i/pas_img), digit)
    plt.savefig(name_pic, bbox_inches='tight', dpi=300)

    ax.clear()
    ax2.clear()
    plt.close(fig)
    print(i/theta.shape[0])

print("img sucessed")

# ffmpeg -i img/%05d.png -r 30 -pix_fmt yuv420p hexagon.mp4
# ffmpeg -r 10 -i img/%04d.png -vcodec libx264 -y -an test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" (if 'width not divisible by two' error)
