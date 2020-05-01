import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
import numpy as np
import warnings


plt.style.use('fivethirtyeight')


def animate(i):
    data = pd.read_csv('EARvsRxyz.csv')
    frame_number = data['frame_number']
    leftEAR = data['leftEAR']
    rightEAR = data['rightEAR']
    Rx = data['Rx']
    Ry = data['Ry']
    Rz = data['Rz']

    leftEAR = leftEAR.rolling(5).mean()
    rightEAR = rightEAR.rolling(5).mean()
    Rx = Rx.rolling(5).mean()
    leftEAR = leftEAR.dropna()
    rightEAR = rightEAR.dropna()
    Rx = Rx.dropna()
    print(leftEAR)
    #leftEAR1 = gaussian_filter(leftEAR, sigma=1, order=0)
    #leftEAR2 = gaussian_filter(leftEAR, sigma=2, order=0)
    leftEAR3 = gaussian_filter(leftEAR, sigma=3, order=0)
    print(leftEAR)
    print(Rx)
    plt.cla()

    z = np.polyfit(Rx, leftEAR3, 3)
    p = np.poly1d(z)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        p30 = np.poly1d(np.polyfit(Rx, leftEAR3, 30))
    fig = plt.figure(1)
    plt.title("Opened Eyes Looking Ahead\nEARs vs Rx")
    plt.xlabel('Rx (degrees)')
    plt.ylabel('Eye Aspect Ratio (EAR)')
    plt.plot(Rx, leftEAR, linewidth=3, label='Left EAR')
    #plt.plot(Rx, leftEAR1, label='Left EAR Gaussian $\sigma$=1')
    #plt.plot(Rx, leftEAR2, label='Left EAR Gaussian $\sigma$=2')
    plt.plot(Rx, leftEAR3,linewidth=3, label='Left EAR Gaussian $\sigma$=3')
    #plt.plot(Rx, p(Rx),'--', label='polyfit order=3')
    plt.plot(Rx, p30(Rx),'*', label='Fitted Polynomial order=30')
    #plt.plot(Rx, rightEAR, label='Right EAR')
	
    plt.legend(loc='bottom left')
    plt.tight_layout()


#ani = FuncAnimation(plt.gcf(), animate, interval=500) #update twice a second
animate(1) 	

plt.tight_layout()
plt.show()
