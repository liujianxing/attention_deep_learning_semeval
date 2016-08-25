#!/usr/local/bin/python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.interpolate import spline
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt

"""
"""

fp = open("GRU.pickle",'rb')
y_gru = np.array(cPickle.load(fp))

fp = open("LSTM.pickle",'rb')
y_lstm = np.array(cPickle.load(fp))

fp = open("BiRNN.pickle",'rb')
y_birnn = np.array(cPickle.load(fp))

x = np.array([i for i in range(len(y_gru))])

x_smooth = np.linspace(x.min(),x.max(),300)
y_gru_smooth = spline(x, y_gru, x_smooth)
y_lstm_smooth = spline(x, y_lstm, x_smooth)
y_birnn_smooth = spline(x, y_birnn, x_smooth)

plt.figure()
plt.plot(x_smooth,y_gru_smooth,':')
plt.plot(x_smooth,y_lstm_smooth,'--')
plt.plot(x_smooth,y_birnn_smooth,'-')
plt.grid(alpha=0.5)
plt.legend(['GRU', 'LSTM', 'BiRNN'], loc='upper right')
plt.show()

fp = open("BiRNN_slot3.pickle",'rb')
y_birnn3= np.array(cPickle.load(fp))

x = np.array([i for i in range(len(y_birnn3))])
x_smooth = np.linspace(x.min(),x.max(),20)
y_birnn3_smooth = spline(x, y_birnn3, x_smooth)

plt.figure()
plt.plot(x_smooth,y_birnn3_smooth,'-')
plt.grid(alpha=0.5)
plt.legend(['BiRNN'], loc='upper right')
plt.show()

