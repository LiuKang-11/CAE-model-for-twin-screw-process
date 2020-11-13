
##############data treatment
import pandas as pd
import pyodbc
from keras.models import load_model
from keras import backend as K
import time
from Load_Data import data_treatment
import numpy as np
import datetime
import requests
from scipy.optimize import differential_evolution
import time
import copy
import os
from matplotlib import pyplot as plt
import matplotlib
from tools import createFolder as cf
K.clear_session()
import matplotlib.dates as md

c2h2_tar=1.05
p_cons=9.5
p_consl=6.5
cat_h2_UL=45
cat_h2_LL=30
h2_UL=10
h2_LL=0
cat_UL=100
cat_LL=0
ratio=input_std['MHDP_1FIC-201A.PV']/16/input_std['MHDP_1FIC-200A.PV']

def fun(x):

    mv = copy.copy(x_de_test)
    mvv = np.array([x[0], x[1], x[0] + x[2], x[1] + x[3], x[0] + x[2] + x[4], x[1] + x[3] + x[5]])
    mvv = mvv.reshape(1, 3, 2)
    mvv = np.concatenate((mvv[0, 0, :].reshape(1, 1, 2), mvv[0, 0, :].reshape(1, 1, 2), mvv[0, 0, :].reshape(1, 1, 2),
                          mvv[0, 1, :].reshape(1, 1, 2), mvv[0, 1, :].reshape(1, 1, 2), mvv[0, 1, :].reshape(1, 1, 2),
                          mvv[0, 2, :].reshape(1, 1, 2), mvv[0, 2, :].reshape(1, 1, 2), mvv[0, 2, :].reshape(1, 1, 2)),
                         axis=1)
    mv[:, 0:9, [2, 3]] = x_de_test[:, 0:9, [2, 3]] + mvv
    mv[:, 9:18, [2, 3]] = mv[:, 8, [2, 3]]

    sv = model.predict([x_en_test, mv])
    sv = database.inverse_transform_zscore(sv, source='output')
    sv = sv.reshape(18, 4)
    sv = sv + bias

    f1 = 1e4 * np.sum(np.abs(c2h2_tar - sv[0:9, 0]))
    f3 = 1e4 * np.sum(np.abs(c2h2_tar - sv[9:, 0]))
    f2 = 20 * np.sum(x ** 2)
    cat=mv[:,0:9,2]*input_std['MHDP_1FIC-201A.PV']+input_mean['MHDP_1FIC-201A.PV']
    h2=mv[:,0:9,3]*input_std['MHDP_1FIC-2312.PV']+input_mean['MHDP_1FIC-2312.PV']
    g1 = 1e6*np.sum(np.abs(p_cons-sv[0:9,1])*((p_cons-sv[0:9,1])<0).astype(int))
    g11 = 1e6*np.sum(np.abs(sv[0:9,1]-p_consl)*((sv[0:9,1]-p_consl)<0).astype(int))
    g2 = 1e6*np.sum(np.abs(cat_h2_LL-cat/h2)*(cat/h2<cat_h2_LL).astype(int))
    g3 = 1e6*np.sum(np.abs(cat/h2-cat_h2_UL)*(cat/h2>cat_h2_UL).astype(int))
    g21 = 1e16 * np.sum(
        ((mv[0, :, 3] * input_std['MHDP_1FIC-2312.PV'] + input_mean['MHDP_1FIC-2312.PV']) > h2_UL).astype(int))
    g22 = 1e16 * np.sum(
        ((mv[0, :, 3] * input_std['MHDP_1FIC-2312.PV'] + input_mean['MHDP_1FIC-2312.PV']) < h2_LL).astype(int))
    g31 = 1e16 * np.sum(
        ((mv[0, :, 2] * input_std['MHDP_1FIC-201A.PV'] + input_mean['MHDP_1FIC-201A.PV']) > cat_UL).astype(int))
    g32 = 1e16 * np.sum(((mv[0, :, 2] * input_std['MHDP_1FIC-201A.PV'] + input_mean['MHDP_1FIC-201A.PV']) < cat_LL).astype(int))

    return f1+f2+f3+g1+g11+g2+g3+(g21+g22+g31+g32)


x1b=60/input_std['MHDP_1FIC-201A.PV']
x2b=2/input_std['MHDP_1FIC-2312.PV']
bounds=[(-x1b,x1b),(-x2b,x2b),(-x1b,x1b),(-x2b,x2b),(-x1b,x1b),(-x2b,x2b)]
mpc= differential_evolution(fun, bounds, popsize=20, disp=True, maxiter=100,strategy='best1bin')

# 'best1bin''best1exp''rand1exp''randtobest1exp''best2exp''rand2exp''randtobest1bin''best2bin''rand2bin''rand1bin'
print(mpc)
dx=(mpc.x*np.array([input_std['MHDP_1FIC-201A.PV'],input_std['MHDP_1FIC-2312.PV'],input_std['MHDP_1FIC-201A.PV'],input_std['MHDP_1FIC-2312.PV'],input_std['MHDP_1FIC-201A.PV'],input_std['MHDP_1FIC-2312.PV']])).reshape(3,2)
print(dx)


mv=copy.copy(x_de_test)
x=copy.copy(mpc.x)




#
