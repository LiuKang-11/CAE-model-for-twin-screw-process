import pandas as pd
import pyodbc
from keras.models import load_model
from keras import backend as K
import time
from Load_Data import data_treatment

K.clear_session()
import numpy as np
import datetime
import requests
from scipy.optimize import differential_evolution
import time
import copy
import os
import matplotlib.pyplot as plt
from tools import createFolder as cf
import matplotlib.dates as md
import matplotlib

# from requests.auth import HTTPBasicAuth

####parameter change
run = 5  # model_times
all_model='20200620_S2S_rolling_Ims240_Om_and_s180_dtp0'#model_name
data_time='20200620'
now_time = "2020-10-23 08:55:00"
for g in range(20):
    timeArray = time.strptime(now_time, "%Y-%m-%d %H:%M:%S")
    tend = time.strftime("%Y-%m-%d %H:%M:%S",timeArray)

    ##############data treatment

    en_window_mins = 240
    de_window_mins = 180
    H=int(de_window_mins/10)
    prefix = all_model
    target=all_model+'_instant_15mins_20201023_package'
    input_file = ['mpc_inputs%s.csv'%(g)]
    output_file = ['mpc_outputs%s.csv'%(g)]
    en_mv_and_sv = ['MHDP_C-H2C2-231.PV', 'MHDP_1PI-2311.PV', 'MHDP_1TI-2311.PV', 'MHDP_1TI-2313.PV',
                    'MHDP_1FI-2015.PV', 'MHDP_1FIC-2311.PV', 'MHDP_1FIC-201A.PV', 'MHDP_1FIC-2312.PV',
                    'MHDP_1FIC-200A.PV', 'c4h8', 'c6h14_total', 'MHDP_1TIC-231H.PV', 'MHDP_1TIC-231I.PV',
                    'MHDP_1FIC-231A.PV']
    de_mv = ['MHDP_C-H2C2-231.PV', 'MHDP_1PI-2311.PV', 'MHDP_1TI-2311.PV', 'MHDP_1TI-2311.PV',
             'MHDP_1FI-2015.PV', 'MHDP_1FIC-2311.PV', 'MHDP_1FIC-201A.PV', 'MHDP_1FIC-2312.PV',
             'MHDP_1FIC-200A.PV', 'c4h8', 'c6h14_total', 'MHDP_1TIC-231H.PV', 'MHDP_1TIC-231I.PV', 'MHDP_1FIC-231A.PV']

    y_sv = ['MHDP_C-H2C2-231.PV', 'MHDP_1PI-2311.PV', 'MHDP_1TI-2311.PV', 'MHDP_1TI-2313.PV']

    sampling_interval_in_min = 10
    pred_qv_next_min = 0

    database = data_treatment(input_file=input_file, output_file=output_file)
    database.preprocessing_data_drop_na(source='input')
    database.preprocessing_data_drop_na(source='output')
    input_mean = pd.read_csv('./data/input_mean_' + data_time + '.csv', squeeze=True, index_col=0, header=None)
    input_std = pd.read_csv('./data/input_std_' + data_time + '.csv', squeeze=True, index_col=0, header=None)
    output_mean = pd.read_csv('./data/output_mean_' + data_time + '.csv', squeeze=True, index_col=0, header=None)
    output_std = pd.read_csv('./data/output_std_' + data_time + '.csv', squeeze=True, index_col=0, header=None)
    database.set_tag_mean_std(x_mean=[input_mean], x_std=[input_std], y_mean=[output_mean], y_std=[output_std])

    database.transform_zscore(source='input')
    database.transform_zscore(source='output')
    # S2S data form
    dataset = database.get_data_time_series_S2S_zscore(en_input_tag=en_mv_and_sv, de_input_tag=de_mv,
                                                       en_input_in_mins=en_window_mins, de_input_in_mins=de_window_mins,
                                                       de_y_different_in_mins=pred_qv_next_min,
                                                       sampling_interval_in_min=sampling_interval_in_min)

    for i in range(1):
        x_en, x_de, y, date = dataset[0][0][0], dataset[0][0][1], dataset[0][0][2], dataset[0][0][3]

    ##############prediction

    x_de = x_de[:, :, -10:]

    x_de_test = copy.copy(x_de[1, :, :]).reshape(1, 18, 10)
    x_en_test = copy.copy(x_en[1, :, :]).reshape(1, 24, 14)

    model = load_model('./h5/%s/S2S' % (prefix) + '_run_' + str(run) + '.h5')
    # make a test prediction
    y_pred_test = model.predict([x_en, x_de])
    test_y_cov = database.inverse_transform_zscore(y, source='output')
    y_pred_test_cov = database.inverse_transform_zscore(y_pred_test, source='output')

    K.clear_session()
    bias = test_y_cov[1, 0, :] - y_pred_test_cov[0, 0, :]

    # %%



    # %% AIMPC
    c2h2_tar = 1.01

    p_cons = 10
    p_consl = 7
    cat_h2_UL = 100
    cat_h2_LL = 30
    h2_UL = 10
    h2_LL = 1
    cat_UL = 100
    cat_LL = 0

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
        f2 = 10 * np.sum(x ** 2)
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

        return f1+f2+f3+g1+g11+g3+g2+(g21+g22+g31+g32)


    x1b=100/input_std['MHDP_1FIC-201A.PV']
    x2b=4/input_std['MHDP_1FIC-2312.PV']
    bounds=[(-x1b,x1b),(-x2b,x2b),(-x1b,x1b),(-x2b,x2b),(-x1b,x1b),(-x2b,x2b)]
    mpc= differential_evolution(fun, bounds, popsize=20, disp=True, maxiter=100,strategy='best1bin')

    # 'best1bin''best1exp''rand1exp''randtobest1exp''best2exp''rand2exp''randtobest1bin''best2bin''rand2bin''rand1bin'
    print(mpc)
    dx=(mpc.x*np.array([input_std['MHDP_1FIC-201A.PV'],input_std['MHDP_1FIC-2312.PV'],input_std['MHDP_1FIC-201A.PV'],input_std['MHDP_1FIC-2312.PV'],input_std['MHDP_1FIC-201A.PV'],input_std['MHDP_1FIC-2312.PV']])).reshape(3,2)
    print(dx)


    mv=copy.copy(x_de_test)
    x=copy.copy(mpc.x)

    mvv=np.array([x[0],x[1],x[0]+x[2],x[1]+x[3],x[0]+x[2]+x[4],x[1]+x[3]+x[5]])
    mvv=mvv.reshape(1,3,2)
    mvv=np.concatenate((mvv[0,0,:].reshape(1,1,2),mvv[0,0,:].reshape(1,1,2),mvv[0,0,:].reshape(1,1,2),mvv[0,1,:].reshape(1,1,2),mvv[0,1,:].reshape(1,1,2),mvv[0,1,:].reshape(1,1,2),mvv[0,2,:].reshape(1,1,2),mvv[0,2,:].reshape(1,1,2),mvv[0,2,:].reshape(1,1,2)),axis=1)
    mv[:,0:9,[2, 3]]=x_de_test[:,0:9,[2, 3]]+mvv
    mv[:,9:18,[2, 3]]=mv[:,8,[2, 3]]
    svvo = model.predict([x_en_test, mv])
    svv=database.inverse_transform_zscore(svvo, source='output')
    svv=svv.reshape(18,4)
    svv=svv+bias
    print(mv[:,0:9,[2, 3]]*np.array([input_std['MHDP_1FIC-201A.PV'],input_std['MHDP_1FIC-2312.PV']]).reshape(1,1,2)+np.array([input_mean['MHDP_1FIC-201A.PV'],input_mean['MHDP_1FIC-2312.PV']]).reshape(1,1,2))
    print(svv)

    np.savetxt('./data/mpc_mv_run'+str(run)+'.csv',mv.reshape(18,10), delimiter=',')
    np.savetxt('./data/mpc_sv_run'+str(run)+'.csv',svv, delimiter=',')

    mpc_mv_pre = copy.copy(mv[0, :, :])
    mpc_sv_pre = copy.copy(svv[:, :])
    mpc_mv = copy.copy(mv[0, [0], :])
    mpc_sv = copy.copy(svv[[0], :])
    dv = ['MHDP_1FI-2015.PV', 'MHDP_1FIC-2311.PV', 'MHDP_1FIC-201A.PV', 'MHDP_1FIC-2312.PV', 'MHDP_1FIC-200A.PV',
          'c4h8', 'c6h14_total', 'MHDP_1TIC-231H.PV',
          'MHDP_1TIC-231I.PV', 'MHDP_1FIC-231A.PV']

    # mpc_mv=np.append(mpc_mv,mv[0,[0,1,2],:],axis=0)
    # mpc_sv=np.append(mpc_sv,svv[[0,1,2],:],axis=0)
    date_instant = pd.read_csv('./data/mpc_inputs%s.csv' % (g))
    # date_instant["DateTime"] = pd.to_datetime(date_instant["DateTime"])
    date_instant = pd.DataFrame(date_instant.iloc[25:-1, 0])

    mpc_mv_2 = pd.DataFrame(mpc_mv_pre, columns=dv, index=date_instant.iloc[:])
    mpc_sv_2 = pd.DataFrame(mpc_sv_pre, columns=y_sv, index=date_instant.iloc[:])
    mpc_mv_1 = pd.DataFrame(mpc_mv, columns=dv, index=date_instant.iloc[0])
    mpc_sv_1 = pd.DataFrame(mpc_sv, columns=y_sv, index=date_instant.iloc[0])
    for i in dv:
        mpc_mv_2.loc[:, i] = mpc_mv_2.loc[:, i] * input_std[i] + input_mean[i]
        mpc_mv_1.loc[:, i] = mpc_mv_1.loc[:, i] * input_std[i] + input_mean[i]

    cf('./fig/%s' % (target))

    if g == 0:
        mpc_mv_1.to_csv('./fig/%s/mpc_mv_simulation_run' % (target) + str(run) + '.csv')
        mpc_sv_1.to_csv('./fig/%s/mpc_sv_simulation_run' % (target) + str(run) + '.csv')
        mpc_mv_2.to_csv('./fig/%s/mpc_mv_run' % (target) + str(run) + '.csv')
        mpc_sv_2.to_csv('./fig/%s/mpc_sv_run' % (target) + str(run) + '.csv')
    else:
        mpc_mv_1.to_csv('./fig/%s/mpc_mv_simulation_run' % (target) + str(run) + '.csv', mode="a", header=False)
        mpc_sv_1.to_csv('./fig/%s/mpc_sv_simulation_run' % (target) + str(run) + '.csv', mode="a", header=False)
        mpc_mv_2.to_csv('./fig/%s/mpc_mv_run' % (target) + str(run) + '.csv', mode="a", header=False)
        mpc_sv_2.to_csv('./fig/%s/mpc_sv_run' % (target) + str(run) + '.csv', mode="a", header=False)

    # np.savetxt('./data/mpc_mv_run'+str(run)+'.csv',mpc_mv, delimiter=',')
    # np.savetxt('./data/mpc_sv_run'+str(run)+'.csv',mpc_sv, delimiter=',')
    ####H2C2
    dv = ['MHDP_1FI-2015.PV', 'MHDP_1FIC-2311.PV', 'MHDP_1FIC-201A.PV', 'MHDP_1FIC-2312.PV', 'MHDP_1FIC-200A.PV',
          'c4h8', 'c6h14_total', 'MHDP_1TIC-231H.PV',
          'MHDP_1TIC-231I.PV', 'MHDP_1FIC-231A.PV']
    # mpc_mv=np.append(mpc_mv,mv[0,[0,1,2],:],axis=0)
    # mpc_sv=np.append(mpc_sv,svv[[0,1,2],:],axis=0)
    cf('./fig/%s' % (target))

    # np.savetxt('./data/mpc_mv_run'+str(run)+'.csv',mpc_mv, delimiter=',')
    # np.savetxt('./data/mpc_sv_run'+str(run)+'.csv',mpc_sv, delimiter=',')
    ####H2C2

    ### 15mins
    ####H2C2
    data = pd.read_csv('./data/DATA_231_20190709_all_15min.csv')
    mpc_m = pd.read_csv('./fig/%s/mpc_mv_simulation_run' % (target) + str(run) + '.csv')
    # mpc_m = pd.DataFrame(mpc_m.values,
    #                      columns=['Series', 'MHDP_1FI-2015.PV', 'MHDP_1FIC-2311.PV', 'MHDP_1FIC-201A.PV',
    #                               'MHDP_1FIC-2312.PV', 'MHDP_1FIC-200A.PV', 'c4h8', 'c6h14_total',
    #                               'MHDP_1TIC-231H.PV',
    #                               'MHDP_1TIC-231I.PV', 'MHDP_1FIC-231A.PV'])
    # for i in dv:
    #     mpc_m.loc[:, i] = mpc_m.loc[:, i] * input_std[i] + input_mean[i]
    mpc_v = pd.read_csv('./fig/%s/mpc_sv_simulation_run' % (target) + str(run) + '.csv')
    mpc_v = pd.DataFrame(mpc_v.values,
                         columns=['Series', 'MHDP_C-H2C2-231.PV', 'MHDP_1PI-2311.PV', 'MHDP_1TI-2311.PV',
                                  'MHDP_1TI-2313.PV'])
    real_mv_short = data.loc[0:1,
                    ['MHDP_1FI-2015.PV', 'MHDP_1FIC-2311.PV', 'MHDP_1FIC-201A.PV', 'MHDP_1FIC-2312.PV',
                     'MHDP_1FIC-200A.PV', 'c4h8', 'c6h14_total', 'MHDP_1TIC-231H.PV',
                     'MHDP_1TIC-231I.PV', 'MHDP_1FIC-231A.PV']]
    real_sv_short = data.loc[0:1,
                    ['MHDP_C-H2C2-231.PV', 'MHDP_1PI-2311.PV', 'MHDP_1TI-2311.PV', 'MHDP_1TI-2313.PV']]
    AIMPC_value = data.loc[:, ['MHDP_1FIC-2312 SUG.PV', 'MHDP_1FIC-201A SUG.PV']]
    mpc_mv1 = pd.concat([real_mv_short, mpc_m])
    mpc_sv1 = pd.concat([real_sv_short, mpc_v])
    # tags_range_low = 0.6
    # tags_range_up = 1.2
    # plt.ylim(tags_range_low, tags_range_up)
    date_R = data.iloc[2:, 0]
    date_R_1 = []
    for i in range(int(len(date_R) / 2)):
        date_R_1.append(date_R.iloc[2 * i])
    date_R_1 = pd.DataFrame(date_R_1)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    date = md.date2num(data.iloc[:, 0])

    # tags_range_low = 6.5
    # tags_range_up = 9.5
    # plt.ylim(tags_range_low, tags_range_up)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    date = md.date2num(data.iloc[:, 0])

    ####201A

    R_m = pd.read_csv('./data/mv.csv')
    R_v = pd.read_csv('./data/sv.csv')
    date_R = data.iloc[2:, 0]
    date_R_1 = []
    for i in range(int(len(date_R) / 2)):
        date_R_1.append(date_R.iloc[2 * i])
    date_R_1 = pd.DataFrame(date_R_1)
    R_m_h2 = []
    R_m_cat = []
    R_v_h2C = []
    R_v_p = []
    for x in range(0, int(len(R_m) / 9)):
        R_m_h2.append(R_m.iloc[9 * x, 2])
        R_m_cat.append(R_m.iloc[9 * x, 1])
    for x in range(0, int(len(R_v) / 18)):
        R_v_h2C.append(R_v.iloc[18 * x, 1])
        R_v_p.append(R_v.iloc[18 * x, 2])

    ### 15mins
    dv = ['MHDP_1FI-2015.PV', 'MHDP_1FIC-2311.PV', 'MHDP_1FIC-201A.PV', 'MHDP_1FIC-2312.PV', 'MHDP_1FIC-200A.PV',
          'c4h8', 'c6h14_total', 'MHDP_1TIC-231H.PV',
          'MHDP_1TIC-231I.PV', 'MHDP_1FIC-231A.PV']
    data = pd.read_csv('./data/DATA_231_20190709_all_15min.csv')
    mpc_m = pd.read_csv('./fig/%s/mpc_mv_simulation_run' % (target) + str(run) + '.csv')
    mpc_m = pd.DataFrame(mpc_m.values,
                         columns=['Series', 'MHDP_1FI-2015.PV', 'MHDP_1FIC-2311.PV', 'MHDP_1FIC-201A.PV',
                                  'MHDP_1FIC-2312.PV', 'MHDP_1FIC-200A.PV', 'c4h8', 'c6h14_total',
                                  'MHDP_1TIC-231H.PV',
                                  'MHDP_1TIC-231I.PV', 'MHDP_1FIC-231A.PV'])
    # for i in dv:
    #     mpc_m.loc[:, i] = mpc_m.loc[:, i] * input_std[i] + input_mean[i]
    mpc_v = pd.read_csv('./fig/%s/mpc_sv_simulation_run' % (target) + str(run) + '.csv')
    mpc_v = pd.DataFrame(mpc_v.values,
                         columns=['Series', 'MHDP_C-H2C2-231.PV', 'MHDP_1PI-2311.PV', 'MHDP_1TI-2311.PV',
                                  'MHDP_1TI-2313.PV'])
    real_mv_short = data.loc[0:1,
                    ['MHDP_1FI-2015.PV', 'MHDP_1FIC-2311.PV', 'MHDP_1FIC-201A.PV', 'MHDP_1FIC-2312.PV',
                     'MHDP_1FIC-200A.PV', 'c4h8', 'c6h14_total', 'MHDP_1TIC-231H.PV',
                     'MHDP_1TIC-231I.PV', 'MHDP_1FIC-231A.PV']]
    real_sv_short = data.loc[0:1,
                    ['MHDP_C-H2C2-231.PV', 'MHDP_1PI-2311.PV', 'MHDP_1TI-2311.PV', 'MHDP_1TI-2313.PV']]
    AIMPC_value = data.loc[:, ['MHDP_1FIC-2312 SUG.PV', 'MHDP_1FIC-201A SUG.PV']]
    mpc_mv1 = pd.concat([real_mv_short, mpc_m])
    mpc_sv1 = pd.concat([real_sv_short, mpc_v])
    plt.figure(figsize=(16.4, 6.6))
    matplotlib.rcParams.update({'font.size': 23})
    # tags_range_low = 0.6
    # tags_range_up = 1.2
    # plt.ylim(tags_range_low, tags_range_up)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    date = md.date2num(data.iloc[:, 0])
    plt.plot_date(data.iloc[:, 0], data.loc[:, 'MHDP_C-H2C2-231.PV'], 'b-', label='Real data', linewidth=4,
                  linestyle='--')
    plt.plot_date(data.iloc[:len(mpc_sv1), 0], mpc_sv1.iloc[:, 0], 'r-', linewidth=4, label='AIMPC-Setup3',
                  linestyle='-', marker='o', markersize=10)
    # plt.plot_date(date_R_1.iloc[:len(R_v_h2C), 0], R_v_h2C, 'k-', marker='^', linewidth=4, markersize=10,
    #               label='AIMPC-Doctor')

    matplotlib.rcParams.update({'font.size': 25})
    plt.title('%s' % ('MHDP_C-H2C2-231.PV'))
    plt.legend(loc=4)
    plt.xticks(fontsize=21.3, rotation=20)
    plt.axhline(c2h2_tar, linewidth=3, color='g')
    plt.grid()
    cf('./fig/%s/%s' % (target, 'mpc_'))
    plt.savefig('./fig/%s/%s/%s_mpc_run_' % (target, 'mpc_', 'MHDP_C-H2C2-231.PV') + str(run) + '.png')
    ####pressure
    plt.figure(figsize=(16.4, 6.6))
    matplotlib.rcParams.update({'font.size': 23})
    # tags_range_low = 6.5
    # tags_range_up = 9.5
    # plt.ylim(tags_range_low, tags_range_up)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    date = md.date2num(data.iloc[:, 0])
    plt.plot_date(data.iloc[:, 0], data.loc[:, 'MHDP_1PI-2311.PV'], 'b-', label='Real data', linewidth=4,
                  linestyle='--')
    plt.plot_date(data.iloc[:len(mpc_sv1), 0], mpc_sv1.iloc[:, 1], 'r-', linewidth=4, label='AIMPC-Setup3',
                  linestyle='-', marker='o', markersize=10)
    # plt.plot_date(date_R_1.iloc[:len(R_v_p), 0], R_v_p, 'k-', marker='^', linewidth=4, markersize=10,
    #               label='AIMPC-Doctor')

    matplotlib.rcParams.update({'font.size': 25})
    plt.title('%s' % ('MHDP_1PI-2311.PV'))
    plt.legend(loc=4)
    plt.grid()
    plt.xticks(fontsize=21.3, rotation=20)
    cf('./fig/%s/%s' % (target, 'mpc_'))
    plt.savefig('./fig/%s/%s/%s_mpc_run_' % (target, 'mpc_', 'MHDP_1PI-2311.PV') + str(run) + '.png')
    ####201A
    plt.figure(figsize=(16.4, 6.6))
    matplotlib.rcParams.update({'font.size': 23})
    # tags_range_low = cat_LL
    # # tags_range_up = cat_UL
    # plt.ylim(tags_range_low, tags_range_up)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    date = md.date2num(data.iloc[:, 0])
    plt.plot_date(data.iloc[:, 0], data.loc[:, 'MHDP_1FIC-201A.PV'], 'b-', label='Real data', linewidth=4,
                  linestyle='--')

    plt.plot_date(data.iloc[:, 0], AIMPC_value.iloc[:, 1], 'g-', linewidth=4, label='AIMPC-Real', linestyle='--')
    plt.plot_date(data.iloc[:len(mpc_mv1), 0], mpc_mv1.loc[:, 'MHDP_1FIC-201A.PV'], 'r-', linewidth=4,
                  label='AIMPC-Setup3', linestyle='-', marker='o', markersize=10)
    # plt.plot_date(date_R_1.iloc[:len(R_m_cat), 0], R_m_cat, 'k-', marker='^', linewidth=4, markersize=10,
    #               label='AIMPC-Doctor')
    matplotlib.rcParams.update({'font.size': 25})
    plt.title('%s' % ('MHDP_1FIC-201A.PV'))
    plt.legend(loc=1)
    plt.grid()
    plt.xticks(fontsize=21.3, rotation=20)
    cf('./fig/%s/%s' % (target, 'mpc_'))
    plt.savefig('./fig/%s/%s/%s_mpc_run_' % (target, 'mpc_', 'MHDP_1FIC-201A.PV') + str(run) + '.png')
    ####FIC_2312
    plt.figure(figsize=(16.4, 6.6))
    matplotlib.rcParams.update({'font.size': 23})
    # tags_range_low = h2_LL
    # tags_range_up = h2_UL
    # plt.ylim(tags_range_low, tags_range_up)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    date = md.date2num(data.iloc[:, 0])
    plt.plot_date(data.iloc[:, 0], data.loc[:, 'MHDP_1FIC-2312.PV'], 'b-', label='Real data', linewidth=3,
                  linestyle='--')
    plt.plot_date(data.iloc[:, 0], AIMPC_value.iloc[:, 0], 'g-', linewidth=4, label='AIMPC-Real', linestyle='-')
    plt.plot_date(data.iloc[:len(mpc_mv1), 0], mpc_mv1.loc[:, 'MHDP_1FIC-2312.PV'], 'r-', linewidth=3,
                  label='AIMPC-Setup3', linestyle='-', marker='o', markersize=10)
    # plt.plot_date(date_R_1.iloc[:len(R_m_h2), 0], R_m_h2, 'k-', marker='^', linewidth=4, markersize=10,
    #               label='AIMPC-Doctor')
    matplotlib.rcParams.update({'font.size': 25})
    plt.title('%s' % ('MHDP_1FIC-2312.PV'))
    plt.legend(loc=1)
    plt.grid()
    plt.xticks(fontsize=21.3, rotation=20)
    cf('./fig/%s/%s' % (target, 'mpc_'))
    plt.savefig('./fig/%s/%s/%s_mpc_run_' % (target, 'mpc_', 'MHDP_1FIC-2312.PV') + str(run) + '.png')
    plt.close('all')

    R_m_h2 = []
    R_m_cat = []
    R_v_h2C = []
    R_v_p = []
    for x in range(0, int(len(R_m) / 3)):
        R_m_h2.append(R_m.iloc[3 * x, 2])
        R_m_cat.append(R_m.iloc[3 * x, 1])
    for x in range(0, int(len(R_v) / 3)):
        R_v_h2C.append(R_v.iloc[3 * x, 1])
        R_v_p.append(R_v.iloc[3 * x, 2])
    #
    #
    #

    ####60mins
    data = pd.read_csv('./data/DATA_231_20190709_all_15min.csv')
    mpc_m = pd.read_csv('./fig/%s/mpc_mv_run%s.csv' % (target, str(run)))
    mpc_m = pd.DataFrame(mpc_m.values,
                         columns=['Series', 'MHDP_1FI-2015.PV', 'MHDP_1FIC-2311.PV', 'MHDP_1FIC-201A.PV',
                                  'MHDP_1FIC-2312.PV', 'MHDP_1FIC-200A.PV', 'c4h8', 'c6h14_total',
                                  'MHDP_1TIC-231H.PV',
                                  'MHDP_1TIC-231I.PV', 'MHDP_1FIC-231A.PV'])
    # for i in dv:
    #     mpc_m.loc[:, i] = mpc_m.loc[:, i] * input_std[i] + input_mean[i]
    mpc_v = pd.read_csv('./fig/%s/mpc_sv_run%s.csv' % (target, str(run)))
    mpc_v = pd.DataFrame(mpc_v.values,
                         columns=['Series', 'MHDP_C-H2C2-231.PV', 'MHDP_1PI-2311.PV', 'MHDP_1TI-2311.PV',
                                  'MHDP_1TI-2313.PV'])
    real_mv_short = data.loc[0:1,
                    ['MHDP_1FI-2015.PV', 'MHDP_1FIC-2311.PV', 'MHDP_1FIC-201A.PV', 'MHDP_1FIC-2312.PV',
                     'MHDP_1FIC-200A.PV', 'c4h8', 'c6h14_total', 'MHDP_1TIC-231H.PV',
                     'MHDP_1TIC-231I.PV', 'MHDP_1FIC-231A.PV']]
    real_sv_short = data.loc[0:1,
                    ['MHDP_C-H2C2-231.PV', 'MHDP_1PI-2311.PV', 'MHDP_1TI-2311.PV', 'MHDP_1TI-2313.PV']]
    AIMPC_value = data.loc[:, ['MHDP_1FIC-2312 SUG.PV', 'MHDP_1FIC-201A SUG.PV']]

    mpc_mv1 = pd.concat([real_mv_short, mpc_m])
    mpc_sv1 = pd.concat([real_sv_short, mpc_v])
    plt.figure(figsize=(16.4, 6.6))
    matplotlib.rcParams.update({'font.size': 23})
    # tags_range_low = 0.6
    # tags_range_up = 1.2
    # plt.ylim(tags_range_low, tags_range_up)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    date = md.date2num(data.iloc[:, 0])
    plt.plot_date(data.iloc[:, 0], data.loc[:, 'MHDP_C-H2C2-231.PV'], 'b-', label='Real data', linewidth=4,
                  linestyle='--')
    plt.plot_date(data.iloc[2, 0], mpc_sv1.iloc[2, 0], 'r-', linewidth=4, marker='o', markersize=10)
    instant_data = pd.read_csv('./data/mpc_inputs0.csv')
    instant_data['DateTime'] = pd.to_datetime(instant_data['DateTime'])
    plt.plot_date(instant_data.iloc[25:31, 0], mpc_v.iloc[:6, 1], 'r-', linewidth=4, label='AIMPC-Setup3',
                  linestyle='-')
    for g in range(1, int(len(mpc_v) / 18 / 4)):
        print(g * 4)
        instant_data = pd.read_csv('./data/mpc_inputs%s.csv' % (g * 4))
        instant_data['DateTime'] = pd.to_datetime(instant_data['DateTime'])
        plt.plot_date(instant_data.iloc[25, 0], mpc_v.iloc[4 * g * 18, 1], 'r-', marker='o', markersize=10)
        plt.plot_date(instant_data.iloc[25:31, 0], mpc_v.iloc[4 * g * 18:4 * g * 18 + 6, 1], 'r-', linewidth=4)
    # plt.plot_date(date_R_1.iloc[0, 0], R_v.iloc[0, 1], 'k-', marker='^', linewidth=4, markersize=10)
    # plt.plot_date(date_R_1.iloc[:3, 0], R_v_h2C[:3], 'k-', linewidth=4, markersize=10,
    #               label='AIMPC-Doctor')
    # for z in range(1, int(len(R_v_h2C) / 6 / 2 + 1)):
    #     plt.plot_date(date_R_1.iloc[2 * z, 0], R_v_h2C[2 * 6 * z], 'k-', marker='^', markersize=10)
    #     plt.plot_date(date_R_1.iloc[2 * z:2 * z + 3, 0], R_v_h2C[2 * 6 * z:2 * 6 * z + 3], 'k-', linewidth=4,
    #                   markersize=10)
    matplotlib.rcParams.update({'font.size': 25})
    plt.title('%s' % ('MHDP_C-H2C2-231.PV'))
    plt.legend(loc=4)
    plt.xticks(fontsize=21.3, rotation=20)
    plt.axhline(c2h2_tar, linewidth=3, color='g')
    plt.grid()
    cf('./fig/%s/%s' % (target, 'mpc_'))
    plt.savefig('./fig/%s/%s/%s_mpc_60mins_run_' % (target, 'mpc_', 'MHDP_C-H2C2-231.PV') + str(run) + '.png')
    ####pressure

    plt.figure(figsize=(16.4, 6.6))
    matplotlib.rcParams.update({'font.size': 23})
    # tags_range_low = 6.5
    # tags_range_up = 9.5
    # plt.ylim(tags_range_low, tags_range_up)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    date = md.date2num(data.iloc[:, 0])
    plt.plot_date(data.iloc[:, 0], data.loc[:, 'MHDP_1PI-2311.PV'], 'b-', label='Real data', linewidth=4,
                  linestyle='--')

    plt.plot_date(data.iloc[2, 0], mpc_sv1.iloc[2, 1], 'r-', linewidth=4, marker='o', markersize=10)
    instant_data = pd.read_csv('./data/mpc_inputs0.csv')
    instant_data['DateTime'] = pd.to_datetime(instant_data['DateTime'])
    plt.plot_date(instant_data.iloc[25:31, 0], mpc_v.iloc[:6, 2], 'r-', linewidth=4, label='AIMPC-Setup3',
                  linestyle='-')
    for g in range(1, int(len(mpc_v) / 18 / 4)):
        instant_data = pd.read_csv('./data/mpc_inputs%s.csv' % (g * 4))
        instant_data['DateTime'] = pd.to_datetime(instant_data['DateTime'])
        plt.plot_date(instant_data.iloc[25, 0], mpc_v.iloc[4 * g * 18, 2], 'r-', marker='o', markersize=10)
        plt.plot_date(instant_data.iloc[25:31, 0], mpc_v.iloc[4 * g * 18:4 * g * 18 + 6, 2], 'r-', linewidth=4)
    # plt.plot_date(date_R_1.iloc[0, 0], R_v.iloc[0, 2], 'k-', marker='^', linewidth=4, markersize=10)
    # plt.plot_date(date_R_1.iloc[:3, 0], R_v_p[:3], 'k-', linewidth=4, markersize=10,
    #               label='AIMPC-Doctor')
    # for z in range(1, int(len(R_v_h2C) / 6 / 2 + 1)):
    #     plt.plot_date(date_R_1.iloc[2 * z, 0], R_v_p[2 * 6 * z], 'k-', marker='^', markersize=10)
    #     plt.plot_date(date_R_1.iloc[2 * z:2 * z + 3, 0], R_v_p[2 * 6 * z:2 * 6 * z + 3], 'k-', linewidth=4,
    #                   markersize=10)
    matplotlib.rcParams.update({'font.size': 25})
    plt.title('%s' % ('MHDP_1PI-2311.PV'))
    plt.legend(loc=4)
    plt.grid()
    plt.xticks(fontsize=21.3, rotation=20)
    cf('./fig/%s/%s' % (target, 'mpc_'))
    plt.savefig('./fig/%s/%s/%s_mpc_60mins_run_' % (target, 'mpc_', 'MHDP_1PI-2311.PV') + str(run) + '.png')
    ####201A
    plt.figure(figsize=(16.4, 6.6))
    matplotlib.rcParams.update({'font.size': 23})
    # tags_range_low = cat_LL
    # # tags_range_up = cat_UL
    # plt.ylim(tags_range_low, tags_range_up)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    date = md.date2num(data.iloc[:, 0])
    plt.plot_date(data.iloc[:, 0], data.loc[:, 'MHDP_1FIC-201A.PV'], 'b-', label='Real data', linewidth=4,
                  linestyle='--')
    plt.plot_date(data.iloc[:, 0], AIMPC_value.iloc[:, 1], 'g-', linewidth=4, label='AIMPC-Real', linestyle='--')
    matplotlib.rcParams.update({'font.size': 25})

    plt.plot_date(data.iloc[2, 0], mpc_mv1.iloc[2, 2], 'r-', linewidth=4, marker='o', markersize=10)
    instant_data = pd.read_csv('./data/mpc_inputs0.csv')
    instant_data['DateTime'] = pd.to_datetime(instant_data['DateTime'])
    plt.plot_date(instant_data.iloc[25:31, 0], mpc_m.iloc[:6, 3], 'r-', linewidth=4, label='AIMPC-Setup3',
                  linestyle='-')
    for g in range(1, int(len(mpc_v) / 18 / 4)):
        instant_data = pd.read_csv('./data/mpc_inputs%s.csv' % (g * 4))
        instant_data['DateTime'] = pd.to_datetime(instant_data['DateTime'])
        plt.plot_date(instant_data.iloc[25, 0], mpc_m.iloc[4 * g * 18, 3], 'r-', marker='o', markersize=10)
        plt.plot_date(instant_data.iloc[25:31, 0], mpc_m.iloc[4 * g * 18:4 * g * 18 + 6, 3], 'r-', linewidth=4)

    # plt.plot_date(date_R_1.iloc[0, 0], R_m.iloc[0, 1], 'k-', marker='^', linewidth=4, markersize=10)
    # plt.plot_date(date_R_1.iloc[:3, 0], R_m_cat[:3], 'k-', linewidth=4, markersize=10,
    #               label='AIMPC-Doctor')
    # for z in range(1, int(len(R_m_cat) / 3 / 2 + 1)):
    #     plt.plot_date(date_R_1.iloc[2 * z, 0], R_m_cat[6 * z], 'k-', marker='^', markersize=10)
    #     plt.plot_date(date_R_1.iloc[2 * z:2 * z + 3, 0], R_m_cat[6 * z:6 * z + 3], 'k-', linewidth=4, markersize=10)

    plt.title('%s' % ('MHDP_1FIC-201A.PV'))
    plt.legend(loc=1)
    plt.grid()
    plt.xticks(fontsize=21.3, rotation=20)
    cf('./fig/%s/%s' % (target, 'mpc_'))
    plt.savefig('./fig/%s/%s/%s_mpc_60mins_run_' % (target, 'mpc_', 'MHDP_1FIC-201A.PV') + str(run) + '.png')
    ####FIC_2312
    plt.figure(figsize=(16.4, 6.6))
    matplotlib.rcParams.update({'font.size': 23})
    # tags_range_low = h2_LL
    # tags_range_up = h2_UL
    # plt.ylim(tags_range_low, tags_range_up)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    date = md.date2num(data.iloc[:, 0])
    plt.plot_date(data.iloc[:, 0], data.loc[:, 'MHDP_1FIC-2312.PV'], 'b-', label='Real data', linewidth=3,
                  linestyle='--')

    plt.plot_date(data.iloc[:, 0], AIMPC_value.iloc[:, 0], 'g-', linewidth=4, label='AIMPC-Real', linestyle='--')

    plt.plot_date(data.iloc[2, 0], mpc_mv1.iloc[2, 3], 'r-', linewidth=4, marker='o', markersize=10)
    instant_data = pd.read_csv('./data/mpc_inputs0.csv')
    instant_data['DateTime'] = pd.to_datetime(instant_data['DateTime'])
    plt.plot_date(instant_data.iloc[25:31, 0], mpc_m.iloc[:6, 4], 'r-', linewidth=4, label='AIMPC-Setup3',
                  linestyle='-')
    for g in range(1, int(len(mpc_v) / 18 / 4)):
        instant_data = pd.read_csv('./data/mpc_inputs%s.csv' % (g * 4))
        instant_data['DateTime'] = pd.to_datetime(instant_data['DateTime'])
        plt.plot_date(instant_data.iloc[25, 0], mpc_m.iloc[4 * g * 18, 4], 'r-', marker='o', markersize=10)
        plt.plot_date(instant_data.iloc[25:31, 0], mpc_m.iloc[4 * g * 18:4 * g * 18 + 6, 4], 'r-', linewidth=4)

    # plt.plot_date(date_R_1.iloc[0, 0], R_m.iloc[0, 2], 'k-', marker='^', linewidth=4, markersize=10)
    # plt.plot_date(date_R_1.iloc[:3, 0], R_m_h2[:3], 'k-', linewidth=4, markersize=10,
    #               label='AIMPC-Doctor')
    # for z in range(1, int(len(R_m_cat) / 3 / 2 + 1)):
    #     plt.plot_date(date_R_1.iloc[2 * z, 0], R_m_h2[6 * z], 'k-', marker='^', markersize=10)
    #     plt.plot_date(date_R_1.iloc[2 * z:2 * zm + 3, 0], R_m_h2[6 * z:6 * z + 3], 'k-', linewidth=4, markersize=10)
    matplotlib.rcParams.update({'font.size': 25})
    plt.title('%s' % ('MHDP_1FIC-2312.PV'))
    plt.legend(loc=1)
    plt.grid()
    plt.xticks(fontsize=21.3, rotation=20)
    cf('./fig/%s/%s' % (target, 'mpc_'))
    plt.savefig('./fig/%s/%s/%s_mpc_60mins_run_' % (target, 'mpc_', 'MHDP_1FIC-2312.PV') + str(run) + '.png')
    plt.close('all')