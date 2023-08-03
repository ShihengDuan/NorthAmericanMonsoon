import os

import numpy as np
import pandas as pd
import shapefile
import xarray as xa
from shapely.geometry import Point, shape

from utils import TUTT, load_Surge, tutt_PAU


def start_end_time(sub_id):
    start = []
    end = []
    ext_times = ext_days[sub_id]
    length = len(ext_times)
    i = 0
    while i <= length - 1:
        j = 1
        while (j + i <= length - 1) and ((ext_times[i + j] - ext_times[i + j - 1]) / np.timedelta64(1, 'D') == 1):
            j += 1
        start.append(ext_times[i] - np.timedelta64(12, 'h'))
        end.append(ext_times[i + j - 1] + np.timedelta64(12, 'h'))
        i += j
    return start, end


def is_monsoon_precip(month):
    return (month <= 9) & (month >= 6)


def distance_criteria(xs, ys, lons, lats):  # return distance.
    dis = 12345
    for x, y in zip(xs, ys):
        if x > 180: x = x - 360
        for lon, lat in zip(lons, lats):
            if lon > 180: lon = lon - 360
            distance = np.sqrt(np.square(x - lon) + np.square(y - lat))
            if distance <= 5:
                return True
            if dis > distance:
                dis = distance
    return dis


def TUTT_precip(sub_id, sub_lons, sub_lats):
    start, end = start_end_time(sub_id)
    TUTT_induced_flag = np.zeros(len(start))
    TUTT_direction = np.zeros(len(start))
    for ind, (time1, time2) in enumerate(zip(start, end)):  # time is 12Z.
        print(time1, time2)
        total_flag = 0
        start_time = time1
        end_time = time2
        for i in range(len(tutts)):
            tutt = tutts[i]
            tutt_time = tutt.time[0]
            tutt_start = tutt_time[0]
            tutt_end = tutt_time[-1]
            if tutt_start > end_time:
                continue
            if tutt_end < start_time:
                continue
            common_start = np.max([start_time, tutt_start])
            common_end = np.min([end_time, tutt_end])
            diffstart = (tutt_time - common_start) / np.timedelta64(1, 'h')
            diffend = (tutt_time - common_end) / np.timedelta64(1, 'h')
            start_ind = np.where(diffstart >= 0, diffstart, np.inf).argmin()  # keep the positive and find the min
            end_ind = np.where(diffend <= 0, diffend, -np.inf).argmax()  # keep the negative and find the max.
            tutt_lat = tutt.lats[0][start_ind: end_ind + 1]
            tutt_lon = tutt.lons[0][start_ind: end_ind + 1]
            dis_flag = distance_criteria(xs=tutt_lon, ys=tutt_lat, lons=sub_lons, lats=sub_lats)
            if dis_flag == True:
                total_flag = True
            print(i, time1, start_ind, end_ind, 'START, END', dis_flag)
            if dis_flag == True:
                initial_lon = tutt.lons[0][0] # whole track direction
                end_lon = tutt.lons[0][-1]
                if end_lon-initial_lon>0:
                    print('Eastward RWB')
                    if TUTT_direction[ind] == 0:
                        TUTT_direction[ind] = -1
                    elif TUTT_direction[ind]>0:
                        print('Conflict!!!!!!!!!!!!!!!!')
                else:
                    print('Westward TUTTs')
                    if TUTT_direction[ind]<0:
                        print('Conflict!!!!!!!!!!!!!!!!')
                    else:
                        TUTT_direction[ind] = 1
        if total_flag:
            TUTT_induced_flag[ind] = 1
            print('TUTT')
    return TUTT_induced_flag, TUTT_direction

data = xa.open_dataarray("/tempest/duan0000/exprecip/cpc-global/NAM_sub_precip")  # CPC
monsoon_precip = data.sel(time=(is_monsoon_precip(data.time.dt.month)))
monsoon_precip = monsoon_precip.sel(time=(monsoon_precip.time.dt.year < 2019))
del data
ext_days = {}
for sub_id in range(1, 8):
    print("sub_id: ", sub_id)
    precip = monsoon_precip.sel(sub_id=sub_id)
    precip_data = precip.data
    q1 = np.quantile(precip_data[precip_data > 1], 0.05)
    q2 = np.quantile(precip_data[precip_data > 1], 0.95)
    ext_time = precip.where(precip > q2, drop=True).time.data
    print(len(ext_time), q2)
    ext_days[sub_id] = ext_time

# TUTT
# tutts = load_TUTT()
tutts, total_N = tutt_PAU('/tempest/duan0000/exprecip/ERA5_TUTT_tracks_PV2e-6_noTC.txt')

for sub_id in range(1, 8):
    print(sub_id, ' SUB_ID ')
    lons_sub = np.load('Calculations/' + str(sub_id) + '_lons.npy')
    lats_sub = np.load('Calculations/' + str(sub_id) + '_lats.npy')
    TUTT_induced_flag, TUTT_direction = TUTT_precip(sub_id, sub_lons=lons_sub, sub_lats=lats_sub)
    np.save('PAU/'+str(sub_id)+'_TUTT_direction-5degree', TUTT_direction)
    np.save('PAU/'+str(sub_id)+'_TUTT_induced-5degree', TUTT_induced_flag)
