import os

import numpy as np
import pandas as pd

import xarray as xa

from tqdm import tqdm
from utils import tutt_PAU


def is_monsoon_precip(month):
    return (month <= 9) & (month >= 6)


def start_end_time(sub_id):  # for all precipitation events.
    start = []
    end = []
    ext_times = precip_days[sub_id]
    length = len(ext_times)
    i = 0
    while i <= length - 1:
        start.append(ext_times[i] - np.timedelta64(12, 'h'))
        end.append(ext_times[i] + np.timedelta64(12, 'h'))  # 12Z to 12Z
        i += 1
    return start, end


def distance_criteria(xs, ys, lons, lats):  # return distance.
    dis = 12345
    for x, y in zip(xs, ys):
        if x > 180:
            x = x - 360
        for lon, lat in zip(lons, lats):
            if lon > 180:
                lon = lon - 360
            distance = np.sqrt(np.square(x - lon) + np.square(y - lat))
            if distance <= 5:
                return True
            if dis > distance:
                dis = distance
    return dis


def TUTT_precip(sub_id, sub_lons, sub_lats):
    TUTT_induced_flag = np.zeros(precip_days[sub_id].shape)
    TUTT_direction = np.zeros(precip_days[sub_id].shape)
    start, end = start_end_time(sub_id)
    for ind, (time1, time2) in enumerate(tqdm(zip(start, end))):  # time is 12Z.
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
            # keep the positive and find the min
            start_ind = np.where(diffstart >= 0, diffstart, np.inf).argmin()
            # keep the negative and find the max.
            end_ind = np.where(diffend <= 0, diffend, -np.inf).argmax()
            tutt_lat = tutt.lats[0][start_ind: end_ind + 1]
            tutt_lon = tutt.lons[0][start_ind: end_ind + 1]
            dis_flag = distance_criteria(
                xs=tutt_lon, ys=tutt_lat, lons=sub_lons, lats=sub_lats)
            if dis_flag == True:
                total_flag = True
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
            print(start_ind, end_ind, 'START, END', dis_flag)
        if total_flag:
            TUTT_induced_flag[ind] = 1
            print('TUTT')
    return TUTT_induced_flag, TUTT_direction


data = xa.open_dataarray(
    "/tempest/duan0000/exprecip/cpc-global/NAM_sub_precip")  # CPC
monsoon_precip = data.sel(time=(is_monsoon_precip(data.time.dt.month)))
monsoon_precip = monsoon_precip.sel(time=(monsoon_precip.time.dt.year < 2019))

precip_days = {}
for sub_id in range(1, 8):
    print("sub_id: ", sub_id)
    precip = monsoon_precip.sel(sub_id=sub_id)
    precip_data = precip.data
    precip_time = precip.where(precip > 1, drop=True).time.data
    print(len(precip_time))
    precip_days[sub_id] = precip_time

# TUTT
tutts, total_N = tutt_PAU(
    '/tempest/duan0000/exprecip/ERA5_TUTT_tracks_PV2e-6_noTC.txt')
# tutts, total_N = tutt_PAU('/tempest/duan0000/exprecip/ERA5_TUTT_tracks_Psi1e6_10deg_noTC.txt')


for sub_id in range(1, 8):
    print(sub_id, ' SUB_ID ')
    lons_sub = np.load('../ERA5/Calculations/' + str(sub_id) + '_lons.npy')
    lats_sub = np.load('../ERA5/Calculations/' + str(sub_id) + '_lats.npy')
    TUTT_induced_flag, TUTT_direction = TUTT_precip(sub_id, sub_lats=lats_sub, sub_lons=lons_sub)
    np.save(str(sub_id) + '_all_TUTT-5degree', TUTT_induced_flag)
    np.save(str(sub_id) + '_all_TUTT-types-5degree', TUTT_direction)
