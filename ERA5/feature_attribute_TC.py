from datetime import datetime
import os
import pickle
import numpy as np
import pandas as pd
import xarray as xa
from shapely.geometry import Point, shape

from utils import load_Surge, tutt_PAU


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


def TC_precip(sub_id, sub_lons, sub_lats):
    start, end = start_end_time(sub_id)
    tc_induced_flag = np.zeros(len(start))
    tc_types = {}
    for ind, (time1, time2) in enumerate(zip(start, end)):  # time is 12Z.
        total_flag = 0
        start_time = time1
        end_time = time2
        print(start_time, end_time)
        tc_type = []
        for i in range(num_TCs):
            tc = tc_record.isel(storm=i)
            tc_time = tc.time.dropna(dim='date_time').data
            tc_start = tc_time[0]
            tc_end = tc_time[-1]
            if tc_start > end_time:
                continue
            if tc_end < start_time:
                continue
            common_start = np.max([start_time, tc_start])
            common_end = np.min([end_time, tc_end])
            diffstart = (tc_time - common_start) / np.timedelta64(1, 'h')
            diffend = (tc_time - common_end) / np.timedelta64(1, 'h')
            # tc_slice = tc.sel(date_time=slice(common_start, common_end))
            start_ind = np.where(diffstart >= 0, diffstart, np.inf).argmin()  # keep the positive and find the min
            end_ind = np.where(diffend <= 0, diffend, -np.inf).argmax()  # keep the negative and find the max.
            tc_lat = tc.lat.isel(date_time=slice(start_ind, end_ind + 1)).data
            tc_lon = tc.lon.isel(date_time=slice(start_ind, end_ind + 1)).data
            dis_flag = distance_criteria(xs=tc_lon, ys=tc_lat, lons=sub_lons, lats=sub_lats)
            if dis_flag == True:
                total_flag = True
                print(set(tc.usa_status.isel(date_time=slice(start_ind, end_ind+1)).data))
                tc_type.append(set(tc.usa_status.isel(date_time=slice(start_ind, end_ind+1)).data))
            
            # print(start_ind, end_ind, 'START, END')
        if total_flag == True:
            tc_induced_flag[ind] = 1
            print('TC')
            tc_types[time2] = tc_type
    return tc_induced_flag, tc_types


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
# TC
tc_record = xa.open_dataset('/tempest/duan0000/exprecip/ERA5/sub_IBTrack.nc')
num_TCs = len(tc_record.storm)
print('NUM_TCs: ', num_TCs)

for sub_id in range(1, 8):
    print(sub_id, ' SUB_ID ')
    lons_sub = np.load('Calculations/' + str(sub_id) + '_lons.npy')
    lats_sub = np.load('Calculations/' + str(sub_id) + '_lats.npy')
    
    tc_induced_flag, tc_types = TC_precip(sub_id=sub_id, sub_lons=lons_sub, sub_lats=lats_sub)
    with open(str(sub_id)+'_tc_types.pickle', 'wb') as handle:
        pickle.dump(tc_types, handle, protocol=pickle.HIGHEST_PROTOCOL)

    