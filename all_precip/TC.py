import os

import numpy as np
import pandas as pd

import xarray as xa

from tqdm import tqdm


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


def TC_precip(sub_id, sub_lons, sub_lats):
    start, end = start_end_time(sub_id)
    tc_induced_flag = np.zeros(len(start))
    for ind, (time1, time2) in enumerate(tqdm(zip(start, end))):  # time is 12Z.
        total_flag = 0
        start_time = time1
        end_time = time2
        print(start_time, end_time)
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
            # keep the positive and find the min
            start_ind = np.where(diffstart >= 0, diffstart, np.inf).argmin()
            # keep the negative and find the max.
            end_ind = np.where(diffend <= 0, diffend, -np.inf).argmax()
            tc_lat = tc.lat.isel(date_time=slice(start_ind, end_ind + 1)).data
            tc_lon = tc.lon.isel(date_time=slice(start_ind, end_ind + 1)).data
            dis_flag = distance_criteria(
                xs=tc_lon, ys=tc_lat, lons=sub_lons, lats=sub_lats)
            if dis_flag == True:
                # check type
                tc_type = set(tc.usa_status.isel(date_time=slice(start_ind, end_ind+1)).data)
                if tc_type.intersection(set([b'TS', b'TY', b'ST', b'TC', b'HU', b'HR']))==set():
                    pass
                else:
                    total_flag += 1

        if total_flag > 0:
            tc_induced_flag[ind] = 1
            print('TC')
    return tc_induced_flag


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

# TC
tc_record = xa.open_dataset('/tempest/duan0000/exprecip/ERA5/sub_IBTrack.nc')
num_TCs = len(tc_record.storm)
print('NUM_TCs: ', num_TCs)

for sub_id in range(7, 8):
    print(sub_id, ' SUB_ID ')
    lons_sub = np.load('../ERA5/Calculations/' + str(sub_id) + '_lons.npy')
    lats_sub = np.load('../ERA5/Calculations/' + str(sub_id) + '_lats.npy')
    TC_induced_flag = TC_precip(
        sub_id=sub_id, sub_lats=lats_sub, sub_lons=lons_sub)
    np.save('CorrectedTC/'+str(sub_id) + '_all_TC', TC_induced_flag)
