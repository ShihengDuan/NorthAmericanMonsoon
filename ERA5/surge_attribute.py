import os

import numpy as np
import pandas as pd

import xarray as xa


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
def surge_precip(sub_id):
    start, end = start_end_time(sub_id)
    surge_induced_flag = np.zeros(len(start))
    day = days[str(sub_id)] # time window size
    for ind, (time1, time2) in enumerate(zip(start, end)):  # time is 12Z.
        total_flag = 0
        start_time = time1 - np.timedelta64(day, 'D')
        end_time = time2
        for surge in surge_record_6h:
            surge_start = surge.start_time
            surge_end = surge.end_time
            if surge_start > end_time:
                continue
            if surge_end < start_time:
                continue
            common_start = np.max([start_time, surge_start])
            common_end = np.min([end_time, surge_end])
            if common_start <= common_end:
                total_flag += 1
        if total_flag > 0:
            surge_induced_flag[ind] += total_flag
    return surge_induced_flag

def is_monsoon_precip(month):
    return (month <= 9) & (month >= 6)

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


surge_record_6h = load_Surge()
days = {'1':2, '2':1, '3':2, '4':2, '5':2, '6':3, '7':0}
for sub_id in range(1, 8):
    print(sub_id, ' SUB_ID ')
    surge_induced_flag = surge_precip(sub_id=sub_id)
    print(np.sum(surge_induced_flag>0), ' ', len(surge_induced_flag))
    np.save('PAU/' + str(sub_id) + '_surgeWINDOW_induced_flag', surge_induced_flag)
    