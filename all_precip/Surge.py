import os

import numpy as np
import pandas as pd
import shapefile
import xarray as xa
from shapely.geometry import Point, shape
from tqdm import tqdm
from utils import load_Surge


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


def surge_precip(sub_id):
    days = {'1': 2, '2': 1, '3': 2, '4': 2, '5': 2, '6': 3, '7': 0}
    start, end = start_end_time(sub_id)
    surge_induced_flag = np.zeros(len(start))
    precip_amount = np.zeros(len(start))
    day = days[str(sub_id)]  # time window size
    for ind, (time1, time2) in enumerate(zip(start, end)):  # time is 12Z.
        precip_amount[ind] = monsoon_precip.sel(
            sub_id=sub_id).sel(time=time1+np.timedelta64(12, 'h'))
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
    return surge_induced_flag, precip_amount


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

surge_record_6h = load_Surge()

for sub_id in range(1, 8):
    print(sub_id, ' SUB_ID ')

    surge_induced_flag, precip_amount = surge_precip(sub_id=sub_id)
    np.save(str(sub_id) + '_all_surge', surge_induced_flag)
