import os

import numpy as np
import pandas as pd

import xarray as xa



def is_monsoon_precip(month):
    return (month <= 9) & (month >= 6)
def start_end_time(sub_id): # for all precipitation events. 
    start = []
    end = []
    ext_times = precip_days[sub_id]
    length = len(ext_times)
    i = 0
    while i <= length - 1:
        start.append(ext_times[i] - np.timedelta64(12, 'h'))
        end.append(ext_times[i] + np.timedelta64(12, 'h')) # 12Z to 12Z
        i += 1
    return start, end

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
def distance_criteria_midtro(x, y, lons, lats):  # return distance.
    dis = 12345
    if x > 180: x = x - 360
    for lon, lat in zip(lons, lats):
        if lon > 180: lon = lon - 360
        distance = np.sqrt(np.square(x - lon) + np.square(y - lat))
        if distance <= 5:
            return True
        if dis > distance:
            dis = distance
    return dis


data = xa.open_dataarray("/tempest/duan0000/exprecip/cpc-global/NAM_sub_precip")  # CPC
monsoon_precip = data.sel(time=(is_monsoon_precip(data.time.dt.month)))
monsoon_precip = monsoon_precip.sel(time=(monsoon_precip.time.dt.year < 2019))

def cal_anomaly(var):
    month_date = var.time.dt.strftime("%m-%d")
    var = var.assign_coords(month_date=("time", month_date))
    climate_var = var.groupby(var.month_date).mean()
    std_var = var.groupby(var.month_date).std()
    anomaly_var = var.groupby(var.month_date) - climate_var
    anomaly_var_stand = anomaly_var.groupby(anomaly_var.month_date) / std_var
    return anomaly_var_stand, anomaly_var


z500 = xa.open_dataarray("../ERA5/meteoData/1979-2018_z500_all.nc")
z500_anomaly_std, z500_anomaly = cal_anomaly(z500)
print(z500_anomaly)

precip_days = {}
for sub_id in range(1, 8):
    print("sub_id: ", sub_id)
    precip = monsoon_precip.sel(sub_id=sub_id)
    precip_data = precip.data
    precip_time = precip.where(precip > 1, drop=True).time.data
    print(len(precip_time))
    precip_days[sub_id] = precip_time

for sub_id in range(1, 3):
    print(sub_id, ' SUB_ID ')
    lons_sub = np.load('../ERA5/Calculations/' + str(sub_id) + '_lons.npy')
    lats_sub = np.load('../ERA5/Calculations/' + str(sub_id) + '_lats.npy')

    start, end = start_end_time(sub_id)
    min_lat = np.min(lats_sub)
    max_lat = np.max(lats_sub)
    min_lon = np.min(lons_sub)
    max_lon = np.max(lons_sub)
    z500_anomaly_sub = z500_anomaly.sel(latitude=slice(max_lat + 5, min_lat - 5),
                                        longitude=slice(min_lon + 360 - 5, max_lon + 360 + 5))
    print(sub_id, ' ', z500_anomaly_sub.shape)
    mid_feature = np.zeros(len(end))
    for t, time in enumerate(end):
        time_frame = time
        # print(time_frame, ' ', type(time_frame), ' ', time)
        z500_anomaly_sub_frame = z500_anomaly_sub.sel(time=time)
        for i in range(len(z500_anomaly_sub.latitude)):
            for j in range(len(z500_anomaly_sub.longitude)):
                if z500_anomaly_sub_frame.isel(latitude=i, longitude=j) <= -1000:
                    if distance_criteria_midtro(z500_anomaly_sub.longitude[j], z500_anomaly_sub.latitude[i], lons_sub,
                                         lats_sub) == True:
                        mid_feature[t] += 1
        print(mid_feature[t], ' ', time)
    
    np.save(str(sub_id) + '_all_midTropo', mid_feature)
    