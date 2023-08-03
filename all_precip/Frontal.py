import os

import numpy as np
import pandas as pd
import shapefile
import xarray as xa
from shapely.geometry import Point, shape
from tqdm import tqdm

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

def front_precip(sub_id):
    lons_sub = np.load('../ERA5/Calculations/' + str(sub_id) + '_lons.npy')
    lats_sub = np.load('../ERA5/Calculations/' + str(sub_id) + '_lats.npy')
    min_lat = np.min(lats_sub)
    max_lat = np.max(lats_sub)
    min_lon = np.min(lons_sub)
    max_lon = np.max(lons_sub)

    start, end = start_end_time(sub_id)
    front_induced_flag = np.zeros(len(start))
    for ind, (time1, time2) in enumerate(zip(start, end)):  # time is 12Z.
        year = pd.to_datetime(time1).year
        if year < 2003:
            front_induced_flag[ind] = -1
            continue
        else:
            print(time1, time2)
            month = pd.to_datetime(time1).month
            day1 = pd.to_datetime(time1).day
            day2 = pd.to_datetime(time2).day
            file = '/tempest/duan0000/exprecip/front/CODSUS/MERRA2/codsus_merra2-1deg_1wide_' + str(year) + '.nc'
            front = xa.open_dataset(file)
            front = front.fronts
            front_slice = front.sel(time=slice(time1, time2))
            front_sub = front_slice.sel(lat=slice(min_lat - 5, max_lat + 5), lon=slice(min_lon - 5, max_lon + 5))
            # time, lat, lon
            for x in front_sub.lon:
                for y in front_sub.lat:
                    # print(front_sub.shape, ' ', front_sub)
                    if front_sub.sel(lon=x, lat=y, front=slice(0, 4)).sum(
                            dim=['time', 'front']) > 0:  # there is a front point
                        dis = distance_criteria_midtro(x, y, lons_sub, lats_sub)
                        if dis == True:
                            front_induced_flag[ind] += 1
            if front_induced_flag[ind]>0:
                print('Front: ', front_induced_flag[ind])

    return front_induced_flag

data = xa.open_dataarray("/tempest/duan0000/exprecip/cpc-global/NAM_sub_precip")  # CPC
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

for sub_id in range(1, 8):
    print(sub_id, ' SUB_ID ')
    lons_sub = np.load('../ERA5/Calculations/' + str(sub_id) + '_lons.npy')
    lats_sub = np.load('../ERA5/Calculations/' + str(sub_id) + '_lats.npy')
    front_induced_flag = front_precip(sub_id)
    print(front_induced_flag)

    np.save(str(sub_id) + '_all_front', front_induced_flag)
    