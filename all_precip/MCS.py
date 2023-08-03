import os

import numpy as np
import pandas as pd
import shapefile
import xarray as xa
from shapely.geometry import Point, shape
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


def check_date(date):  # date-12h
    # for MCS
    start = date - np.timedelta64(24, "h") + np.timedelta64(30, "m")
    end = date - np.timedelta64(30, "m")
    year = pd.to_datetime(start).year
    month = pd.to_datetime(start).month
    day = pd.to_datetime(start).day
    path = "../MCS/mcstracking/" + str(year) + "/" + str(month).zfill(2) + "/mcstrack_" + str(year) + str(month).zfill(
        2) + str(day).zfill(2) + "_"
    path1 = [path + str(12 + i) + "30.nc" for i in range(12)]
    path1 = [path for path in path1 if os.path.isfile(path)]
    day = pd.to_datetime(end).day
    path = "../MCS/mcstracking/" + str(year) + "/" + str(month).zfill(2) + "/mcstrack_" + str(year) + str(month).zfill(
        2) + str(day).zfill(2) + "_"
    path2 = [path + str(i).zfill(2) + "30.nc" for i in range(12)]
    path2 = [path for path in path2 if os.path.isfile(path)]
    path = path1+path2
    if len(path) == 0:
        return None, None, None
    else:
        data = xa.open_mfdataset(path)
        cloudtracknumber = data.cloudtracknumber
        cloudtracknumber = cloudtracknumber.compute()
        lat = cloudtracknumber.lat
        lon = cloudtracknumber.lon
        cloudtracknumber = cloudtracknumber.sum(dim='time', skipna=True)
        return cloudtracknumber, lat, lon


def MCS_field(sub_id, field):  # check one timeframe
    fname = "/tempest/duan0000/exprecip/cpc-global/NAM_SUB_SHAPES/NAM-mask-" + \
        str(sub_id) + ".shp"
    shape_file = shapefile.Reader(fname)
    feature = shape_file.shapeRecords()[0]
    first = feature.shape.__geo_interface__
    polygon = shape(first)
    lon0, lat0, lon1, lat1 = polygon.bounds
    sub_field = field.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
    total = 0
    for i, lat in enumerate(sub_field.lat):
        for j, lon in enumerate(sub_field.lon):
            if polygon.contains(Point(lon, lat)) and sub_field[i, j] > 0:
                total += 1
    return total


def MCS_precip(sub_id):
    start, end = start_end_time(sub_id)
    MCS_flag = np.zeros(len(start))
    for i, date in enumerate(tqdm(end, desc='MCS')):
        year = pd.to_datetime(date).year
        if year > 2000:
            cloudtracknumber, lat, lon = check_date(date)
            if cloudtracknumber is None:  # Missing files
                MCS_flag[i] = -2
            else:
                total_point = MCS_field(sub_id, cloudtracknumber)
                if total_point > 1:
                    MCS_flag[i] = 1
        else:
            MCS_flag[i] = -1  # dataset not available
    return MCS_flag


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


for sub_id in range(1, 8):
    print(sub_id, ' SUB_ID ')
    lons_sub = np.load('../ERA5/Calculations/' + str(sub_id) + '_lons.npy')
    lats_sub = np.load('../ERA5/Calculations/' + str(sub_id) + '_lats.npy')
    MCS_induced_flag = MCS_precip(sub_id)
    np.save(str(sub_id) + '_all_MCS', MCS_induced_flag)
