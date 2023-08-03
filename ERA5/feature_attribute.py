import os

import numpy as np
import pandas as pd
import shapefile
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
    for ind, (time1, time2) in enumerate(zip(start, end)):  # time is 12Z.
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
            start_ind = np.where(diffstart >= 0, diffstart, np.inf).argmin()  # keep the positive and find the min
            end_ind = np.where(diffend <= 0, diffend, -np.inf).argmax()  # keep the negative and find the max.
            tc_lat = tc.lat.isel(date_time=slice(start_ind, end_ind + 1)).data
            tc_lon = tc.lon.isel(date_time=slice(start_ind, end_ind + 1)).data
            dis_flag = distance_criteria(xs=tc_lon, ys=tc_lat, lons=sub_lons, lats=sub_lats)
            total_flag += dis_flag

        if total_flag > 0:
            tc_induced_flag[ind] = 1
            print('TC')
    return tc_induced_flag


def TUTT_precip(sub_id, sub_lons, sub_lats):
    TUTT_induced_flag = np.zeros(ext_days[sub_id].shape)
    start, end = start_end_time(sub_id)
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
            print(start_ind, end_ind, 'START, END', dis_flag)
        if total_flag:
            TUTT_induced_flag[ind] = 1
            print('TUTT')
    return TUTT_induced_flag


def surge_precip(sub_id):
    start, end = start_end_time(sub_id)
    surge_induced_flag = np.zeros(len(start))
    for ind, (time1, time2) in enumerate(zip(start, end)):  # time is 12Z.
        total_flag = 0
        start_time = time1
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
    if len(path)==0:
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
    fname = "/tempest/duan0000/exprecip/cpc-global/NAM_SUB_SHAPES/NAM-mask-" + str(sub_id) + ".shp"
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
    for i, date in enumerate(end):
        year = pd.to_datetime(date).year
        if year > 2000:
            cloudtracknumber, lat, lon = check_date(date)
            if cloudtracknumber is None: # Missing files
                MCS_flag[i]=-2
            else:
                total_point = MCS_field(sub_id, cloudtracknumber)
                if total_point > 1:
                    MCS_flag[i] = 1
        else:
            MCS_flag[i] = -1  # dataset not available
    return MCS_flag


def Surge_TC_TUTT(sub_id, sub_lons, sub_lats):
    start, end = start_end_time(sub_id)
    TUTT_induced_flag = np.zeros(len(start))
    tc_induced_flag = np.zeros(len(start))
    surge_induced_flag = np.zeros(len(start))
    for ind, (time1, time2) in enumerate(zip(start, end)):  # time is 12Z.
        print(time1, ' ', time2)
        start_time = time1
        end_time = time2
        # TUTT
        total_flag = 0
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
            # print(start_ind, end_ind, 'START, END')
        if total_flag == True:
            TUTT_induced_flag[ind] = 1
            print('TUTT')
        # Surge
        total_flag = 0
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
                # print('Surge ', common_start, common_end)
        if total_flag > 0:
            surge_induced_flag[ind] += total_flag
            print('Surge')
        # TC
        total_flag = 0
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
            start_ind = np.where(diffstart >= 0, diffstart, np.inf).argmin()  # keep the positive and find the min
            end_ind = np.where(diffend <= 0, diffend, -np.inf).argmax()  # keep the negative and find the max.
            tc_lat = tc.lat.isel(date_time=slice(start_ind, end_ind + 1)).data
            tc_lon = tc.lon.isel(date_time=slice(start_ind, end_ind + 1)).data
            dis_flag = distance_criteria(xs=tc_lon, ys=tc_lat, lons=sub_lons, lats=sub_lats)
            if dis_flag == True:
                total_flag = True
            # print(start_ind, end_ind, 'START, END')
        if total_flag == True:
            tc_induced_flag[ind] = 1
            print('TC')
    return surge_induced_flag, tc_induced_flag, TUTT_induced_flag


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
# TUTT
# tutts = load_TUTT()
tutts, total_N = tutt_PAU('/tempest/duan0000/exprecip/ERA5_TUTT_tracks_PV2e-6_noTC.txt')
# tutts, total_N = tutt_PAU('/tempest/duan0000/exprecip/ERA5_TUTT_tracks_Psi1e6_10deg_noTC.txt')
# Surge
surge_record_6h = load_Surge()

for sub_id in range(1, 8):
    print(sub_id, ' SUB_ID ')
    lons_sub = np.load('Calculations/' + str(sub_id) + '_lons.npy')
    lats_sub = np.load('Calculations/' + str(sub_id) + '_lats.npy')
    MCS_induced_flag = MCS_precip(sub_id)
    surge_induced_flag, tc_induced_flag, TUTT_induced_flag = Surge_TC_TUTT(sub_id=sub_id, sub_lats=lats_sub,
                                                                           sub_lons=lons_sub)

    np.save('PAU/' + str(sub_id) + '_surge_induced_flag', surge_induced_flag)
    np.save('PAU/' + str(sub_id) + '_tc_induced_flag', tc_induced_flag)
    # np.save('PAU/Psi-' + str(sub_id) + '_TUTT_induced_flag', TUTT_induced_flag)
    np.save('PAU/' + str(sub_id) + '_TUTT_induced_flag', TUTT_induced_flag)
    np.save('PAU/MCS_' + str(sub_id), MCS_induced_flag)
    print(np.sum(tc_induced_flag > 0), ' TC')
    print(np.sum(surge_induced_flag > 0), ' Surge')
    print(np.sum(TUTT_induced_flag > 0), ' TUTT')
    print(np.sum(MCS_induced_flag > 0), ' MCS')
    feature_induced_flag = tc_induced_flag.astype(bool) | surge_induced_flag.astype(bool) | \
                           TUTT_induced_flag.astype(bool) | (MCS_induced_flag > 0).astype(bool)
    start, end = start_end_time(sub_id)
    start = np.array(start)
    end = np.array(end)

    with open("PAU/PV-non_features_days_MCS.txt", "a") as outfile:
        for time1, time2 in zip(start[~feature_induced_flag], end[~feature_induced_flag]):
            print(sub_id, "\t", time1, '\t', time2, file=outfile)
