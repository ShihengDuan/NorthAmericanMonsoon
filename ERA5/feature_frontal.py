import numpy as np
import pandas as pd
import xarray as xa
import pickle

def distance_criteria(x, y, lons, lats):  # return distance.
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


def front_precip(sub_id):
    lons_sub = np.load('Calculations/' + str(sub_id) + '_lons.npy')
    lats_sub = np.load('Calculations/' + str(sub_id) + '_lats.npy')
    min_lat = np.min(lats_sub)
    max_lat = np.max(lats_sub)
    min_lon = np.min(lons_sub)
    max_lon = np.max(lons_sub)

    start, end = start_end_time(sub_id)
    front_induced_flag = np.zeros(len(start))
    front_type = {}
    for ind, (time1, time2) in enumerate(zip(start, end)):  # time is 12Z.
        year = pd.to_datetime(time1).year
        if year < 2003:
            front_induced_flag[ind] = -1
            front_type[ind]=-1
            continue
        else:
            types = []
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
                        dis = distance_criteria(x, y, lons_sub, lats_sub)
                        if dis == True:
                            front_induced_flag[ind] += 1
                            for f in range(0, 4):
                                if front_sub.sel(lon=x, lat=y, front=f).sum(dim='time')>0:
                                    types.append(f)
                                    # print('type: ', f)

            if front_induced_flag[ind]>0:
                print('Front: ', front_induced_flag[ind])
                front_type[ind] = types
                print(types)
    return front_induced_flag, front_type


for sub_id in range(1, 8):
    print(sub_id, ' SUB_ID')
    front_induced_flag, front_type = front_precip(sub_id)
    np.save('PAU/' + str(sub_id) + '_front_updated', front_induced_flag)
    with open('PAU/front_type_'+str(sub_id)+'.pickle', 'wb') as handle:
        pickle.dump(front_type, handle, protocol=pickle.HIGHEST_PROTOCOL)
