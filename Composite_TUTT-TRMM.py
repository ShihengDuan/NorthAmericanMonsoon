import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import shapefile
import xarray as xa
from shapely.geometry import Point, shape
from matplotlib import pyplot as plt
from ERA5.utils import tutt_PAU


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

tutts, total_N = tutt_PAU('/tempest/duan0000/exprecip/ERA5_TUTT_tracks_PV2e-6_noTC.txt') # 6-hr. 0, 6, 12, 18
print(len(tutts))
# Clean TUTTs to save only NAM season
NAM_tutt = []
for tutt in tutts:
    tutt_time = tutt.time[0]
    tutt_start = tutt_time[0]
    tutt_end = tutt_time[-1]
    pd_time_start = pd.to_datetime(tutt_start)
    pd_time_end = pd.to_datetime(tutt_end)

    if pd_time_start.month>9 or pd_time_end.month<6 or pd_time_start.year>2012 or pd_time_start.year<1998:
        continue
    else:
        NAM_tutt.append(tutt)
print(len(NAM_tutt))

NAM_min_lat = 22.75 - 20
NAM_max_lat = 39.75 + 20
NAM_min_lon = -117.75 + 360 - 20
NAM_max_lon = -105.75 + 360 + 20

trmm_anomaly = xa.open_dataarray('/tempest/duan0000/exprecip/TRMM_NAM_anomaly.nc')
anomaly_composite = []
all_tutt_lon = []
all_tutt_lat = []
count = np.zeros((160, 160))
for tutt in tqdm(NAM_tutt[:]):
    tutt_time = tutt.time[0]
    lons = tutt.lons[0]
    lats = tutt.lats[0]
    for i, time in enumerate(tutt_time):
        count[:]+=1
        time_pd = pd.to_datetime(time)
        year = time_pd.year
        month = time_pd.month
        date = time_pd.day
        hour = time_pd.hour
        # print(year, month, date, hour)
        # Find time step in TRMM dataset
        timestep1 = np.datetime64(str(year)+'-'+str(month).zfill(2)+'-'+str(date).zfill(2)+'T'+str(hour).zfill(2))-np.timedelta64(90, 'm')
        timestep2 = np.datetime64(str(year)+'-'+str(month).zfill(2)+'-'+str(date).zfill(2)+'T'+str(hour).zfill(2))+np.timedelta64(90, 'm')
        
        lon = lons[i]
        lat = lats[i]
        all_tutt_lon.append(lon)
        all_tutt_lat.append(lat)
        min_lon = lon-20+360
        max_lon = lon+20+360
        min_lat = lat-20
        max_lat = lat+20
        
        precip_anomaly1 = trmm_anomaly.sel(time=timestep1)
        precip_anomaly1 = precip_anomaly1.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon)).data
        # print(precip_anomaly1.shape)
        # print(min_lat, max_lat, min_lon, max_lon)
        precip_anomaly2 = trmm_anomaly.sel(time=timestep2)
        precip_anomaly2 = precip_anomaly2.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon)).data # Lat, Lon
        if precip_anomaly1.shape[0]<160:
            ind = precip_anomaly1.shape[0]
            
            temp = np.zeros((160, 160))
            temp[:] = np.NAN
            temp[:precip_anomaly1.shape[0]] = precip_anomaly1
            precip_anomaly1 = temp
            count[ind:, :] = count[ind:, :]-1

        if precip_anomaly2.shape[0]<160:
            temp = np.zeros((160, 160))
            temp[:] = np.NAN
            temp[:precip_anomaly2.shape[0]] = precip_anomaly2
            precip_anomaly2 = temp
        precip_anomaly1 = precip_anomaly1[:160, :160]
        precip_anomaly2 = precip_anomaly2[:160, :160]
        precip_anomaly = (precip_anomaly1+precip_anomaly2)/2
        # print(precip_anomaly2.shape)
        anomaly_composite.append(precip_anomaly.reshape(1, 160, 160)) # time, lat, lon

print(np.min(all_tutt_lat), ' ', np.max(all_tutt_lat))
print(np.min(all_tutt_lon), ' ', np.max(all_tutt_lon))

anomaly_composite = np.concatenate(anomaly_composite)
print(anomaly_composite.shape)
np.save('TUTT_anomaly_trmm_20degree', anomaly_composite)
np.save('TUTT_anomaly_trmm_20degree_count', count)
plt.imshow(anomaly_composite.mean(axis=0))
plt.show()