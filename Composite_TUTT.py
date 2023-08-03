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

tutts, total_N = tutt_PAU('/tempest/duan0000/exprecip/ERA5_TUTT_tracks_PV2e-6_noTC.txt')
print(len(tutts))
# Clean TUTTs to save only NAM season
NAM_tutt = []
for tutt in tutts:
    tutt_time = tutt.time[0]
    tutt_start = tutt_time[0]
    tutt_end = tutt_time[-1]
    pd_time_start = pd.to_datetime(tutt_start)
    pd_time_end = pd.to_datetime(tutt_end)

    if pd_time_start.month>9 or pd_time_end.month<6 or pd_time_start.year>2019:
        continue
    else:
        NAM_tutt.append(tutt)
print(len(NAM_tutt))

cpc_anomaly = xa.open_dataarray('/tempest/duan0000/exprecip/cpc-global/nc-data/Anomaly_cpc.nc')
anomaly_composite = []
for tutt in tqdm(NAM_tutt[:]):
    tutt_time = tutt.time[0]
    lons = tutt.lons[0]
    lats = tutt.lats[0]
    for i, time in enumerate(tutt_time):
        time_pd = pd.to_datetime(time)
        year = time_pd.year
        month = time_pd.month
        date = time_pd.day
        hour = time_pd.hour
        print(year, month, date, hour)
        # Find time step in CPC dataset
        timestep = np.datetime64(str(year)+'-'+str(month).zfill(2)+'-'+str(date).zfill(2))
        if hour>12:
            timestep+=np.timedelta64(1, 'D')
        lon = lons[i]
        lat = lats[i]
        min_lon = lon-20
        max_lon = lon+20
        min_lat = lat-20
        max_lat = lat+20
        precip_anomaly = cpc_anomaly.sel(time=timestep)
        precip_anomaly = precip_anomaly.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon+360, max_lon+360))
        anomaly_composite.append(precip_anomaly.data[:80, :80].reshape(1, 80, 80)) # time, lat, lon

anomaly_composite = np.concatenate(anomaly_composite)
print(anomaly_composite.shape)
np.save('TUTT_anomaly_precip_20degree', anomaly_composite)

plt.imshow(anomaly_composite.mean(axis=0))
plt.show()