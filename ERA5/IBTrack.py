# Subset of IBTrack record.
# storm dimension: 13547
import numpy as np
import pandas as pd
import xarray as xa
from tqdm import tqdm


def distance(x, y, a, b):
    dis = np.sqrt(np.square(x - a) + np.square(y - b))
    if dis <= 50:
        return True
    else:
        return False


min_lons = []
max_lons = []
min_lats = []
max_lats = []
for sub_id in range(1, 8):
    lons_sub = np.load('Calculations/' + str(sub_id) + '_lons.npy')
    lats_sub = np.load('Calculations/' + str(sub_id) + '_lats.npy')
    min_lons.append(np.min(lons_sub))
    max_lons.append(np.max(lons_sub))
    min_lats.append(np.min(lats_sub))
    max_lats.append(np.max(lats_sub))
min_lat = np.min(min_lats) - 10
min_lon = np.min(min_lons) - 10
max_lat = np.max(max_lats) + 10
max_lon = np.max(max_lons) + 10
print(min_lat, max_lat)
print(min_lon, max_lon)

file = '../IBTrACS.ALL.v04r00.nc'
record = xa.open_dataset(file)
candidates = []
for i in tqdm(range(13547)):
    time = record.time.isel(storm=i).data
    start_time = time[0]
    start_year = pd.to_datetime(start_time).year
    # print(start_year)
    if start_year >= 1979 and start_year <= 2019:
        lat = record.lat.isel(storm=i).data
        lon = record.lon.isel(storm=i).data
        lon = np.where(lon > 180, lon - 360, lon)
        # print(np.nanmin(lon), ' ', np.nanmax(lon))
        flag = 0
        for x, y in zip(lon, lat):
            if distance(x, y, min_lon, min_lat) or distance(x, y, max_lon, min_lat) \
                    or distance(x, y, min_lon, max_lat) or distance(x, y, max_lon, max_lat):
                flag = 1
                break
        if flag == 1:
            candidates.append(i)
print(len(candidates))
print(10843 in candidates)
sub_record = record.isel(storm=candidates)
print(sub_record)
sub_record.to_netcdf('sub_IBTrack.nc')
