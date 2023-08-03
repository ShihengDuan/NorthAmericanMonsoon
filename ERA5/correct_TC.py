import numpy as np
import pickle
import xarray as xa

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


def start_end_time(sub_id):
    start = []
    end = []
    ext_times = ext_days[sub_id]
    length = len(ext_times)
    i = 0
    while i <= length - 1:
        j = 1
        while (j + i <= length - 1) and ((ext_times[i + j] - ext_times[i + j - 1]) / np.timedelta64(1, "D") == 1):
            j += 1
        start.append(ext_times[i] - np.timedelta64(12, "h"))
        end.append(ext_times[i + j - 1] + np.timedelta64(12, "h"))
        i += j
    return start, end

for sub_id in range(1, 8):
    start, end = start_end_time(sub_id=sub_id)
    tc_induced = np.load("PAU/" + str(sub_id) + "_tc_induced_flag.npy")
    with open(str(sub_id)+'_tc_types.pickle', 'rb') as pfile:
        tc_type = pickle.load(pfile)
    for i, t in enumerate(end):
        if tc_induced[i]==1:
            l = len(tc_type[t])
            A = set()
            for e in tc_type[t]:
                A = A.union(e)
            if A.intersection(set([b'TS', b'TY', b'ST', b'TC', b'HU', b'HR']))==set():
                print('Not desired: ', t, ' tc_induced: ', tc_induced[i])
                tc_induced[i]=0
    np.save("PAU/" + str(sub_id) + "_tc_corrected_flag.npy", tc_induced)
