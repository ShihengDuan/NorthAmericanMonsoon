import numpy as np
import pandas as pd
import xarray as xa
from tqdm import tqdm

from scipy.stats import gamma, genpareto, norm
from scipy import optimize
import scipy.stats as st

def genpareto_objective_fn(theta, x, dis_func):
    scale = theta[-1]
    loc = 1 # starts from 1
    x = (np.asarray(x)-loc) / scale
    a = theta[0]
    args = (a, scale)
    pdf = dis_func.pdf(x, a)/scale
    logpdf = np.log(pdf)
    return -np.sum(logpdf, axis=0)

def genpareto_nnlf_and_penalty(x, args, dis_func):
    a = args[0]
    scale = args[-1]
    cond0 = (dis_func.a<=x)&(x<=dis_func.b)
    n_bad = np.count_nonzero(cond0, axis=0)
    pdf = dis_func.pdf(x, a)/scale
    logpdf = np.log(pdf)
    logpdf = dis_func.logpdf(x, a, loc=0, scale=scale)
    finite_logpdf = np.isfinite(logpdf)
    # print(np.sum(finite_logpdf))
    n_bad += np.sum(~finite_logpdf, axis=0)
    # print(np.sum(pdf))
    return -np.sum(logpdf, axis=0) # negative log liklihood

def genpareto_penalized_nnlf(theta, x, dis_func):
    args = tuple(theta[:-1])
    a = theta[0]
    scale = theta[-1]
    # print(args, scale)
    loc = 1
    x = (np.asarray(x)-loc) / scale
    n_log_scale = len(x) * np.log(scale)
    shape = args[0]
    if a<0:
        shape_penalty = 1000
    else:
        shape_penalty = 0
    args = (a, scale)
    return genpareto_nnlf_and_penalty(x, args, dis_func) + n_log_scale + shape_penalty

def fit_genpareto_parameters(data, dis_func):
    function = genpareto_penalized_nnlf
    start = dis_func._fitstart(data)
    # print(start)
    args = optimize.fmin(func=function, x0=(0, start[-1]), args=(np.ravel(data), dis_func), disp=0, maxiter=500)
    # print(args)
    return args

def gamma_objective_fn(theta, x, dis_func):
    args = tuple(theta[:-1])
    a = args[0]
    scale = theta[-1]
    loc = 0 # starts from 0
    x = (np.asarray(x)-loc) / scale
    n_log_scale = len(x) * np.log(scale)
    shape = args[0]
    args = (a, scale)
    return gamma_nnlf_and_penalty(x, args, dis_func)

def gamma_nnlf_and_penalty(x, args, dis_func):
    a = args[0]
    scale = args[-1]
    cond0 = (dis_func.a<=x)&(x<=dis_func.b)
    n_bad = np.count_nonzero(cond0, axis=0)
    pdf = dis_func.pdf(x, a, scale=1)/scale
    total_pdf = 1 - dis_func.cdf(1, a, loc=0, scale=scale) # from 1 to inf
    pdf = pdf/total_pdf
    logpdf = np.log(pdf)
    finite_logpdf = np.isfinite(logpdf)
    n_bad += np.sum(~finite_logpdf, axis=0)
    # print(np.sum(pdf))
    return -np.sum(logpdf, axis=0) # negative log liklihood

def gamma_penalized_nnlf(theta, x, dis_func):
    args = tuple(theta[:-1])
    a = theta[0]
    scale = theta[-1]
    # print(args, scale)
    loc = 0
    x = (np.asarray(x)-loc) / scale
    n_log_scale = len(x) * np.log(scale)
    shape = args[0]
    if a>1:
        shape_penalty = 1000
    else:
        shape_penalty = 0
    args = (a, scale)
    return gamma_nnlf_and_penalty(x, args, dis_func) + n_log_scale + shape_penalty

def fit_gamma_parameters(data, dis_func):
    function = gamma_penalized_nnlf
    start = dis_func._fitstart(data)
    # print(start)
    args = optimize.fmin(func=function, x0=(0, start[-1]), args=(np.ravel(data), dis_func), disp=0, maxiter=500)
    # print(args)
    return args
def fit_gamma_genpareto(data, ax, x):
    a1, scale1 = fit_gamma_parameters(data, gamma)
    pdf1 = gamma.pdf(x, a1, 0, scale1)/(1-gamma.cdf(1, a1, 0, scale1))
    score_ga = gamma_objective_fn(x=data, theta=(a1, scale1), dis_func=gamma)
    # print('Gamma: ', score_ga)
    if ax is not None:
        ax.plot(x, pdf1, color='red')
    a2, scale2 = fit_genpareto_parameters(data, genpareto)
    pdf2 = genpareto.pdf(x, a2, 1, scale2)
    score_gp = genpareto_objective_fn(x=data, theta=(a2, scale2), dis_func=genpareto)
    # print('GenPareto: ', score_gp)
    if ax is not None:
        ax.plot(x, pdf2, color='blue', label='GenPareto')
    return (a1, scale1), (a2, scale2), (pdf1, pdf2), (score_ga, score_gp)
def get_pdf(precip, x):
    (a1, scale1), (a2, scale2), pdf_max, (score_ga, score_gp) = fit_gamma_genpareto(precip, None, x)
    if score_ga<score_gp:
        pdf = gamma.pdf(x, a1, 0, scale1)/(1-gamma.cdf(1, a1, 0, scale1))
        dis = 'Gamma'
        print(a1, scale1, dis)
    else:
        pdf = genpareto.pdf(x, a2, 1, scale2)
        dis = 'GenPareto'
        print(a2, scale2, dis)
    return pdf




class TUTT:
    def __init__(self, num_points):
        self.num_points = num_points
        self.lons = []
        self.lats = []
        self.time = []

    def add_track(self, lon, lat, dtime):
        self.lons.append(lon)
        self.lats.append(lat)
        self.time.append(dtime)


class Surge:
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time


def load_TUTT():
    tutt_record = np.load("/tempest/duan0000/exprecip/cpc-global/tutt_record.npy")
    print('TUTT_RECORD SHAPE: ', tutt_record.shape)
    df = pd.DataFrame(tutt_record, columns=["track_id", "year", "month", "day", "hour", "i", "j", "lon", "lat"], )
    df = df.astype({"track_id": "int32", "year": "int32", "month": "int32", "day": "int32", "hour": "int32"})
    df["track+year"] = df.apply(lambda row: str(int(row.year)) + "_" + str(int(row.track_id)), axis=1)
    unique_id = df["track+year"]
    print('UNIQUE UNIQUE_ID: ', np.unique(unique_id, return_counts=True))
    unique_id = np.unique(unique_id)

    tutts = []
    for u_id in tqdm(unique_id[:]):
        record = df[df["track+year"] == u_id]
        num_points = record.shape[0]
        lons = record.lon - 360
        lats = record.lat
        years = np.array(record.year)
        months = np.array(record.month)
        days = np.array(record.day)
        hours = np.array(record.hour)
        dates = [
            np.datetime64(str(year) + "-" + str(month).zfill(2) + "-" + str(day).zfill(2) + "T" + str(hour).zfill(2))
            for year, month, day, hour in zip(years, months, days, hours)]
        tutt = TUTT(num_points=num_points)
        tutt.add_track(lon=lons, lat=lats, dtime=dates)
        tutts.append(tutt)
    return tutts

def tutt_PAU(file):
    total_record = []
    with open(file, "r") as tracks:
        total_N = -1
        lons = []
        lats = []
        times = []
        for line in tracks:
            items = line.split("\t")
            if items[0] == "start":
                if total_N >= 0:
                    tutt = TUTT(num_points=0)
                    tutt.add_track(lons, lats, times)
                    total_record.append(tutt)
                    lons = []
                    lats = []
                    times = []
                else:
                    total_N += 1
                    lons = []
                    lats = []
                    times = []
                    continue
            else:
                lon = float(items[3])
                if lon > 0:
                    lon = lon - 360
                lons.append(lon)
                lats.append(float(items[4]))
                year = items[5]
                month = items[6].zfill(2)
                date = items[7].zfill(2)
                hour = items[8].split("\n")[0].zfill(2)
                time_string = year + "-" + month + "-" + date + "T" + hour
                time = np.datetime64(time_string)
                times.append(time)
    tutt = TUTT(num_points=0)
    tutt.add_track(lons, lats, times)
    total_record.append(tutt)
    return total_record, total_N


def load_Surge():
    surge_data_6h = "/tempest/duan0000/exprecip/nam_ivt/MoistureSurge-6h.nc"
    surge_data_6h = xa.open_dataset(surge_data_6h)
    print('SURGE: ', surge_data_6h)

    surge_record_6h = []
    for i in range(529):
        surge = Surge(start_time=surge_data_6h.start_time.data[i], end_time=surge_data_6h.end_time.data[i])
        surge_record_6h.append(surge)
    return surge_record_6h


def feature_induced(sub_id, tc_induced_flag, surge_6h_induced_flag, tutt_induced_flag):
    TC_induced = np.where(tc_induced_flag >= 1)[0]
    surge_induced = np.where(surge_6h_induced_flag >= 1)[0]
    tutt_induced = np.where(tutt_induced_flag >= 1)[0]
    total_set = np.arange(len(ext_days[sub_id]))
    TC_only = set(TC_induced) - set(surge_induced) - set(tutt_induced)
    tutt_only = set(tutt_induced) - set(surge_induced) - set(TC_induced)
    surge_only = set(surge_induced) - set(tutt_induced) - set(TC_induced)
    TC_Surge = set(TC_induced) & set(surge_induced) - set(tutt_induced)
    TC_Tutt = set(TC_induced) & set(tutt_induced) - set(surge_induced)
    Tutt_Surge = set(tutt_induced) & set(surge_induced) - set(TC_induced)
    TC_Surge_Tutt = set(TC_induced) & set(tutt_induced) & set(surge_induced)
    return TC_only, tutt_only, surge_only, TC_Surge, TC_Tutt, Tutt_Surge, TC_Surge_Tutt
