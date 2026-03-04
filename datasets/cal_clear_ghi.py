import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np, glob, os
from datetime import datetime, timedelta
import pandas as pd


# ===== 晴空GHI计算函数 =====
def solar_position_spa_numpy(lat, lon, time_point):
    jd = time_point.to_julian_date()
    jc = (jd - 2451545.0) / 36525.0

    L0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)

    M_rad = np.radians(M)
    C = (1.914602 - jc * (0.004817 + 0.000014 * jc)) * np.sin(M_rad) \
        + (0.019993 - 0.000101 * jc) * np.sin(2 * M_rad) \
        + 0.000289 * np.sin(3 * M_rad)

    true_long = L0 + C
    omega = 125.04 - 1934.136 * jc
    lambda_sun = true_long - 0.00569 - 0.00478 * np.sin(np.radians(omega))

    epsilon0 = 23 + (26 + ((21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60)) / 60
    epsilon = epsilon0 + 0.00256 * np.cos(np.radians(omega))

    delta = np.degrees(np.arcsin(
        np.sin(np.radians(epsilon)) * np.sin(np.radians(lambda_sun))
    ))

    y = np.tan(np.radians(epsilon / 2)) ** 2
    Etime = 4 * np.degrees(
        y * np.sin(2 * np.radians(L0)) - 2 * e * np.sin(M_rad)
        + 4 * e * y * np.sin(M_rad) * np.cos(2 * np.radians(L0))
        - 0.5 * y ** 2 * np.sin(4 * np.radians(L0))
        - 1.25 * e ** 2 * np.sin(2 * M_rad)
    )

    mins = (time_point.hour * 60
            + time_point.minute
            + time_point.second / 60)

    true_solar_time = (mins + Etime + 4 * lon) % 1440
    ha = (true_solar_time / 4 - 180)

    lat_rad = np.radians(lat)
    ha_rad = np.radians(ha)
    delta_rad = np.radians(delta)

    cs = np.sin(lat_rad) * np.sin(delta_rad) + \
         np.cos(lat_rad) * np.cos(delta_rad) * np.cos(ha_rad)
    cs = np.clip(cs, -1, 1)

    zenith = np.degrees(np.arccos(cs))
    return zenith


def compute_clearsky_ineichen_np(lat, lon, altitude, time_point, TL=3.0):
    zenith = solar_position_spa_numpy(lat, lon, time_point)
    zenith_rad = np.radians(zenith)
    cos_z = np.cos(zenith_rad)
    cos_z = np.where(cos_z < 0, 0, cos_z)

    AM = 1.0 / (np.cos(zenith_rad) + 0.15 / (93.885 - zenith) ** 1.253)
    AM = np.where(cos_z <= 0, np.inf, AM)

    doy = time_point.day_of_year
    I0 = 1367.0 * (1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.0))

    cg1 = 5.09e-5 * altitude + 0.868
    cg2 = 3.92e-5 * altitude + 0.0387
    fh1 = np.exp(-altitude / 8000.0)
    fh2 = np.exp(-altitude / 1250.0)

    ghi_clear = (cg1 * I0 * cos_z *
                 np.exp(-cg2 * AM * (fh1 + fh2 * (TL - 1.0))) *
                 np.exp(0.01 * AM ** 1.8))
    ghi_clear = np.where(cos_z <= 0, 0, ghi_clear)
    return ghi_clear


