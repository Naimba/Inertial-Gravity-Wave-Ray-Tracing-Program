def IGWRT(zwn, mwn, lon_list, lat_list, p_list,
               freq, dt, t_start, tstep, ttotal, xcyclic,
               read_dtype, cal_dtype, inputfile, wrfile, bffile):
    from constants import hour, day
    from wr_IGW import WR
    import os
    # from netCDF4 import Dataset
        
    wr1 = WR(
        zwn, mwn,
        lon_list, lat_list, p_list, 
        dt,
        tstep * hour,
        ttotal * day,
        freq,
        t_start = t_start,
        read_dtype=read_dtype,
        cal_dtype=cal_dtype,
        inputfile=inputfile)
    
    if os.path.exists(bffile):
        print(f"Read from bffile: {bffile}")
        wr1.read_bffile(bffile)
        print(f"Read bffile: {bffile} successfully!")
    else: 
        print(f"bffile: {bffile} not exits, now calculating...")
        wr1.bf.loadbf_ncfile(inputfile)
        wr1.bf.ready(xcyclic=xcyclic)
        wr1.bf.output(bffile)
        print(f"bffile: {bffile} calculates successfully!")
    wr1.set_source_array(lon_list, lat_list)
    # wr1.set_source_matrix(SW_lon, SW_lat, dlon, dlat, nnx, nny)
    wr1.ray_info()
    wr1.ray_run()
    wr1.output(wrfile)    
    
if __name__ == '__main__':
    import numpy as np
    
    # import warnings
    # warnings.filterwarnings("ignore")
    
    parameters = {
        'zwn': np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]), 'mwn': np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
        # 'zwn': np.array([-5,5]), 'mwn': np.array([-5,5]),
        'lon_list': np.array([180]), # zonal wave sources
        'lat_list': np.array([0]), # Meridional wave sources
        # 'lon_list': np.array([110,115,120,125,130]), # zonal wave sources
        # 'lat_list': np.array([-15,-10,-5]), # Meridional wave sources
        'p_list': np.array([500]), # source pressure (hPa)
        'freq': 2 * np.pi / 5 / 86400, # 波动的起始频率
        'dt': 86400*6, # 速度场数据的时间分辨率，units: seconds
        't_start': 0,# 起始积分的时间点
        'tstep': 10/60, # 积分的时间间隔 unit: hour
        'ttotal': 5., # 积分的总时长 unit: day
        'xcyclic': True,
        'read_dtype': 'float64', # dtype read from nc
        'cal_dtype':'float64', # Only support float64
        'inputfile' : 'input_theory_theta.nc', #  background field
        'bffile'  : 'bf_theory_theta.nc', # output background field file
        'wrfile'  : 'wr_theory_theta.nc', # output wave ray ncfile
        # 'input_theta' : 'input_air.nc', #  background field
        # 'bffile'  : 'bf_air.nc', # output background field file
        # 'wrfile'  : 'wr_air.nc', # output wave ray ncfile
        }
    IGWRT(**parameters)