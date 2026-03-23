def IGWRT(Lx, Ly, lon_list, lat_list, p_list,
               freq, dt, t_start, tstep, ttotal, xcyclic,
               read_dtype, cal_dtype, inputfile, wrfile):
    from wr_IGW import WR
    # import os
    # from netCDF4 import Dataset
    
    print('Initializing the array of IGWRT WR class ...')
    wr1 = WR(Lx, Ly,
        lon_list, lat_list, p_list, 
        dt,
        tstep, ttotal,
        freq,
        t_start = t_start,
        read_dtype = read_dtype,
        cal_dtype = cal_dtype,
        inputfile = inputfile)
    print(f'Reading data from: {inputfile} ...')
    
    wr1.bf.loadbf_ncfile(inputfile)
    wr1.bf.ready(xcyclic=xcyclic)

    wr1.set_source_array(lon_list, lat_list)
    wr1.ray_info()
    wr1.ray_run()
    wr1.output(wrfile)    
    
if __name__ == '__main__':
    import numpy as np
    from constants import hour, km
    import time

    # import warnings
    # warnings.filterwarnings("ignore")
    
    start = time.perf_counter()

    parameters = {
        'Lx' : np.concatenate((np.arange(-1000, -100+0.01, 100), np.arange(100, 1000+0.01, 100))) * km, # The initial zonal wavelength (meter)
        'Ly' : np.concatenate((np.arange(-1000, -100+0.01, 100), np.arange(100, 1000+0.01, 100))) * km, # The initial meridional wavelength (meter)
        'lon_list': np.arange(115, 130.0001, 0.5), # Zonal wave sources (degree)
        'lat_list': np.arange(-22, -15.0001, 0.5), # Meridional wave sources (degree)
        'p_list': np.array([500, 850]), # Vertical source (hPa)
        
        'freq': 2 * np.pi /( 8 * hour), # The initial frequency of waves (rad/s)
        'dt': 1 * hour, # The time resolution of backgroud data (second)
        't_start': 3600,# The starting point of integration (second)
        'tstep': 1/60 * hour, # The time interval of integration (second)
        'ttotal': 10 * hour, # The total time of integration (second)
        
        'xcyclic': True,
        'read_dtype': 'float32', # dtype read from nc
        'cal_dtype':'float64', # Only support float64
        
        'inputfile' : '/home/liuyifan/myPython/IGWRT/data/input_airuv_2019_025.nc', #  background field
        'wrfile'    : '/home/liuyifan/myPython/IGWRT/data/wr_2019_new_numba_wl.nc', # output wave ray ncfile
        }
    IGWRT(**parameters)
    
    end = time.perf_counter()
    print(f"Running time: {end-start:.3f} s")