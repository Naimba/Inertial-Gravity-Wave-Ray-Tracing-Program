import numpy as np
from constants import day, hour, rad2deg, rearth, undef, deg2rad, pi, grav, H0, p0
from bf_IGW import BF
from bf_IGW import cal_kz
from netCDF4 import Dataset
import numba as nb
import sys

'''
奥卡姆剃刀原理：如无必要，勿增实体
Occam's Razor: Entities should not be multiplied unnecessarily
Numquam ponenda est pluralitas sine necessitate
'''

# nb_para_dic = {
#     'nopython': True,
#     # 'fastmath': True,
#     'cache': True,
# }


def progress_bar(current, total, bar_length=50):
    percent = float(current) / total
    arrow = '=' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write(
        f"\rprocess: [{arrow + spaces}] {int(round(percent * 100))}%")
    sys.stdout.flush()


# signature1 = r'Tuple((' + r'f8[:,:,:],' * 8 + r'))' + \
#     r'(f8[:],' + r'f8[:,:,:],' * 20 + r')'  
# @nb.jit([signature1], **nb_para_dic)
@nb.jit(nopython=True, cache=False, fastmath=False)
def core_diffun(freq_intrinsic, kx, ky, kz, ln_theta_xx, ln_theta_xy, ln_theta_xz, ln_theta_xt, 
                ln_theta_yx, ln_theta_yy, ln_theta_yz, ln_theta_yt, ln_theta_zz, ln_theta_zt,
                u_x, v_x, u_y, v_y, u_z, v_z, u_t, v_t, ug, vg, wg, lat):
                # ln_theta_yx, ln_theta_yy, ln_theta_yz, ln_theta_yt, ln_theta_zx, ln_theta_zy, ln_theta_zz, ln_theta_zt, ug, vg, wg, lat):
    # K_2^2 and K_3^2
    kk_2 = kx**2 + ky**2
    kk_3 = kk_2  + kz**2
    
    # print('ln_theta_zz = ', ln_theta_zz.reshape(-1))
    # print('u_x = ', u_x.reshape(-1))
    # print('v_x = ', v_x.reshape(-1))
    # print('rearth = ', rearth)
    # print('kx = ', kx.reshape(-1))
    # print('grav = ', grav)
    # print('kk_2 = ', kk_2.reshape(-1))
    # print('ln_theta_xz = ', ln_theta_xz.reshape(-1))
    # print('kz = ', kz.reshape(-1))
    # print('ln_theta_xx = ', ln_theta_xx.reshape(-1))
    # print('ky = ', ky.reshape(-1))
    # print('ln_theta_yx = ', ln_theta_yx.reshape(-1))
    # print('freq_inrinsic = ', freq_intrinsic.reshape(-1))
    # print('kk_3 = ', kk_3.reshape(-1))

    # calculate dkx/dt, dky/dt and dkz/dt
    dzwn = -(u_x + v_x) / rearth * kx - ( grav * kk_2 * ln_theta_xz - kz * grav * (kx * ln_theta_xx + ky * ln_theta_yx) / rearth ) / (2 * freq_intrinsic * kk_3)
    dmwn = -(u_y + v_y) / rearth * ky - ( grav * kk_2 * ln_theta_yz - kz * grav * (kx * ln_theta_xy + ky * ln_theta_yy) / rearth ) / (2 * freq_intrinsic * kk_3)
    dvwn = -(u_z + v_z) * kz - ( rearth * grav * kk_2 * ln_theta_zz - kz * grav * (kx * ln_theta_xz + ky * ln_theta_yz) ) / (2 * freq_intrinsic * kk_3)
    
    # print('dzwn = ', dzwn.reshape(-1))
    # print('dmwn = ', dmwn.reshape(-1))
    # print('dvwn = ', dvwn.reshape(-1))
    
    # calculate domega/dt
    # print('u_t = ', u_t.reshape(-1))
    # print('v_t = ', v_t.reshape(-1))
    # print('ln_theta_xt = ', ln_theta_xt.reshape(-1))
    # print('ln_theta_yt = ', ln_theta_yt.reshape(-1))
    # print('ln_theta_zt = ', ln_theta_zt.reshape(-1))
    dfreq = u_t + v_t + ( grav * kk_2 * ln_theta_zt - kz * grav * (kx * ln_theta_xt + ky * ln_theta_yt) / rearth ) / (2 * freq_intrinsic * kk_3)
    # 形成导数数组 (注意经度、纬度、波数等以地球半径归一化)
    dlon = ug / rearth / np.cos(lat)
    dlat = vg / rearth
    dz = wg
    dkx = dzwn
    dky = dmwn
    dkz = dvwn

    # 额外将 ug, vg 作为状态变量导数存储 (按原Fortran实现，也除以Rearth)
    dug = ug
    dvg = vg
    dwg = wg
    
    return dlon, dlat, dz, dkx, dky, dkz, dug, dvg, dwg, dfreq


# sign1 = 'f8[:,:,:,:,:](f8[:,:,:,:,:],f8[:,:,:,:,:],f8[:,:,:,:,:],f8[:,:,:,:,:],f8[:,:,:,:,:],f8[:])'
# @nb.jit([sign1], **nb_para_dic)
@nb.jit(nopython=True, cache=False, fastmath=False)
def core_rk4_step(y, k1, k2, k3, k4, dt):
    temp = y
    ks = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    temp[0:6] = y[0:6] + ks[0:6]
    temp[6:9] = ks[6:9] / dt
    temp[-1] = y[-1] + ks[-1]
    # print('temp.shape = ', temp.shape)
    return temp

# @nb.jit('f8[:,:,:](f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:])', **nb_para_dic)
@nb.jit(nopython=True, cache=False, fastmath=False)
def cal_dis(lon_curr,lat_curr,lon_prev,lat_prev):

    # 计算经纬度差值 (弧度)
    dlon = lon_curr - lon_prev # 对应 Haversine 公式中的 (lambda2 - lambda1)
    dlat = lat_curr - lat_prev # 对应 Haversine 公式中的 (phi2 - phi1)

    # Haversine 公式计算步骤
    # 计算 a
    a = np.sin(dlat / 2.0)**2 + np.cos(lat_prev) * np.cos(lat_curr) * np.sin(dlon / 2.0)**2

    # 计算 c (角距离)
    # 使用 np.arctan2 比 np.arccos 在数值上更稳定，尤其当两点非常接近时
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return np.abs(c)

# sign2 = 'Tuple((f8[:,:,:],f8[:,:,:],f8[:,:,:]))(f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:])'
# @nb.jit([sign2], **nb_para_dic)
@nb.jit(nopython=True, cache=False, fastmath=False)
def cal_group_velocity_extent(u, v, ln_theta_x, ln_theta_y, ln_theta_z, f, freq_intrinsic, kx, ky, kz):
    kk_3 = kx * kx + ky * ky + kz * kz

    # print('kx.shape = ',kx.shape)
    
    ug = u - (2 * rearth * freq_intrinsic * freq_intrinsic * kx - 2 * rearth * kx * grav * ln_theta_z + kz * grav * ln_theta_x) / (2 * freq_intrinsic * kk_3)
    vg = v - (2 * rearth * freq_intrinsic * freq_intrinsic * ky - 2 * rearth * ky * grav * ln_theta_z + kz * grav * ln_theta_y) / (2 * freq_intrinsic * kk_3)
    
    # print('ug = ', ug.reshape(-1))
    # print('vg = ', vg.reshape(-1))
    
    wg = - ( 2 * rearth * kz * (freq_intrinsic**2 - f**2) + kx * grav * ln_theta_x + ky * grav * ln_theta_y)/ (2 * freq_intrinsic * kk_3)
    # print('分子 = ', 2 * rearth * kz * (freq_intrinsic**2 - f**2) + kx * grav * ln_theta_x + ky * grav * ln_theta_y)
    # print('分母 = ', 2 * freq_intrinsic * kk_3)
    # print('wg = ', wg)
    # print('ug.shape = ',ug.shape)
    # print('wg.shape = ',wg.shape)
    return ug, vg, wg

class WR:
    """
    波射线追踪类:
    属性:
      Lx, Ly      - zonal and meridional wavelength array (length: nzwn and nmwn)
      lon_list, lat_list - 波源水平经纬度 (单位度)
      p_list      - 垂向波源所在气压层 (单位: hPa)
      dt         - 输入数据的时间间隔
      tstep      - 积分时间步长 (秒)
      ttotal     - 总积分时间 (秒)
      freq       - 初始波频率
      cal_dtype, read_dtype - 计算精度和读取精度
      cut_off - 防止速度过快进行的阶段处理
      t_start - 积分的时间起点
      inputfile - 输入的基本场文件，里面需要有温度（或位温）数据和速度数据
    """
    def __init__(self, Lx, Ly, lon_list, lat_list, p_list, dt, tstep=1. * hour, ttotal=20. * day,
                 freq=0, cal_dtype='float64', read_dtype='float32',
                 cut_off=0.1, t_start=0, inputfile=None):
        # Initialize the basic background field
        self.all_dtype = cal_dtype
        if inputfile is None:
            raise ValueError('inputfile is need')
        else:
            nx, ny, nz, nt = self.get_nxyzt(inputfile)
        self.bf = BF(nx, ny, nz, nt, dt, read_dtype=read_dtype, cal_dtype=cal_dtype)

        # 赋已给诸值
        nLx = len(Lx)
        nLy = len(Ly)
        self.nLx = nLx
        self.nLy = nLy
        if cal_dtype == 'float32':
            self.tstep = np.float32(tstep)
            self.freq = np.float32(freq)
            self.ttotal = np.float32(ttotal)
        elif cal_dtype == 'float64':
            self.tstep = np.float64(tstep)
            self.freq = np.float64(freq)
            self.ttotal = np.float64(ttotal)
        else:
            self.tstep = np.float64(tstep)
            self.freq = np.float64(freq)
            self.ttotal = np.float64(ttotal)

        # 分配数组
        self.Lx = np.array(Lx, dtype=self.all_dtype)
        self.Ly = np.array(Ly, dtype=self.all_dtype)
        self.p_list =  np.array(p_list, dtype=self.all_dtype)

        # 无量纲化波数
        self.zwn = 2 * pi / self.Lx * rearth
        self.mwn = 2 * pi / self.Ly * rearth
        
        nnx = len(lon_list)
        nny = len(lat_list)
        nsource = nnx * nny * len(p_list)
        self.nsource = nsource
        
        # 分配数组
        self.source_lon = np.zeros(nsource, dtype=self.all_dtype)
        self.source_lat = np.zeros(nsource, dtype=self.all_dtype)
        self.source_p = np.zeros(nsource, dtype=self.all_dtype)
        self.source_z = np.zeros(nsource, dtype=self.all_dtype)
        # 计算步数 nnt
        self.nnt = int(self.ttotal / self.tstep) + 1
        self.t_start = t_start
        # 分配射线跟踪数组: 第三维大小为2，对应每初始波数可能存在的2条射线 (2个m根)
        shape = (self.nnt, 2, nsource, nLy, nLx)
        self.rfreq = np.full(shape, undef, dtype=self.all_dtype)
        self.rlon = np.full(shape, undef, dtype=self.all_dtype)
        self.rlat = np.full(shape, undef, dtype=self.all_dtype)
        self.rz = np.full(shape, undef, dtype=self.all_dtype)
        self.rzwn = np.full(shape, undef, dtype=self.all_dtype)
        self.rmwn = np.full(shape, undef, dtype=self.all_dtype)
        self.rvwn = np.full(shape, undef, dtype=self.all_dtype)
        self.rug = np.full(shape, undef, dtype=self.all_dtype)
        self.rvg = np.full(shape, undef, dtype=self.all_dtype)
        self.rwg = np.full(shape, undef, dtype=self.all_dtype)
        self.cut_off = cut_off  * self.tstep / 3600.

    def get_nxyzt(self, inputfile):
        '''
        通过读取nc文件确定网格数
        如果文件中存在lon和lat，使用它们的维度
        如果不存在lon或lat，读取位温或温度的维度，-1维为lon，-2维为lat，并给出warning
        '''
        import netCDF4 as nc
        ds = nc.Dataset(inputfile)
        
        if 'theta' in ds.variables:
            temp = np.array(ds.variables['theta'][:], dtype=self.all_dtype)
        elif 'T' in ds.variables:
            temp = np.array(ds.variables['T'][:], dtype=self.all_dtype)
        elif 'air' in ds.variables:
            temp = np.array(ds.variables['air'][:], dtype=self.all_dtype)
        nt = temp.shape[0]
        
        # 自动识别纬度、经度变量
        lat_candidates = ['lat', 'latitude', 'Lat', 'Latitude']
        lon_candidates = ['lon', 'longitude', 'Lon', 'Longitude']
        p_candidates = ['p', 'pressure', 'P', 'Pressure', 'level', 'levels']

        p_data = None
        lat_data = None
        lon_data = None
        

        for name in lat_candidates:
            if name in ds.variables:
                lat_data = np.array(
                    ds.variables[name][:],
                    dtype=self.all_dtype)
                break
        
        for name in lon_candidates:
            if name in ds.variables:
                lon_data = np.array(
                    ds.variables[name][:],
                    dtype=self.all_dtype)
                break
        
        for name in p_candidates:
            if name in ds.variables:
                p_data = np.array(
                    ds.variables[name][:],
                    dtype=self.all_dtype)
                break
            
        message = ''
        if (lat_data is None) or (lon_data is None):
            nz, ny, nx = temp.shape[-3], temp.shape[-2], temp.shape[-1]
            message = '!!!Attention: Using temp.shape[-2] and temp.shape[-1] as nlat and nlon!!!'
            print(message)
        else:
            nz, ny, nx = p_data.shape[0], lat_data.shape[0], lon_data.shape[0]
        ds.close()
        return nx, ny, nz, nt
    
    def read_bffile(self, bffile):
        """从bffile读取bf模块的变量"""
        ds = Dataset(bffile, 'r')
        self.bf.lon = ds.variables['lon'][:]
        self.bf.lat = ds.variables['lat'][:]
        self.bf.p = ds.variables['level'][:]
        self.bf.z = -H0 * np.log(self.bf.p/p0)
        self.bf.t = ds.variables['t'][:]
        self.bf.fields = ds.variables['fields'][:]
        ds.close()
        
    def check_time_broad(self):
        """检查积分时间是否越界"""
        end_integration_time = self.t_start + self.ttotal
        end_time = self.bf.nt * self.bf.dt
        if end_integration_time > end_time:
            sys.exit('ERROR: The end time of integration is greater than input data time! Please check')
                
    # def set_zwn_and_mwn(self, zwn_array, mwn_array):
    #     """Set initial zonal and meridional wavenumber array"""
    #     self.zwn[:] = np.array(zwn_array, dtype=self.all_dtype)
    #     self.mwn[:] = np.array(mwn_array, dtype=self.all_dtype)
        


    def set_source_array(self, lon_list, lat_list):
        """
        直接设置源点数组 (lon_list, lat_list)。
        lon/lat 单位为度，将转换为弧度后存储。
        替代了set_source_matrix函数
        """
        
        if np.any(np.array(lat_list) > 89.0) or np.any(np.array(lat_list) < -89.0):
            raise ValueError("source latitude out of -90~90 range!")
        
        p_list = self.p_list
        nnx = len(lon_list)
        nny = len(lat_list)
        nnz = len(p_list)
        
        
        # 填充源点数据
        idx = 0
        for iy in range(nny):
            for ix in range(nnx):
                # 将二维索引转换为一维
                lon_deg = lon_list[ix]
                lat_deg = lat_list[iy]
                
                for iz in range(nnz):
                    # 规范化经度到 [0, 360)
                    lon_deg = lon_deg % 360.0
                    
                    self.source_lon[idx] = lon_deg * deg2rad
                    self.source_lat[idx] = lat_deg * deg2rad
                    self.source_p[idx] = p_list[iz]
                    self.source_z[idx] = -H0 * np.log(p_list[iz] / p0)
                    idx += 1

    def ray_info(self):
        """输出波射线追踪的初始信息"""
        print(
            "==============================================================================")
        print("IGWRT Package: Inertial Gravity Wave Ray Tracing Information ")
        print(
            f" Basic State Grid (nlon x nlat x np x nt): {self.bf.nx} x {self.bf.ny} x {self.bf.nz} x {self.bf.nt}")
        
        print(f" Initial Period: {2 * np.pi / self.freq/ hour:.2f} h")
        
        print(f" Initial Zonal Wavelength (km) (total {self.nLx} wsavelength):")
        print(" " * 15 + " ".join(f"{z:.1f}" for z in self.Lx/1000))
        print(f" Initial Meridional Wavelength (km) (total {self.nLy} wavelength):")
        print(" " * 15 + " ".join(f"{z:.1f}" for z in self.Ly/1000))
        
        print(f" Source Locations (total {self.nsource} points):")
        # 打印每个源点经纬度（度）
        for i in range(self.nsource):
            lon_deg = self.source_lon[i] * rad2deg
            lat_deg = self.source_lat[i] * rad2deg
            source_p = self.source_p[i]
            print(" " * 15 + f"{lon_deg:7.2f}, {lat_deg:7.2f}, {source_p:7.2f} hPa")
        print(f" Time Step (s): {self.tstep:.2f}")
        print(f" Total Integration Time (hour): {self.ttotal/hour:.2f}")
        print(f" Total Steps (nnt): {self.nnt}")
        print(
            "==============================================================================")
        
        
    def cal_group_velocity(self, u, v, ln_theta_x, ln_theta_y, ln_theta_z, f, freq_intrinsic, kx, ky, kz):
        """
        计算群速 (ug, vg, wg)
        """
        
        # 这段不知道是干啥的，改编自Ms. Y. N. Yang 的Rossby波射线追踪程序
        # nans = np.einsum('ij,j->ij', kz * 0, ln_theta_x * 0) + 1
        # nans[np.isnan(nans)] = 0
        
        kk_3 = kx * kx + ky * ky + kz * kz
        
        
        ug = u[:, None] - (2 * rearth * freq_intrinsic * freq_intrinsic * kx - 2 * rearth * kx * grav * ln_theta_z[:, None] + kz * grav * ln_theta_x[:, None]) / (2 * freq_intrinsic * kk_3)
        vg = v[:, None] - (2 * rearth * freq_intrinsic * freq_intrinsic * ky - 2 * rearth * ky * grav * ln_theta_z[:, None] + kz * grav * ln_theta_y[:, None]) / (2 * freq_intrinsic * kk_3)
        wg = - (2 * rearth * kz * (freq_intrinsic * freq_intrinsic - f[:, None] * f[:, None]) + kx * grav * ln_theta_x[:, None] + ky * grav * ln_theta_y[:, None])/ (2 * freq_intrinsic * kk_3)

        
        return ug, vg, wg

    def ray_initial(self):
        """
        初始化射线的起始状态:
         - 将所有射线的初始位置设置为源点位置
         - 计算每个源点每个初始k and l 对应的垂向波数根 (最多2个)，赋初始 k, l, m 等
         - 计算初始群速度
        """
        # 为方便，将 initial 状态索引用 idx0 表示 (Python 0号索引对应 Fortran 初始状态第1步)
        idx0 = 0
        # 将所有波数和根的经纬初始位置设为源点
        self.rlon[idx0, :, :, :, :] = self.source_lon[None, None, :, None, None]
        self.rlat[idx0, :, :, :, :] = self.source_lat[None, None, :, None, None]
        self.rz[idx0, :, :, :, :] = self.source_z[None, None, :, None, None]
        # 插值基本场到源点位置
        lon0 = self.source_lon[:]
        lat0 = self.source_lat[:]
        z0 = self.source_z[:]
        t0 = self.t_start
        res = self.bf.cal_bf_point_initial(lon0, lat0, z0, t0)
        (ln_theta_x, ln_theta_y, ln_theta_z, u, v, f) = res
        # 设置局地 k (zwn) 初值
        self.rzwn[idx0, :, :, :, :] = self.zwn[None, None, None, None, :]
        self.rmwn[idx0, :, :, :, :] = self.mwn[None, None, None, :, None]
        for ik in range(self.nLx):
            for il in range(self.nLy):
                kx = self.zwn[ik]
                ky = self.mwn[il]
                
                # 计算垂向波数
                freq_intrinsic = self.freq - u * kx / rearth - v * ky / rearth
                # print('freq = ', self.freq)
                # print('u = ', u)
                # print('kx = ', kx)
                # print('v = ', v)
                # print('ky = ', ky)
                # print('freq_intrinsic = ', freq_intrinsic)
                # print('f = ', f)
                # print('ln_theta_x = ', ln_theta_x)
                # print('ln_theta_y = ', ln_theta_y)
                # print('ln_theta_z = ', ln_theta_z)
                m_list, rootnum = cal_kz(ln_theta_x, ln_theta_y, ln_theta_z, f, freq_intrinsic, kx, ky)
                m_val = np.transpose(m_list, (1, 0))
                # print('m_val = ', m_val)
                self.rvwn[idx0, :, :, il, ik] = m_val
                #设置初始频率
                self.rfreq[idx0, :, :, il, ik] = self.freq
                # 计算初始群速度并存储
                freq = self.freq
                ug0, vg0, wg0 = self.cal_group_velocity(u, v, ln_theta_x, ln_theta_y, ln_theta_z, f, freq, kx, ky, m_list)
                self.rug[idx0, :, :, il, ik] = np.transpose(ug0)
                self.rvg[idx0, :, :, il, ik] = np.transpose(vg0)
                self.rwg[idx0, :, :, il, ik] = np.transpose(wg0)
                
    def diffun(self, y, t):
        """
        计算状态变量 y 的导数 dy/dt，用于 Runge-Kutta 积分。
        y 为长度10的数组: [lon, lat, z*, kx, ky, kz, ug, vg, wg, freq]
        返回相同长度的导数组 dk (或在不继续积分时返回 error_num = 1)。
        """
        # print('y = ', y)
        lon, lat, z, kx, ky, kz, freq = y[0], y[1], y[2], y[3], y[4], y[5], y[-1]
        error_num = 0
        # print('z = ',z)
        if (abs(lat) >= 0.5 * pi).all():
            error_num = 1
        # 计算当前位置背景场变量 
        # print('')
        # print('lon = ', lon.reshape(-1))
        # print('lat = ', lat.reshape(-1))
        # print('z = ', z.reshape(-1))

        # print('')
        # print(z)
        res = self.bf.cal_bf_point(lon.reshape(-1), lat.reshape(-1), z.reshape(-1), t)
        
        (ln_theta_x, ln_theta_y, ln_theta_z, ln_theta_t,
         ln_theta_xx, ln_theta_xy, ln_theta_xz, ln_theta_xt,
         ln_theta_yx, ln_theta_yy, ln_theta_yz, ln_theta_yt,
         ln_theta_zz, ln_theta_zt, f, u, v, u_x, v_x, u_y, v_y,
         u_z, v_z, u_t, v_t) = res
                
        (ln_theta_x, ln_theta_y, ln_theta_z, ln_theta_t,
         ln_theta_xx, ln_theta_xy, ln_theta_xz, ln_theta_xt,
         ln_theta_yx, ln_theta_yy, ln_theta_yz, ln_theta_yt,
         ln_theta_zz, ln_theta_zt, f, u, v, u_x, v_x, u_y, v_y,
         u_z, v_z, u_t, v_t) = (ln_theta_x.reshape(lon.shape),
                                ln_theta_y.reshape(lon.shape),
                                ln_theta_z.reshape(lon.shape),
                                ln_theta_t.reshape(lon.shape),
                                ln_theta_xx.reshape(lon.shape),
                                ln_theta_xy.reshape(lon.shape),
                                ln_theta_xz.reshape(lon.shape),
                                ln_theta_xt.reshape(lon.shape),
                                ln_theta_yx.reshape(lon.shape),
                                ln_theta_yy.reshape(lon.shape),
                                ln_theta_yz.reshape(lon.shape),
                                ln_theta_yt.reshape(lon.shape),
                                ln_theta_zz.reshape(lon.shape),
                                ln_theta_zt.reshape(lon.shape),
                                f.reshape(lon.shape),
                                u.reshape(lon.shape),
                                v.reshape(lon.shape),
                                u_x.reshape(lon.shape),
                                v_x.reshape(lon.shape),
                                u_y.reshape(lon.shape),
                                v_y.reshape(lon.shape),
                                u_z.reshape(lon.shape),
                                v_z.reshape(lon.shape),
                                u_t.reshape(lon.shape),
                                v_t.reshape(lon.shape))
                                
        # 如果 kz 过小，停止积分
        if (abs(kz) <= 1000.0).all():
            error_num = 1
        nans_indices = np.where(np.abs(kz) <= 1000)
        kx[nans_indices] = np.nan
        ky[nans_indices] = np.nan
        kz[nans_indices] = np.nan
        ln_theta_x[nans_indices] = np.nan
        ln_theta_y[nans_indices] = np.nan
        ln_theta_z[nans_indices] = np.nan
        f[nans_indices] = np.nan
        freq[nans_indices] = np.nan
        ln_theta_xx[nans_indices] = np.nan
        ln_theta_xy[nans_indices] = np.nan
        ln_theta_xz[nans_indices] = np.nan
        ln_theta_xt[nans_indices] = np.nan
        ln_theta_yx[nans_indices] = np.nan
        ln_theta_yy[nans_indices] = np.nan
        ln_theta_yz[nans_indices] = np.nan
        ln_theta_yt[nans_indices] = np.nan
        ln_theta_zz[nans_indices] = np.nan
        ln_theta_zt[nans_indices] = np.nan

        # print('ln_theta_xx = ', ln_theta_xx)
        # print('ln_theta_xy = ', ln_theta_xx)
        # print('ln_theta_xz = ', ln_theta_xx)
        # print('ln_theta_xt = ', ln_theta_xx)
        # print('ln_theta_zx = ', ln_theta_xx)
        # 计算固有频率
        freq_intrinsic = freq - u * kx /rearth - v * ky / rearth
        
        # 计算群速 (ug, vg, wg) 及 kx, ky, kz 的个别变化
        # print('freq = ', freq)
        # print('u = ', u)
        # print('v = ', v)
        # print('ln_theta_x = ', ln_theta_x)
        # print('ln_theta_y = ', ln_theta_y)
        # print('ln_theta_z = ', ln_theta_z)
        # print('f = ', f)
        # print('freq_intrinsic = ', freq_intrinsic)
        # print('kx = ', kx)
        # print('ky = ', ky)
        # print('kz = ', kz)
        ug, vg, wg = cal_group_velocity_extent(u, v, ln_theta_x, ln_theta_y, ln_theta_z, f, freq_intrinsic, kx, ky, kz)
        
        # print('ug = ', ug)
        # print('vg = ', vg)
        # print('wg = ', wg)
        
        dlon, dlat, dz, dkx, dky, dkz, dug, dvg, dwg, dfreq = core_diffun(freq_intrinsic, kx, ky, kz, ln_theta_xx, ln_theta_xy, ln_theta_xz, ln_theta_xt, 
                        ln_theta_yx, ln_theta_yy, ln_theta_yz, ln_theta_yt, ln_theta_zz, ln_theta_zt,
                        u_x, v_x, u_y, v_y, u_z, v_z, u_t, v_t, ug, vg, wg, lat)
        
        # dlon[nans_indices] = np.nan
        # dlat[nans_indices] = np.nan
        # dz[nans_indices] = np.nan
        # dkx[nans_indices] = np.nan
        # dky[nans_indices] = np.nan
        # dkz[nans_indices] = np.nan
        # dug[nans_indices] = np.nan
        # dvg[nans_indices] = np.nan
        # dwg[nans_indices] = np.nan
        # dfreq[nans_indices] = np.nan
        # print('dlon = ', dlon.reshape(-1))
        # print('dlat = ', dlat.reshape(-1))
        # print('dz = ', dz.reshape(-1))
        # print('dkx = ', dkx.reshape(-1))
        # print('dky = ', dkx.reshape(-1))
        # print('dkz = ', dkx.reshape(-1))
        # print(' ')
        return np.array([dlon, dlat, dz, dkx, dky, dkz, dug, dvg, dwg, dfreq], dtype=self.all_dtype), error_num

    def rk4_step(self, y, t):
        """对状态变量 y 执行单步四阶Runge-Kutta积分，返回积分后的新output状态。"""
        dt = self.tstep
        
        k1, err = self.diffun(y, t)
        if err == 1:
            return y, err
        
        k2, err = self.diffun(y + 0.5 * dt * k1, t)
        if err == 1:
            return y, err
        
        k3, err = self.diffun(y + 0.5 * dt * k2, t)
        if err == 1:
            return y, err

        k4, err = self.diffun(y + dt * k3, t)
        if err == 1:
            return y, err
        # print('k1 = ', k1.reshape(-1))
        # print('k2 = ', k2.)
        # print('k3 = ', k3)
        # print('k4 = ', k4)
        
        # RK4 更新公式: y_next = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return core_rk4_step(y, k1, k2, k3, k4, dt), err

    def ray_run(self):
        """"执行射线追踪积分"""
        np.seterr(divide='raise', invalid='raise')
        
        print('Initializing ray tracing...')
        self.ray_initial()
        y = np.array([
            self.rlon[0, :, :, :, :],
            self.rlat[0, :, :, :, :],
            self.rz[0, :, :, :, :],
            self.rzwn[0, :, :, :, :],
            self.rmwn[0, :, :, :, :],
            self.rvwn[0, :, :, :, :],
            self.rug[0, :, :, :, :],
            self.rvg[0, :, :, :, :],
            self.rwg[0, :, :, :, :],
            self.rfreq[0, :, :, :, :],
        ], dtype=self.all_dtype)

        print('Starting ray tracing integration...')
        for it in range(self.nnt - 1):
            t = it * self.tstep + self.t_start
            progress_bar(it, self.nnt, bar_length=50)
            # print('rk4 runing')
            result, err = self.rk4_step(y, t)
            if err == 1:
                break
            lon_new, lat_new, z_new = result[0], result[1], result[2]
            kx_new, ky_new, kz_new = result[3], result[4], result[5]
            freq_new = result[-1]
            
            # print('lon_new = ',lon_new.reshape(-1))
            # print('lat_new = ',lat_new.reshape(-1))
            # print('z_new = ',z_new.reshape(-1))

            nan_indices = np.where(np.abs(lat_new) >= 0.5 * pi)
            lon_new[nan_indices] = np.nan
            lat_new[nan_indices] = np.nan
            z_new[nan_indices] = np.nan
            kx_new[nan_indices] = np.nan
            ky_new[nan_indices] = np.nan
            kz_new[nan_indices] = np.nan
            freq_new[nan_indices] = np.nan
            
            # ddis = cal_dis(lon_new,lat_new,self.rlon[it],self.rlat[it])
            # nan_indices = np.where(np.abs(ddis) >= self.cut_off)
            # lon_new[nan_indices] = np.nan
            # lat_new[nan_indices] = np.nan
            # z_new[nan_indices] = np.nan
            # kx_new[nan_indices] = np.nan
            # ky_new[nan_indices] = np.nan
            # kz_new[nan_indices] = np.nan
            # freq_new[nan_indices] = np.nan
            
            if (np.isnan(lon_new)).all() or (np.abs(lat_new) > 0.5 * pi).all():
                break
            
            res = self.bf.cal_bf_point_initial(lon_new.reshape(-1), lat_new.reshape(-1), z_new.reshape(-1), t)

            (ln_theta_x, ln_theta_y, ln_theta_z, u, v, f) = res


            # 计算固有频率
            (u, v, ln_theta_x, ln_theta_y, ln_theta_z, f) = (u.reshape(lon_new.shape),
                                                             v.reshape(lon_new.shape),
                                                             ln_theta_x.reshape(lon_new.shape),
                                                             ln_theta_y.reshape(lon_new.shape),
                                                             ln_theta_z.reshape(lon_new.shape),
                                                             f.reshape(lon_new.shape))
            freq_intrinsic = freq_new - u * kx_new / rearth - v * ky_new / rearth
            
            # 计算群速度
            ug_new, vg_new, wg_new = cal_group_velocity_extent(u, v, ln_theta_x, ln_theta_y, ln_theta_z, f, freq_intrinsic, kx_new, ky_new, kz_new)
            
            # print(ug_new,vg_new)
            self.rlon[it + 1, :, :, :, :] = lon_new
            self.rlat[it + 1, :, :, :, :] = lat_new
            self.rz[it + 1, :, :, :, :] = z_new
            self.rzwn[it + 1, :, :, :, :] = kx_new
            self.rmwn[it + 1, :, :, :, :] = ky_new
            self.rvwn[it + 1, :, :, :, :] = kz_new
            self.rug[it + 1, :, :, :, :] = ug_new
            self.rvg[it + 1, :, :, :, :] = vg_new
            self.rwg[it + 1, :, :, :, :] = wg_new
            self.rfreq[it + 1, :, :, :, :] = freq_new
            
            y = np.array([
                self.rlon[it + 1, :, :, :, :],
                self.rlat[it + 1, :, :, :, :],
                self.rz[it + 1, :, :, :, :],
                self.rzwn[it + 1, :, :, :, :],
                self.rmwn[it + 1, :, :, :, :],
                self.rvwn[it + 1, :, :, :, :],
                self.rug[it + 1, :, :, :, :],
                self.rvg[it + 1, :, :, :, :],
                self.rwg[it + 1, :, :, :, :],
                self.rfreq[it + 1, :, :, :, :]
            ], dtype=self.all_dtype)

    def output(self, wrfile):
        """将射线追踪结果输出到 NetCDF 文件。"""
        from datetime import datetime

        print('\nWriting ray tracing results to file...')

        out_type = self.all_dtype
        ds = Dataset(wrfile, 'w', format='NETCDF4')
        
        # 添加全局属性
        global_attrs = {
            'Title': 'The Result of Inertial Gravity Wave Ray Tracing (IGWRT) Program',
            'Author': 'Dr. Candidate LIU Yifan, Prof. LI Jianping',
            'Version': '2026-03-15 Ver.',
            'Institution': 'COAS/AFO/POL/DOMES/COCN, OUC, Qingdao, China',
            'History': f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        }
        
        for attr_name, attr_value in global_attrs.items():
            ds.setncattr(attr_name, attr_value)
    
        # 定义维度
        ds.createDimension('Lx', self.nLx)
        ds.createDimension('Ly', self.nLy)
        ds.createDimension('source', self.nsource)
        ds.createDimension('root', 2)
        ds.createDimension('time', self.nnt)
        
        # 定义坐标变量并赋值
        ds.createVariable('Lx', out_type, ('Lx',))[:] = self.Lx
        ds.createVariable('Ly', out_type, ('Ly',))[:] = self.Ly
        ds.createVariable('source', 'i4', ('source',))[:] = np.arange(self.nsource)
        ds.createVariable('root', 'i4', ('root',))[:] = np.array([0, 1])
        ds.createVariable('time', out_type, ('time',))[:] = np.arange(self.nnt) * self.tstep
        
        # 定义输出变量
        rlon_var = ds.createVariable(
            'rlon', out_type, ('time', 'root', 'source', 'Ly','Lx'))
        rlat_var = ds.createVariable(
            'rlat', out_type, ('time', 'root', 'source', 'Ly', 'Lx'))
        rz_var = ds.createVariable(
            'rz', out_type, ('time', 'root', 'source', 'Ly', 'Lx'))
        rp_var = ds.createVariable(
            'rp', out_type, ('time', 'root', 'source', 'Ly', 'Lx'))
        rzwn_var = ds.createVariable(
            'rzwn', out_type, ('time', 'root', 'source', 'Ly', 'Lx'))
        rmwn_var = ds.createVariable(
            'rmwn', out_type, ('time', 'root', 'source', 'Ly', 'Lx'))
        rvwn_var = ds.createVariable(
            'rvwn', out_type, ('time', 'root', 'source', 'Ly', 'Lx'))
        rug_var = ds.createVariable(
            'rug', out_type, ('time', 'root', 'source', 'Ly', 'Lx'))
        rvg_var = ds.createVariable(
            'rvg', out_type, ('time', 'root', 'source', 'Ly', 'Lx'))
        rwg_var = ds.createVariable(
            'rwg', out_type, ('time', 'root', 'source', 'Ly', 'Lx'))
        rfreq_var = ds.createVariable(
            'rfreq', out_type, ('time', 'root', 'source', 'Ly', 'Lx'))
        rpreiod_var = ds.createVariable(
            'rpreiod', out_type, ('time', 'root', 'source', 'Ly', 'Lx'))
        # 写入数据（经纬度转换为度）
        rlon_var[:] = self.rlon * rad2deg
        rlat_var[:] = self.rlat * rad2deg
        rz_var[:] = self.rz
        rp_var[:] = p0 *np.exp(-self.rz/H0)
        rzwn_var[:] = self.rzwn
        rmwn_var[:] = self.rmwn
        rvwn_var[:] = self.rvwn
        rug_var[:] = self.rug   # 群速度 (m/s)
        rvg_var[:] = self.rvg
        rwg_var[:] = self.rwg
        rfreq_var[:] = self.rfreq
        rpreiod_var[:] = 2*np.pi / self.rfreq/ 86400
        # 可选：添加属性说明单位
        rlon_var.units = 'degrees'
        rlat_var.units = 'degrees'
        rp_var.units = 'hPa'
        rzwn_var.units = 'rad / meter * Rearth'
        rmwn_var.units = 'rad / meter * Rearth'
        rvwn_var.units = 'rad / meter * Rearth'
        rug_var.units = 'm s**-1'
        rvg_var.units = 'm s**-1'
        rwg_var.units = 'm s**-1'
        rfreq_var.units = 's**-1'
        rpreiod_var.units = 'days'
        
        ds.close()
        print(f'\nResult wrfile was written successfully: {wrfile}')



















































