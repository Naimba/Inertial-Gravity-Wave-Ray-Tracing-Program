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

nb_para_dic = {
    'nopython': True,
    # 'fastmath': True,
    'cache': True,
}


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
@nb.jit(nopython=True, cache=True, fastmath=True)
def core_diffun(freq, kx, ky, kz, ln_theta_xx, ln_theta_xy, ln_theta_xz, ln_theta_xt, 
                ln_theta_yx, ln_theta_yy, ln_theta_yz, ln_theta_yt, ln_theta_zx, ln_theta_zy, ln_theta_zz, ln_theta_zt, ug, vg, wg, lat):
    # K_2^2 and K_3^2
    kk_2 = kx**2 + ky**2
    kk_3 = kk_2  + kz**2
    
    # print('ln_theta_zz = ', ln_theta_zz.reshape(-1))
    # print('ln_theta_zx = ', ln_theta_zx.reshape(-1))
    # print('kx = ', kx.reshape(-1))
    # print('ky = ', ky.reshape(-1))
    # print('kz = ', kz.reshape(-1))
    
    # calculate dkx/dt, dky/dt and dkz/dt
    dzwn = -( grav * kk_2 * ln_theta_zx - kz * grav * (kx * ln_theta_xx + ky * ln_theta_yx) / rearth ) / (2 * freq * kk_3)
    dmwn = -( grav * kk_2 * ln_theta_zy - kz * grav * (kx * ln_theta_xy + ky * ln_theta_yy) / rearth ) / (2 * freq * kk_3)
    dvwn = -( rearth * grav * kk_2 * ln_theta_zz - kz * grav * (kx * ln_theta_xz + ky * ln_theta_yz) ) / (2 * freq * kk_3)
    
    # print('dzwn = ', dzwn.reshape(-1))
    # print('dmwn = ', dmwn.reshape(-1))
    # print('dvwn = ', dvwn.reshape(-1))
    
    # calculate domega/dt
    dfreq = ( grav * kk_2 * ln_theta_zt - kz * grav * (kx * ln_theta_xt + ky * ln_theta_yt) / rearth ) / (2 * freq * kk_3)
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
@nb.jit(nopython=True, cache=True, fastmath=True)
def core_rk4_step(y, k1, k2, k3, k4, dt):
    temp = y
    ks = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    temp[0:6] = y[0:6] + ks[0:6]
    temp[6,7,8] = ks[6,7,8] / dt
    temp[-1] = y[-1] + ks[-1]
    # print('temp.shape = ', temp.shape)
    return temp

# @nb.jit('f8[:,:,:](f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:])', **nb_para_dic)
@nb.jit(nopython=True, cache=True, fastmath=True)
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
@nb.jit(nopython=True, cache=True, fastmath=True)
def cal_group_velocity_extent(ln_theta_x, ln_theta_y, ln_theta_z, f, freq, kx, ky, kz):
    kk_3 = kx * kx + ky * ky + kz * kz

    # ug = - (2 * freq * freq * kx - 2 * kx * grav * ln_theta_z - kz * grav * ln_theta_x) / (2 * freq * kk_3)
    # vg = - (2 * freq * freq * ky - 2 * ky * grav * ln_theta_z - kz * grav * ln_theta_y) / (2 * freq * kk_3)
    # wg = - kz * (freq * freq - f * f) / (freq * kk_3)
    # print('ln_theta_z.shape = ',ln_theta_z.shape)
    # print('kx.shape = ',kx.shape)
    
    # print('freq = ', freq.reshape(-1))
    # print('kx = ', kx.reshape(-1))
    # print('ky = ', ky.reshape(-1))
    # print('kz = ', kz.reshape(-1))
    # print('ln_theta_x = ', ln_theta_x.reshape(-1))
    # print('ln_theta_y = ', ln_theta_y.reshape(-1))
    # print('ln_theta_z = ', ln_theta_z.reshape(-1))


    
    ug = - (2 * rearth * freq * freq * kx - 2 * rearth * kx * grav * ln_theta_z + kz * grav * ln_theta_x) / (2 * freq * kk_3)
    vg = - (2 * rearth * freq * freq * ky - 2 * rearth * ky * grav * ln_theta_z + kz * grav * ln_theta_y) / (2 * freq * kk_3)
    
    # print('ug = ', ug.reshape(-1))
    # print('vg = ', vg.reshape(-1))
    
    wg = - ( 2 * rearth * kz * (freq**2 - f**2) + kx * grav * ln_theta_x + ky * grav * ln_theta_y )/ (2 * freq * kk_3)

    # print('ug.shape = ',ug.shape)
    # print('wg.shape = ',wg.shape)
    return ug, vg, wg

class WR:
    """
    波射线追踪类:
    属性:
      bf         - 基本流场 BF 对象 (组合)
      zwn, mwn      - zonal and meridional wavenumber array (length: nzwn and nmwn)
      nnx, nny    - 纬向和经向波源点数
      p_list      - 垂向波源所在气压层
      dt         - 输入数据的时间间隔
      tstep      - 积分时间步长 (秒)
      ttotal     - 总积分时间 (秒)
      freq       - 初始波频率
      cal_dtype, read_dtype - 计算精度和读取精度
      cut_off - 防止速度过快进行的阶段处理
      t_start - 积分的时间起点
      inputfile - 输入的基本场文件，里面需要有温度数据或位温数据

    """
    def __init__(self, zwn, mwn, lon_list, lat_list, p_list, dt, tstep=1. * hour, ttotal=20. * day,
                 freq=0, cal_dtype='float64', read_dtype='float32',
                 cut_off=0.1, t_start=0, inputfile=None):
    # def __init__(self, zwn, mwn, nnx, nny, p_list, dt, tstep=1. * hour, ttotal=20. * day,
                 # freq=0, cal_dtype='float64', read_dtype='float32',
                 # cut_off=0.1, t_start=0, inputfile=None):
        # 初始化基本流场
        self.all_dtype = cal_dtype
        if inputfile is None:
            raise ValueError('inputfile is need')
        else:
            nx, ny, nz, nt = self.get_nxyzt(inputfile)
        self.bf = BF(nx, ny, nz, nt, dt, read_dtype=read_dtype, cal_dtype=cal_dtype)

        # 赋值已给诸值
        nzwn = len(zwn)
        nmwn = len(mwn)
        self.nzwn = nzwn
        self.nmwn = nmwn
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
        self.zwn = np.array(zwn, dtype=self.all_dtype)
        self.mwn = np.array(mwn, dtype=self.all_dtype)
        self.p_list =  np.array(p_list, dtype=self.all_dtype)
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
        shape = (self.nnt, 2, nsource, nmwn, nzwn)
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
        self.bf.lon = np.array(ds.variables['lon'][:], dtype=self.all_dtype)
        self.bf.lat = np.array(ds.variables['lat'][:], dtype=self.all_dtype)
        self.bf.p = np.array(ds.variables['level'][:], dtype=self.all_dtype)
        self.bf.z = -H0 * np.log(self.bf.p/p0)
        self.bf.t = np.array(ds.variables['t'][:], dtype=self.all_dtype)

        self.bf.theta = np.array(ds.variables['theta'][:], dtype=self.all_dtype)
        self.bf.ln_theta = np.array(ds.variables['ln_theta'][:], dtype=self.all_dtype)
        
        self.bf.ln_theta_x = np.array(ds.variables['ln_theta_x'][:], dtype=self.all_dtype)
        self.bf.ln_theta_y = np.array(ds.variables['ln_theta_y'][:], dtype=self.all_dtype)
        self.bf.ln_theta_z = np.array(ds.variables['ln_theta_z'][:], dtype=self.all_dtype)
        self.bf.ln_theta_t = np.array(ds.variables['ln_theta_t'][:], dtype=self.all_dtype)

        self.bf.ln_theta_xx = np.array(ds.variables['ln_theta_xx'][:], dtype=self.all_dtype)
        self.bf.ln_theta_xy = np.array(ds.variables['ln_theta_xy'][:], dtype=self.all_dtype)
        self.bf.ln_theta_xz = np.array(ds.variables['ln_theta_xz'][:], dtype=self.all_dtype)
        self.bf.ln_theta_xt = np.array(ds.variables['ln_theta_xt'][:], dtype=self.all_dtype)

        self.bf.ln_theta_yx = np.array(ds.variables['ln_theta_yx'][:], dtype=self.all_dtype)
        self.bf.ln_theta_yy = np.array(ds.variables['ln_theta_yy'][:], dtype=self.all_dtype)
        self.bf.ln_theta_yz = np.array(ds.variables['ln_theta_yz'][:], dtype=self.all_dtype)
        self.bf.ln_theta_yt = np.array(ds.variables['ln_theta_yt'][:], dtype=self.all_dtype)
        
        self.bf.ln_theta_zx = np.array(ds.variables['ln_theta_zx'][:], dtype=self.all_dtype)
        self.bf.ln_theta_zy = np.array(ds.variables['ln_theta_zy'][:], dtype=self.all_dtype)
        self.bf.ln_theta_zz = np.array(ds.variables['ln_theta_zz'][:], dtype=self.all_dtype)
        self.bf.ln_theta_zt = np.array(ds.variables['ln_theta_zt'][:], dtype=self.all_dtype)
        
        self.bf.fields = np.stack([
            self.bf.ln_theta_x,
            self.bf.ln_theta_y,
            self.bf.ln_theta_z,
            self.bf.ln_theta_t,
            self.bf.ln_theta_xx,
            self.bf.ln_theta_xy,
            self.bf.ln_theta_xz,
            self.bf.ln_theta_xt,
            self.bf.ln_theta_yx,
            self.bf.ln_theta_yy,
            self.bf.ln_theta_yz,
            self.bf.ln_theta_yt,
            self.bf.ln_theta_zx,
            self.bf.ln_theta_zy,
            self.bf.ln_theta_zz,
            self.bf.ln_theta_zt,
        ], axis=-1)
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

    # def set_source_matrix(self, SW_lon, SW_lat, dlon, dlat, nnx, nny):
    #     """
    #     设置源点为规则网格: 以 (SW_lon, SW_lat) 为起点，间隔 dlon, dlat (单位度)，
    #     网格大小 nnx * nny * len(p_list)，总数应等于 nsource。
    #     """
    #     # 检查纬度范围
    #     if SW_lat + (nny - 1) * dlat > 89.0:
    #         raise ValueError("source latitude out of -90~90 range!")
    #     # 规范化起始经度到 [0, 360)
    #     SW_lon = SW_lon % 360.0
    #     idx = 0
    #     p_list = self.p_list
    #     nnz = len(p_list)
    #     for iy in range(nny):
    #         for ix in range(nnx):
    #             for iz in range(nnz):
    #                 lon_deg = SW_lon + ix * dlon
    #                 lat_deg = SW_lat + iy * dlat
    #                 # 经度规范化到 [0, 360)
    #                 lon_deg = lon_deg % 360.0
    #                 self.source_lon[idx] = lon_deg * deg2rad
    #                 self.source_lat[idx] = lat_deg * deg2rad
    #                 self.source_p[idx] = p_list[iz]
    #                 self.source_z[idx] = -H0*np.log(p_list[iz]/p0)
    #                 idx += 1
                
    def ray_info(self):
        """输出波射线追踪的初始信息"""
        print(
            "==============================================================================")
        print("IGWRT Package: Inertial Gravity Wave Ray Tracing Information ")
        print(
            f" Basic State Grid (nlon x nlat x np x nt): {self.bf.nx} x {self.bf.ny} x {self.bf.nz} x {self.bf.nt}")
        
        print(f" Initial Zonal Wavenumbers (nzwn): {self.nzwn}")
        print(" " * 15 + " ".join(f"{z:.1f}" for z in self.zwn))
        
        print(f" Initial Meridional Wavenumbers (nmwn): {self.nmwn}")
        print(" " * 15 + " ".join(f"{z:.1f}" for z in self.mwn))
        
        print(f" Source Locations (total {self.nsource} points):")
        # 打印每个源点经纬度（度）
        for i in range(self.nsource):
            lon_deg = self.source_lon[i] * rad2deg
            lat_deg = self.source_lat[i] * rad2deg
            source_p = self.source_p[i]
            print(" " * 15 + f"{lon_deg:7.2f}, {lat_deg:7.2f}, {source_p:7.2f} hPa")
        print(f" Time Step (s): {self.tstep:.1f}")
        print(f" Total Integration Time (day): {self.ttotal/day:.1f}")
        print(f" Total Steps (nnt): {self.nnt}")
        print(
            "==============================================================================")
        
        
    def cal_group_velocity(self, ln_theta_x, ln_theta_y, ln_theta_z, f, freq, kx, ky, kz):
        """
        计算群速 (ug, vg, wg)
        """
        
        # 这段不知道是干啥的，改编自Ms. Y. N. Yang 的Rossby波射线追踪程序
        # nans = np.einsum('ij,j->ij', kz * 0, ln_theta_x * 0) + 1
        # nans[np.isnan(nans)] = 0
        
        kk_3 = kx * kx + ky * ky + kz * kz
        
        # print('freq = ', freq.reshape(-1))
        # print('kx = ', kx.reshape(-1))
        # print('ky = ', ky.reshape(-1))
        # print('kz = ', kz.reshape(-1))
        # print('ln_theta_x = ', ln_theta_x.reshape(-1))
        # print('ln_theta_y = ', ln_theta_y.reshape(-1))
        # print('ln_theta_z = ', ln_theta_z.reshape(-1))

        
        ug = - (2 * rearth * freq * freq * kx - 2 * rearth * kx * grav * ln_theta_z[:, None] + kz * grav * ln_theta_x[:, None]) / (2 * freq * kk_3)
        vg = - (2 * rearth * freq * freq * ky - 2 * rearth * ky * grav * ln_theta_z[:, None] + kz * grav * ln_theta_y[:, None]) / (2 * freq * kk_3)
        wg = - (2 * rearth * kz * (freq * freq - f[:, None] * f[:, None]) + kx * grav * ln_theta_x[:, None] + ky * grav * ln_theta_y[:, None])/ (2 * freq * kk_3)
        
        # print('ug = ', ug)
        # print('vg = ', vg)
        # print('wg = ', wg)
        
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
        res = self.bf.cal_bf_point(lon0, lat0, z0, t0)
        (ln_theta_x, ln_theta_y, ln_theta_z, ln_theta_t,
         ln_theta_xx, ln_theta_xy, ln_theta_xz, ln_theta_xt,
         ln_theta_yx, ln_theta_yy, ln_theta_yz, ln_theta_yt,
         ln_theta_zx, ln_theta_zy, ln_theta_zz, ln_theta_zt, f) = res
        # 对每个初始 zonal 波数求 m 并初始化状态

        # 设置局地 k (zwn) 初值
        self.rzwn[idx0, :, :, :, :] = self.zwn[None, None, None, None, :]
        self.rmwn[idx0, :, :, :, :] = self.mwn[None, None, None, :, None]
        for ik in range(self.nzwn):
            for il in range(self.nmwn):
                kx = self.zwn[ik]
                ky = self.mwn[il]
                # 计算垂向波数
                m_list, rootnum = cal_kz(ln_theta_x, ln_theta_y, ln_theta_z, f, self.freq, kx, ky)
                m_val = np.transpose(m_list, (1, 0))
                self.rvwn[idx0, :, :, il, ik] = m_val
                #设置初始频率
                self.rfreq[idx0, :, :, il, ik] = self.freq
                # 计算初始群速度并存储
                freq = self.freq
                ug0, vg0, wg0 = self.cal_group_velocity(ln_theta_x, ln_theta_y, ln_theta_z, f, freq, kx, ky, m_list)
                self.rug[idx0, :, :, il, ik] = np.transpose(ug0)
                self.rvg[idx0, :, :, il, ik] = np.transpose(vg0)
                self.rwg[idx0, :, :, il, ik] = np.transpose(wg0)
                
    def diffun(self, y, t):
        """
        计算状态变量 y 的导数 dy/dt，用于 Runge-Kutta 积分。
        y 为长度10的数组: [lon, lat, z*, kx, ky, kz, ug, vg, wg, freq]
        返回相同长度的导数组 dk (或在不继续积分时返回 error_num == 1)。
        """
        lon, lat, z, kx, ky, kz, freq = y[0], y[1], y[2], y[3], y[4], y[5], y[-1]
        error_num = 0
        # print('z = ',z)
        if (abs(lat) >= 0.5 * pi).all():
            error_num = 1
        # 计算当前位置背景场变量 
        # 插值时对于超范围的位置已经设为了nan
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
         ln_theta_zx, ln_theta_zy, ln_theta_zz, ln_theta_zt, f) = res
        
        (ln_theta_x, ln_theta_y, ln_theta_z, ln_theta_t,
         ln_theta_xx, ln_theta_xy, ln_theta_xz, ln_theta_xt,
         ln_theta_yx, ln_theta_yy, ln_theta_yz, ln_theta_yt,
         ln_theta_zx, ln_theta_zy, ln_theta_zz, ln_theta_zt, f) = (ln_theta_x.reshape(lon.shape),
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
                                                                   ln_theta_zx.reshape(lon.shape),
                                                                   ln_theta_zy.reshape(lon.shape),
                                                                   ln_theta_zz.reshape(lon.shape),
                                                                   ln_theta_zt.reshape(lon.shape),
                                                                   f.reshape(lon.shape))
        # 如果 kz 过小，停止积分
        # if (abs(kz) <= 10000.0).all():
        #     error_num = 1
        # nans_indices = np.where(np.abs(kz) <= 1000)
        # kx[nans_indices] = np.nan
        # ky[nans_indices] = np.nan
        # kz[nans_indices] = np.nan
        # ln_theta_x[nans_indices] = np.nan
        # ln_theta_y[nans_indices] = np.nan
        # ln_theta_z[nans_indices] = np.nan
        # f[nans_indices] = np.nan
        # freq[nans_indices] = np.nan
        # ln_theta_xx[nans_indices] = np.nan
        # ln_theta_xy[nans_indices] = np.nan
        # ln_theta_xz[nans_indices] = np.nan
        # ln_theta_xt[nans_indices] = np.nan
        # ln_theta_yx[nans_indices] = np.nan
        # ln_theta_yy[nans_indices] = np.nan
        # ln_theta_yz[nans_indices] = np.nan
        # ln_theta_yt[nans_indices] = np.nan
        # ln_theta_zx[nans_indices] = np.nan
        # ln_theta_zy[nans_indices] = np.nan
        # ln_theta_zz[nans_indices] = np.nan
        # ln_theta_zt[nans_indices] = np.nan

        
        # 计算群速 (ug, vg, wg) 及 kx, ky, kz 的个别变化
        ug, vg, wg = cal_group_velocity_extent(ln_theta_x, ln_theta_y, ln_theta_z, f, freq, kx, ky, kz)
        
        # print('ug = ', ug)
        # print('vg = ', vg)
        # print('wg = ', wg)
        
        dlon, dlat, dz, dkx, dky, dkz, dug, dvg, dwg, dfreq = core_diffun(freq, kx, ky, kz, 
                                                                          ln_theta_xx, ln_theta_xy, ln_theta_xz, ln_theta_xt,
                                                                          ln_theta_yx, ln_theta_yy, ln_theta_yz, ln_theta_yt,
                                                                          ln_theta_zx, ln_theta_zy, ln_theta_zz, ln_theta_zt,
                                                                          ug, vg, wg, lat)
        
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

        return np.array([dlon, dlat, dz, dkx, dky, dkz, dug, dvg, dwg, dfreq], dtype=self.all_dtype), error_num

    def rk4_step(self, y, t):
        """对状态变量 y 执行单步四阶Runge-Kutta积分，返回积分后的新output状态。"""
        dt = self.tstep
        
        k1, err = self.diffun(y, t)
        if err == 1:
            return y, err
        # print('k1 = ', k1.reshape(-1))
        # print('k1.shape = ',k1.shape)
        
        k2, err = self.diffun(y + 0.5 * dt * k1, t)
        if err == 1:
            return y, err
        # print('k2 = ', k2.reshape(-1))
        # print('k2 = ', k2.shape)

        k3, err = self.diffun(y + 0.5 * dt * k2, t)
        if err == 1:
            return y, err
        # print('k3 = ', k3.reshape(-1))
        # print('k3 = ', k3.shape)

        k4, err = self.diffun(y + dt * k3, t)
        if err == 1:
            return y, err
        # print('k4 = ', k4.reshape(-1))
        # print('k4 = ', k4.shape)

        # RK4 更新公式: y_next = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return core_rk4_step(y, k1, k2, k3, k4, dt), err

    def ray_run(self):
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

        for it in range(self.nnt - 1):
            # print('it = ',it)
            t = (it+1)*self.tstep
            progress_bar(it, self.nnt, bar_length=50)
            # print('')
            result, err = self.rk4_step(y, t)
            if err == 1:
                break
            lon_new, lat_new, z_new = result[0], result[1], result[2]
            kx_new, ky_new, kz_new = result[3], result[4], result[5]
            freq_new = result[-1]
            
            # print('')
            # print('lon_new = ',lon_new.reshape(-1))
            # print('lat_new = ',lat_new.reshape(-1))
            # print('z_new = ',z_new.reshape(-1))

            # nan_indices = np.where(np.abs(lat_new) >= 0.5 * pi)
            # lon_new[nan_indices] = np.nan
            # lat_new[nan_indices] = np.nan
            # z_new[nan_indices] = np.nan
            # kx_new[nan_indices] = np.nan
            # ky_new[nan_indices] = np.nan
            # kz_new[nan_indices] = np.nan
            # freq_new[nan_indices] = np.nan
            
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
            
            res = self.bf.cal_bf_point(lon_new.reshape(-1), lat_new.reshape(-1), z_new.reshape(-1), t)

            (ln_theta_x, ln_theta_y, ln_theta_z, ln_theta_t,
             ln_theta_xx, ln_theta_xy, ln_theta_xz, ln_theta_xt,
             ln_theta_yx, ln_theta_yy, ln_theta_yz, ln_theta_yt,
             ln_theta_zx, ln_theta_zy, ln_theta_zz, ln_theta_zt, f) = res

            (ln_theta_x, ln_theta_y, ln_theta_z, f) = (ln_theta_x.reshape(lon_new.shape),
                                                       ln_theta_y.reshape(lon_new.shape),
                                                       ln_theta_z.reshape(lon_new.shape),
                                                       f.reshape(lon_new.shape))
            ug_new, vg_new, wg_new = cal_group_velocity_extent(ln_theta_x, ln_theta_y, ln_theta_z, f, freq_new, kx_new, ky_new, kz_new)
            
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
 
        out_type = self.all_dtype
        ds = Dataset(wrfile, 'w', format='NETCDF4')
        
        # 添加全局属性
        global_attrs = {
            'Title': 'The Result of Inertial Gravity Wave Ray Tracing (IGWRT) Program',
            'Author': 'Dr. Candidate LIU Yifan, Prof. LI Jianping',
            'Version': '2025-12-15 Ver.',
            'Institution': 'DOMES/POL/AFO/COAS/COCN, OUC, Qingdao, China',
            'History': f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        }
        
        for attr_name, attr_value in global_attrs.items():
            ds.setncattr(attr_name, attr_value)
    
        # 定义维度
        ds.createDimension('zwn', self.nzwn)
        ds.createDimension('mwn', self.nmwn)
        ds.createDimension('source', self.nsource)
        ds.createDimension('root', 2)
        ds.createDimension('time', self.nnt)
        
        # 定义坐标变量并赋值
        ds.createVariable('zwn', out_type, ('zwn',))[:] = self.zwn
        ds.createVariable('mwn', out_type, ('mwn',))[:] = self.mwn
        ds.createVariable('source', 'i4', ('source',))[:] = np.arange(self.nsource)
        ds.createVariable('root', 'i4', ('root',))[:] = np.array([0, 1])
        ds.createVariable('time', out_type, ('time',))[:] = np.arange(self.nnt) * self.tstep
        
        # 定义输出变量
        rlon_var = ds.createVariable(
            'rlon', out_type, ('time', 'root', 'source', 'mwn','zwn'))
        rlat_var = ds.createVariable(
            'rlat', out_type, ('time', 'root', 'source', 'mwn', 'zwn'))
        rz_var = ds.createVariable(
            'rz', out_type, ('time', 'root', 'source', 'mwn', 'zwn'))
        rp_var = ds.createVariable(
            'rp', out_type, ('time', 'root', 'source', 'mwn', 'zwn'))
        rzwn_var = ds.createVariable(
            'rzwn', out_type, ('time', 'root', 'source', 'mwn', 'zwn'))
        rmwn_var = ds.createVariable(
            'rmwn', out_type, ('time', 'root', 'source', 'mwn', 'zwn'))
        rvwn_var = ds.createVariable(
            'rvwn', out_type, ('time', 'root', 'source', 'mwn', 'zwn'))
        rug_var = ds.createVariable(
            'rug', out_type, ('time', 'root', 'source', 'mwn', 'zwn'))
        rvg_var = ds.createVariable(
            'rvg', out_type, ('time', 'root', 'source', 'mwn', 'zwn'))
        rwg_var = ds.createVariable(
            'rwg', out_type, ('time', 'root', 'source', 'mwn', 'zwn'))
        rfreq_var = ds.createVariable(
            'rfreq', out_type, ('time', 'root', 'source', 'mwn', 'zwn'))
        rpreiod_var = ds.createVariable(
            'rpreiod', out_type, ('time', 'root', 'source', 'mwn', 'zwn'))
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



















































