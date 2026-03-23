import numpy as np
from constants import pi, undef, H0, grav, omega, p0, R, cp, rearth
import netCDF4 as nc
from netCDF4 import Dataset
# from interpolation import batch_linint4
from scipy.ndimage import convolve
import numba as nb
from interpolation import interp4d_value, interp4d_grad, interp4d_grad2
'''
奥卡姆剃刀原理: 如无必要, 勿增实体
Occam's Razor: Entities should not be multiplied unnecessarily
Numquam ponenda est pluralitas sine necessitate
'''

# nb_para_dic = {
#     'nopython': True,
#     # 'fastmath':True,
#     'cache': True,
# }


# @nb.jit(['c16[:](c16[:])'], **nb_para_dic)  # ,'c8[:](c8[:])'
@nb.jit(nopython=True, cache=False, fastmath=False)
def roots_numba(p):
    return np.roots(p)

class BF:
    '''
    背景场类: 负责读取基本流场数据、计算位温及其导数、提供插值接口等功能。
    nx, ny, nz, nt: 网格尺寸
    dt: 背景场时间间隔
    read_dtype: 读取数据时使用的数据类型, 默认为float32以节省内存
    cal_dtype: 计算过程中使用的数据类型, 默认为float64以提高计算精度
    '''
    
    def __init__(self, nx, ny, nz, nt, dt, read_dtype='float32', cal_dtype='float64'):
        self.all_dtype = read_dtype
        self.all_dtype_ = cal_dtype
        # pi, rearth, omega, undef, delt = np.array([pi, rearth, omega, undef, delt],dtype=cal_dtype)
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nt = nt
        shape1 = (nx, ny, nz, nt)
        self.dt = dt

        # 
        self.theta = np.zeros(shape1, dtype=self.all_dtype)
        self.ln_theta = np.zeros(shape1, dtype=self.all_dtype)
        self.u = np.zeros(shape1, dtype=self.all_dtype)
        self.v = np.zeros(shape1, dtype=self.all_dtype)
        self.lon = np.zeros(nx, dtype=self.all_dtype_)
        self.lat = np.zeros(ny, dtype=self.all_dtype_)
        self.p = np.zeros(nz, dtype=self.all_dtype_)
        self.z = np.zeros(nz, dtype=self.all_dtype_)
        
    # 外部调用, 统一接口
    def getlon(self):
        return self.lon

    def getlat(self):
        return self.lat
    
    def getlevel(self):
        return self.p
    
    def gett(self):
        return self.t
   
    def cal_theta(self, temp, p):
        """
        计算四维温度数据的位温
        参数:
        temp: numpy数组, 四维温度数据 (x, y, p, t), 单位: K
        p: numpy数组, 气压数据, 与temp的第三维对应, 单位: hPa
        p0: float, 参考气压, 默认1000 hPa
        R: float, 气体常数, 默认287 J/(kg·K)
        cp: float, 定压比热容, 默认1004 J/(kg·K)
        返回:
        theta: numpy数组, 位温数据 (x, y, p, t), 单位: K
        """
        
        # 检查维度匹配
        if temp.shape[2] != len(p):
            raise ValueError("气压数据p的长度必须与温度数据temp的第三维长度匹配")
        
        # 重塑气压数组以便广播计算
        # 将p重塑为(1, 1, len(p), 1)以便与temp(x,y,p,t)广播
        p_reshaped = p.reshape((1, 1, len(p), 1))
        
        kappa = R / cp
        
        # 计算位温
        theta = temp * (p0 / p_reshaped) ** kappa
        
        return theta

    def read_uv(self, ds):
        
        u_candidates = ['u', 'uwnd', 'u-wind', 'U']
        v_candidates = ['v', 'vwnd', 'v-wind', 'V']
        u_data = None
        v_data = None
        for name in u_candidates:
            if name in ds.variables:
                u_data = np.array(
                    ds.variables[name][:],
                    dtype=self.all_dtype)
                u_data = np.transpose(u_data)
                break
            
        for name in v_candidates:
            if name in ds.variables:
                v_data = np.array(
                    ds.variables[name][:],
                    dtype=self.all_dtype)
                v_data = np.transpose(v_data)
                break

        return u_data, v_data
    
    def loadbf_ncfile(self, ncfile):
        '''
        如果文件中存在lon和lat, 使用它们做为坐标
        如果不存在lon或lat, 使用维度来构建, 默认-1维为lon, -2维为lat, 并给出warning
        默认为0E->360E和90S->90N
        '''
        
        ds = nc.Dataset(ncfile)
        
        # 自动识别纬度、经度变量
        lat_candidates = ['lat', 'latitude', 'Lat', 'Latitude']
        lon_candidates = ['lon', 'longitude', 'Lon', 'Longitude']
        p_candidates = ['p', 'pressure', 'P', 'Pressure', 'level', 'levels']
        

        lat_data = None
        lon_data = None
        p_data = None
        

        for name in lat_candidates:
            if name in ds.variables:
                lat_data = np.array(
                    ds.variables[name][:],
                    dtype=self.all_dtype)
                lat_data = lat_data * pi / 180
                self.lat[:] = lat_data.astype(self.all_dtype_)  # 转为弧度
                break
            
        for name in lon_candidates:
            if name in ds.variables:
                lon_data = np.array(
                    ds.variables[name][:],
                    dtype=self.all_dtype)
                lon_data = lon_data * pi / 180
                self.lon[:] = lon_data.astype(self.all_dtype_)
                break
            
        for name in p_candidates:
            if name in ds.variables:
                p_data = np.array(
                    ds.variables[name][:],
                    dtype=self.all_dtype)
                # print(f'p_data = {p_data}')
                z_data = -H0 * np.log(p_data/p0)
                # print(f'z_data = {z_data}')
                z_data.astype(self.all_dtype_)
                break
        
        
        # 如果找不到就构造规则网格
        if lat_data is None:
            self.lat = - pi * 0.5 + np.arange(self.ny) * pi / (self.ny - 1)
        if lon_data is None:
            self.lon = np.arange(self.nx) * 2.0 * pi / self.nx
        if p_data is None:
            Warning('No p data in your bffile')
        
        # 确保z坐标是增序
        sort_idz = np.argsort(z_data)
        p_data = p_data[sort_idz]
        z_data = z_data[sort_idz]
            
        self.z = z_data
        self.p = p_data
        
        # 构造积分时间数组
        self.t =  np.arange(self.nt) * self.dt
                
        ds = nc.Dataset(ncfile)
        # 注意: 由于Python读取数组相对于原数组是反的, 所以需要转置
        if 'theta' in ds.variables:
            temp_theta = np.array(ds.variables['theta'][:], dtype=self.all_dtype)
            temp_theta = np.transpose(temp_theta)
            temp_theta = temp_theta[:, :, sort_idz, :]
            u_data, v_data = self.read_uv(ds)
            
        elif 'T' in ds.variables:
            temp_T = np.array(ds.variables['T'][:], dtype=self.all_dtype)
            temp_T = np.transpose(temp_T)
            temp_T = temp_T[:, :, sort_idz, :]
            temp_theta = self.cal_theta(temp_T, p_data)
            u_data, v_data = self.read_uv(ds)
                
        elif 'air' in ds.variables:
            temp_T = np.array(ds.variables['air'][:], dtype=self.all_dtype)
            temp_T = np.transpose(temp_T)
            temp_T = temp_T[:, :, sort_idz, :]
            temp_theta = self.cal_theta(temp_T, p_data)
            u_data, v_data = self.read_uv(ds)
            
        self.theta = temp_theta
                
        # 填充速度场
        if (u_data is None) or (v_data is None):
            message = '###WARNING: u and v not found. Filled with 0 in given grid###'
            print(message)
            self.u = np.zeros_like(self.theta)
            self.v = np.zeros_like(self.theta)
        else:
            u_data = u_data[:, :, sort_idz, :]
            v_data = v_data[:, :, sort_idz, :]
            self.u = u_data
            self.v = v_data

        ds.close()
        
    def cal_lntheta(self):
        '''calculate the ln(theta)'''
        self.ln_theta = np.log(self.theta)          
        
    def ready(self, xcyclic=True):
        self.xcyclic = xcyclic
        self.cal_lntheta()

    def cal_bf_point(self, lon, lat, z, t):
        """
        在时空 (lon, lat, z, t) 上插值计算背景场及其导数, 并转换到局地直角坐标系下。
        返回在该点的一系列物理量。
        """
        lon = lon % (2*np.pi)

        notin_range_indices = np.where(np.abs(lat) > 0.5 * pi)[0]
        xo = lon  # np.array([lon])
        xo[notin_range_indices] = np.nan
        yo = lat  # np.array([lat])
        yo[notin_range_indices] = np.nan
        
        # print(self.lat)
        lnres = interp4d_grad2(
            self.lon, self.lat, self.z, self.t, self.ln_theta,
            lon, lat, z, t, self.xcyclic)
        
        ures = interp4d_grad(
            self.lon, self.lat, self.z, self.t, self.u,
            lon, lat, z, t, self.xcyclic)

        vres = interp4d_grad(
            self.lon, self.lat, self.z, self.t, self.v,
            lon, lat, z, t, self.xcyclic)

        (ln_theta,ln_theta_x,ln_theta_y,ln_theta_z,ln_theta_t,
        ln_theta_xx,ln_theta_xy,ln_theta_xz,ln_theta_xt,
        ln_theta_yx,ln_theta_yy,ln_theta_yz,ln_theta_yt,
        ln_theta_zz,ln_theta_zt) = lnres

        (u, u_x, u_y, u_z, u_t) = ures
        (v, v_x, v_y, v_z, v_t) = vres
        
        cos_phi = np.cos(lat)
        sin_phi = np.sin(lat)
        mask = np.ones(cos_phi.shape, dtype=self.all_dtype_)
        mask[np.abs(cos_phi) <= 0.0175] = 0
        cos_phi = cos_phi * mask + (1 - mask) * 1e-6
        
        ln_theta_x = ln_theta_x * mask / cos_phi
        ln_theta_y = ln_theta_y * mask
        ln_theta_z = ln_theta_z * mask
        ln_theta_t = ln_theta_t * mask
        ln_theta_xx = ln_theta_xx * mask / cos_phi / cos_phi
        ln_theta_xy = (ln_theta_xy  / cos_phi + ln_theta_x * sin_phi /cos_phi/cos_phi )* mask
        ln_theta_xz = ln_theta_xz * mask / cos_phi
        ln_theta_xt = ln_theta_xt * mask / cos_phi
        ln_theta_yx = ln_theta_yx * mask / cos_phi
        ln_theta_yy = ln_theta_yy * mask
        ln_theta_yz = ln_theta_yz * mask
        ln_theta_yt = ln_theta_yt * mask
        ln_theta_zz = ln_theta_zz * mask
        ln_theta_zt = ln_theta_zt * mask
        u = u * mask; v = v * mask
        u_x = u_x * mask / cos_phi
        v_x = v_x * mask / cos_phi
        u_y = u_y * mask
        v_y = v_y * mask
        u_z = u_z * mask
        v_z = v_z * mask
        u_t = u_t * mask
        v_t = v_t * mask
        
        f = 2 * omega * sin_phi

        return np.array([
            ln_theta_x, ln_theta_y, ln_theta_z, ln_theta_t,
            ln_theta_xx, ln_theta_xy, ln_theta_xz, ln_theta_xt,
            ln_theta_yx, ln_theta_yy, ln_theta_yz, ln_theta_yt,
            ln_theta_zz, ln_theta_zt,
            f, u, v,
            u_x, v_x, u_y, v_y,
            u_z, v_z, u_t, v_t
        ], dtype=self.all_dtype_)
        
    def cal_bf_point_initial(self, lon, lat, z, t):
        lon = lon % (2*np.pi)
        lnres = interp4d_grad(
            self.lon, self.lat, self.z, self.t, self.ln_theta,
            lon, lat, z, t, self.xcyclic)

        (ln_theta, ln_theta_x, ln_theta_y, ln_theta_z, ln_theta_t) = lnres

        u = interp4d_value(
            self.lon, self.lat, self.z, self.t, self.u,
            lon, lat, z, t, self.xcyclic)
        
        v = interp4d_value(
            self.lon, self.lat, self.z, self.t, self.v,
            lon, lat, z, t, self.xcyclic)

        cos_phi = np.cos(lat)
        sin_phi = np.sin(lat)
        mask = np.ones(cos_phi.shape, dtype=self.all_dtype_)
        mask[np.abs(cos_phi) <= 0.0175] = 0
        cos_phi = cos_phi * mask + (1 - mask) * 1e-6
        
        ln_theta_x = ln_theta_x * mask / cos_phi
        ln_theta_y = ln_theta_y * mask
        ln_theta_z = ln_theta_z * mask
        u = u * mask
        v = v * mask
        
        f = 2 * omega * sin_phi
        
        return np.array([ln_theta_x, ln_theta_y, ln_theta_z, u, v, f], dtype=self.all_dtype_)

def change_roots_order(kz, deg):
    '''
    一元二次方程根排序与筛选函数    
    kz: 根数组, 包含两个根
    deg: 根的数量 (对于二次方程应为2)
    '''
    if deg == 2:
        # 确保有两个有效根
        if not np.isnan(kz[0]) and not np.isnan(kz[1]):
            # 将正根放在前面
            if kz[0] < 0 and kz[1] >= 0:
                kz[0], kz[1] = kz[1], kz[0]
            
            # 如果两个根都是正的, 将较小的放在前面
            elif kz[0] >= 0 and kz[1] >= 0 and kz[0] > kz[1]:
                kz[0], kz[1] = kz[1], kz[0]
            
            # 如果两个根都是负的, 将较大的（绝对值较小的）放在前面
            elif kz[0] < 0 and kz[1] < 0 and kz[0] < kz[1]:
                kz[0], kz[1] = kz[1], kz[0]
    
    # 检查根的有效性
    # for i in range(2):
    #     if not np.isnan(kz[i]) and abs(kz[i]) > 100.:
    #         kz[i] = np.nan
    #         deg = deg - 1
    
    return kz, deg
        
def cal_kz(ln_theta_x, ln_theta_y, ln_theta_z, f, freq_intrinsic, zwn, mwn):
    """
    计算给定背景场参数下的传播的垂向波数 m (2个解)。
    Input:
        ln_theta_x, ln_theta_y, ln_theta_z: ln(theta)的三维梯度
        f: 科氏参数 
        freq     - 波的频率 (rad/s)
        zwn      - 初始的无量纲 zonal wave number (k * Rearth)
        mwn      - 初始的无量纲 meridional wave number (l * Rearth)
    Output:
        kz_list - 垂向波数数组
        len(real_roots) - 根的实部

    shape = (points,) + (original.shape) for fu,fv,fqx,fqy
    freq, zwn: float or int
    """
    
    # print('ln_theta_x = ', ln_theta_x)
    # print('ln_theta_y = ', ln_theta_y)
    # print('ln_theta_z = ', ln_theta_z)
    # print('f = ', f)
    # print('freq_intrinsic = ', freq_intrinsic)
    # print('zwn = ', zwn)
    # print('mwn = ', mwn)
    
    kz_list = np.ones((len(ln_theta_x), 2)) * np.nan
    lens = np.ones((len(ln_theta_x))) - 1
    
    coeff_ = np.stack([freq_intrinsic**2 - f**2,
                       grav * (zwn * ln_theta_x + mwn * ln_theta_y) / rearth,
                       (zwn**2 + mwn**2) * (freq_intrinsic**2 - grav * ln_theta_z)
                       ], axis=-1)

    
    for i in range(coeff_.shape[0]):
        coeff = coeff_[i, :]
        # roots = roots_numba(coeff[::-1] + 0j) damn!
        roots = roots_numba(coeff[::] + 0j)
        vwn = roots.real
        # 虚根和重根只保留一个解
        if vwn[0]==vwn[1]:
            vwn[1] = np.nan
        # real_roots = [r.real for r in roots if abs(r.imag) < delt]
        roots_num = len(vwn)
        # vwn = np.array(real_roots[:2] + [undef] * (2 - roots_num))
        vwn, roots_num = change_roots_order(vwn, roots_num)

        # 补齐为2个根
        kz_list[i, :] = vwn
        lens[i] = roots_num
        
    return kz_list, lens
            