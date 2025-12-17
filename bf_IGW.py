import numpy as np
from constants import pi, undef, H0, grav, omega, p0, R, cp, rearth
import netCDF4 as nc
from netCDF4 import Dataset
from interpolation import batch_linint4
from scipy.ndimage import convolve
import numba as nb

'''
奥卡姆剃刀原理：如无必要，勿增实体
Occam's Razor: Entities should not be multiplied unnecessarily
Numquam ponenda est pluralitas sine necessitate
'''

nb_para_dic = {
    'nopython': True,
    # 'fastmath':True,
    'cache': True,
}


# @nb.jit(['c16[:](c16[:])'], **nb_para_dic)  # ,'c8[:](c8[:])'
@nb.jit(nopython=True, cache=True, fastmath=True)
def roots_numba(p):
    return np.roots(p)

class BF:
    def __init__(self, nx, ny, nz, nt, dt, read_dtype='float32', cal_dtype='float64'):
        self.all_dtype = read_dtype
        self.all_dtype_ = cal_dtype
        # pi, rearth, omega, undef, delt = np.array([pi, rearth, omega, undef, delt],dtype=cal_dtype)
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nt = nt
        shape = (nx, ny, nz, nt)
        self.dx = np.array([2.0 * pi / self.nx], dtype=self.all_dtype_)
        self.dy = np.array([pi / (self.ny - 1)], dtype=self.all_dtype_)
        self.dt = dt

        # 
        self.theta = np.zeros(shape, dtype=self.all_dtype)
        self.ln_theta = np.zeros(shape, dtype=self.all_dtype)
        self.lon = np.zeros(nx, dtype=self.all_dtype_)
        self.lat = np.zeros(ny, dtype=self.all_dtype_)
        self.p = np.zeros(nz, dtype=self.all_dtype_)
        self.z = np.zeros(nz, dtype=self.all_dtype_)
        # 一阶导数
        self.ln_theta_x = np.zeros(shape, dtype=self.all_dtype_)
        self.ln_theta_y = np.zeros(shape, dtype=self.all_dtype_)
        self.ln_theta_z = np.zeros(shape, dtype=self.all_dtype_)
        # 二阶导数
        self.ln_theta_xx = np.zeros(shape, dtype=self.all_dtype_)
        self.ln_theta_xy = np.zeros(shape, dtype=self.all_dtype_)
        self.ln_theta_xz = np.zeros(shape, dtype=self.all_dtype_)
        self.ln_theta_xt = np.zeros(shape, dtype=self.all_dtype_)
        self.ln_theta_yx = np.zeros(shape, dtype=self.all_dtype_)
        self.ln_theta_yy = np.zeros(shape, dtype=self.all_dtype_)
        self.ln_theta_yz = np.zeros(shape, dtype=self.all_dtype_)
        self.ln_theta_yt = np.zeros(shape, dtype=self.all_dtype_)
        self.ln_theta_zx = np.zeros(shape, dtype=self.all_dtype_)
        self.ln_theta_zy = np.zeros(shape, dtype=self.all_dtype_)
        self.ln_theta_zz = np.zeros(shape, dtype=self.all_dtype_)
        self.ln_theta_zt = np.zeros(shape, dtype=self.all_dtype_)
      
        
    # 外部调用，统一接口
    def getlon(self):
        return self.lon

    def getlat(self):
        return self.lat
    
    def getlevel(self):
        return self.p
    
    def gett(self):
        return self.t

    def gradient_x(self, f_):
        """
        计算纬向的一阶偏导
        (\partial f)/(\partial lambda}
        对经度环向做周期处理
        """
        f = f_.astype(self.all_dtype_)
        fx = np.empty_like(f, dtype=self.all_dtype_)
        # 中部点用中心差分
        fx[1:-1, :, :, :] = (f[2::, :, :, :] - f[0:-2, :, :, :]) / (2.0 * self.dx)
        # 周期边界: 首列和末列
        fx[0, :, :, :] = (f[1, :, :, :] - f[-1, :, :, :]) / (2.0 * self.dx)
        fx[-1, :, :, :] = (f[0, :, :, :] - f[-2, :, :, :]) / (2.0 * self.dx)
        # print('fx_dtype: ',fx.dtype)
        return fx  # .astype(self.all_dtype_)

    def gradient_y(self, f_):
        """
        计算经向的一阶偏导 
        (\partial f)/(\partial phi)
        纬度方向非周期，边界用单侧差分
        """
        f = f_.astype(self.all_dtype_)
        fy = np.empty_like(f, dtype=self.all_dtype_)
        # 中部点中心差分
        fy[:, 1:-1, :, :] = (f[:, 2::, :, :] - f[:, 0:-2, :, :]) / (2.0 * self.dy)
        # 南北边界: 用前向/后向差分近似
        fy[:, 0, :, :] = (f[:, 1, :, :] - f[:, 0, :, :]) / (self.dy)
        fy[:, -1, :, :] = (f[:, -1, :, :] - f[:, -2, :, :]) / (self.dy)
        return fy  # .astype(self.all_dtype_)
    
    def gradient_z(self, f_):
        """计算垂向的一阶导数
        (\partial f)/(\partial z)
        非均匀垂向坐标。
        参数:
            f_: 输入场，形状为 (lon, lat, z, time)
        返回:
            fz: 垂向梯度，形状与输入相同
        """
        f = f_.astype(self.all_dtype_)
        fz = np.empty_like(f, dtype=self.all_dtype_)
        
        # 获取垂向坐标
        z = self.z  # self.z 是一维数组，表示各层的垂向坐标
        # print(f'z = {z}')
        # 中部点：使用中心差分
        # 对于非均匀网格，使用实际坐标差计算导数
        for k in range(1, f.shape[-2] - 1):  # 垂向维度是倒数第二个维度
            dz_forward = z[k+1] - z[k]      # 前向差分步长
            dz_backward = z[k] - z[k-1]     # 后向差分步长
            # print(f'k = {k}')
            # print(f'z[k+1] = {z[k+1]}  z[k] = {z[k]}')
            # print(f'dz_forward = {dz_forward}')
            # print(f'dz_backward = {dz_backward}')
            # 非均匀网格的中心差分公式
            # 使用加权平均，考虑前后网格间距不同
            if dz_forward + dz_backward > 0:  # 避免除零
                fz[..., k, :] = (
                    (f[..., k+1, :] - f[..., k, :]) * dz_backward / (dz_forward * (dz_forward + dz_backward)) +
                    (f[..., k, :] - f[..., k-1, :]) * dz_forward / (dz_backward * (dz_forward + dz_backward))
                ) 
            else:
                fz[..., k, :] = 0
        
        # 上边界（顶部）：使用前向差分
        if f.shape[-2] > 1:  # 确保至少有两层
            dz_top = z[1] - z[0]
            if dz_top > 0:
                fz[..., 0, :] = (f[..., 1, :] - f[..., 0, :]) / dz_top
            else:
                fz[..., 0, :] = 0
        
        # 下边界（底部）：使用后向差分
        if f.shape[-2] > 1:  # 确保至少有两层
            dz_bottom = z[-1] - z[-2]
            if dz_bottom > 0:
                fz[..., -1, :] = (f[..., -1, :] - f[..., -2, :]) / dz_bottom
            else:
                fz[..., -1, :] = 0
        
        return fz
    
    def gradient_t(self, f_):
        """
        计算时间的一阶偏导
        (\partial f)/(\partial t) *d(t)
        时间边界用单侧差分
        """
        f = f_.astype(self.all_dtype_)
        ft = np.empty_like(f, dtype=self.all_dtype_)
        # 中部点中心差分
        ft[:, :, :, 1:-1] = (f[:, :, :, 2::] - f[:, :, :, 0:-2]) / (2.0 * self.dt)
        # 南北边界: 用前向/后向差分近似
        ft[:, :, :, 0] = (f[:, :, :, 1] - f[:, :, :, 0]) / (self.dt)
        ft[:, :, :, -1] = (f[:, :, :, -1] - f[:, :, :, -2]) / (self.dt)
        return ft  # .astype(self.all_dtype_)

    def gradient_xx(self, f_):
        """
        计算纬向的二阶导数
        (\partial^2 f)/(\partial lambda^2)
        """
        f = f_.astype(self.all_dtype_)
        fxx = np.empty_like(f, dtype=self.all_dtype_)
        # 中部点中心差分近似二阶导
        fxx[1:-1, :, :, :] = (f[2::, :, :, :] - 2.0 * f[1:-1, :, :, :] +
                        f[0:-2, :, :, :]) / (self.dx**2)
        # 边界点
        fxx[0, :, :, :] = (f[1, :, :, :] - 2.0 * f[0, :, :, :] + f[-1, :, :, :]) / (self.dx**2)
        fxx[-1, :, :, :] = (f[0, :, :, :] - 2.0 * f[-1, :, :, :] + f[-2, :, :, :]) / (self.dx**2)
        return fxx  # .astype(self.all_dtype_)

    def gradient_xy(self, f):
        """
        计算二阶混合导数
        (\partial^2 f)/(\partial phi \partial lambda)
        先纬向后经向
        """
        fxy = np.empty_like(f, dtype=self.all_dtype_)
        # 中心区域使用四点差分计算混合偏导
        # for i in range(1, nx-1):
        #     for j in range(1, ny-1):
        #         fxy[i, j] = (f[i+1, j+1] - f[i+1, j-1] - f[i-1, j+1] + f[i-1, j-1]) / (4.0 * self.dx * self.dy)
        fxy[1:-1, 1:-1, :, :] = (f[2::, 2::, :, :] - f[2::, 0:-2, :, :] - f[0:-2, 2::, :, :] + f[0:-2, 0:-2, :, :]) / (4.0 * self.dx * self.dy)

        fxy[1:-1, 0, :, :] = fxy[1:-1, 1, :, :]
        fxy[1:-1, -1, :, :] = fxy[1:-1, -2, :, :]

        fxy[0, 1:-1, :, :] = (f[1, 2::, :, :] - f[1, 0:-2, :, :] - f[-1, 2::, :, :] + f[-1, 0:-2, :, :]) / (4 * self.dx * self.dy)
        fxy[-1, 1:-1, :, :] = (f[0, 2::, :, :] - f[0, 0:-2, :, :] - f[-2, 2::, :, :] + f[-2, 0:-2, :, :]) / (4 * self.dx * self.dy)

        fxy[0, 0, :, :] = fxy[0, 1, :, :]
        fxy[0, -1, :, :] = fxy[0, -2, :, :]
        fxy[-1, 0, :, :] = fxy[-1, 1, :, :]
        fxy[-1, -1, :, :] = fxy[-1, -2, :, :]
        return fxy  # .astype(self.all_dtype_)
    
    def gradient_xz(self, f_):
        """
        计算二阶混合导数
        (\partial^2 f)/(\partial z \partial lambda)  * rearth
        先纬向后垂向
        """
        f = f_.astype(self.all_dtype_)
        fxz = np.empty_like(f, dtype=self.all_dtype_)
        
        fx = self.gradient_x(f)
        fxz = self.gradient_z(fx)
        return fxz
    
    def gradient_xt(self, f_):
        """
        计算二阶混合导数
        (\partial^2 f)/(\partial t \partial lambda)
        先纬向后时间
        """
        f = f_.astype(self.all_dtype_)
        fxt = np.empty_like(f, dtype=self.all_dtype_)
        
        fx = self.gradient_x(f)
        fxt = self.gradient_t(fx)
        return fxt

    def gradient_yx(self, f):
        """
        计算二阶混合导数
        (\partial^2 f)/(\partial phi \partial lambda)
        先经向后纬向
        """
        # 由于连续偏导次序可交换，这里直接调用 gradient_xy 实现
        return self.gradient_xy(f)
    
    def gradient_yy(self, f_):
        """
        计算经向的二阶导数
        (\partial^2 f)/(\partial phi^2)
        """
        f = f_.astype(self.all_dtype_)
        fyy = np.empty_like(f, dtype=self.all_dtype_)
        # 中部点中心差分
        fyy[:, 1:-1, :, :] = (f[:, 2::, :, :] - 2.0 * f[:, 1:-1, :, :] +
                        f[:, 0:-2, :, :]) / (self.dy**2)
        # 边界: 复制相邻点的值（假设边界导数与邻格相同）
        fyy[:, 0, :, :] = fyy[:, 1, :, :]
        fyy[:, -1, :, :] = fyy[:, -2, :, :]
        return fyy

    def gradient_yz(self, f_):
        """
        计算二阶混合导数
        (\partial^2 f)/(\partial z \partial phi)
        先经向后垂向
        """
        f = f_.astype(self.all_dtype_)
        fyz = np.empty_like(f, dtype=self.all_dtype_)
        
        fy = self.gradient_y(f)
        fyz = self.gradient_z(fy)
        return fyz

    def gradient_yt(self, f_):
        """
        计算二阶混合导数
        (\partial^2 f)/(\partial t \partial phi)
        先经向后时间
        """
        f = f_.astype(self.all_dtype_)
        fyt = np.empty_like(f, dtype=self.all_dtype_)
        
        fy = self.gradient_y(f)
        fyt = self.gradient_t(fy)
        return fyt
      
    def gradient_zx(self, f):
        """
        计算二阶混合导数
        (\partial^2 f)/(\partial lambda \partial z)
        先垂向后纬向
        """
        # 由于连续偏导次序可交换，这里直接调用 gradient_xy 实现
        return self.gradient_xz(f)
    
    def gradient_zy(self, f):
        """
        计算二阶混合导数
        (\partial^2 f)/(\partial phi \partial z)
        先垂向后经向
        """
        return self.gradient_yz(f)


    def gradient_zz(self, f_):
        """
        计算垂向的二阶导数
        (\partial^2 f)/(\partial z^2) * rearth
        """
        f = f_.astype(self.all_dtype_)
        fzz = np.empty_like(f, dtype=self.all_dtype_)
        
        fz = self.gradient_z(f)
        fzz = self.gradient_z(fz)
        return fzz

    def gradient_zt(self, f_):
        """
        计算二阶混合导数
        (\partial^2 f)/(\partial t \partial z)
        先垂向后时间
        """
        f = f_.astype(self.all_dtype_)
        fzt = np.empty_like(f, dtype=self.all_dtype_)
        
        fz = self.gradient_y(f)
        fzt = self.gradient_t(fz)
        return fzt
   
    def cal_theta(self, temp, p):
        """
        计算四维温度数据的位温
        参数:
        temp: numpy数组, 四维温度数据 (x, y, p, t)，单位：K
        p: numpy数组, 气压数据，与temp的第三维对应，单位：hPa
        p0: float, 参考气压，默认1000 hPa
        R: float, 气体常数，默认287 J/(kg·K)
        cp: float, 定压比热容，默认1004 J/(kg·K)
        返回:
        theta: numpy数组, 位温数据 (x, y, p, t)，单位：K
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

     
    def loadbf_ncfile(self, ncfile):
        '''
        如果文件中存在lon和lat，使用它们做为坐标
        如果不存在lon或lat，使用维度来构建，默认-1维为lon，-2维为lat，并给出warning
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
                self.lat[:] = (
                    lat_data *
                    pi /
                    180).astype(
                    self.all_dtype_)  # 转为弧度
                break
            
        for name in lon_candidates:
            if name in ds.variables:
                lon_data = np.array(
                    ds.variables[name][:],
                    dtype=self.all_dtype)
                self.lon[:] = (lon_data * pi / 180).astype(self.all_dtype_)
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
            self.lat = - pi * 0.5 + np.arange(self.ny) * self.dy
        if lon_data is None:
            self.lon = np.arange(self.nx) * self.dx
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
        # 注意：由于Python读取数组相对于原数组是反的，所以需要转置
        if 'theta' in ds.variables:
            temp_theta = np.array(ds.variables['theta'][:], dtype=self.all_dtype)
            temp_theta = np.transpose(temp_theta)
            temp_theta = temp_theta[:, :, sort_idz, :]
        elif 'T' in ds.variables:
            temp_T = np.array(ds.variables['T'][:], dtype=self.all_dtype)
            temp_T = np.transpose(temp_T)
            temp_T = temp_T[:, :, sort_idz, :]
            temp_theta = self.cal_theta(temp_T, p_data)
        elif 'air' in ds.variables:
            temp_T = np.array(ds.variables['air'][:], dtype=self.all_dtype)
            temp_T = np.transpose(temp_T)
            temp_T = temp_T[:, :, sort_idz, :]
            temp_theta = self.cal_theta(temp_T, p_data)
            
        self.theta = temp_theta
        
        # 确保纬度递增
        if (lat_data is None) or (lon_data is None):
            message = '###WARNING: lon and lat not found. Make sure your lats are from 90S to 90N and lons are from 0E to 360E###'
            print(message)
        elif not (lat_data is None):
            if lat_data[0] > lat_data[-1]:
                lat_data = lat_data[::-1]
                self.theta = temp_theta[:, ::-1, :, :]
        ds.close()
        
    def cal_lntheta(self):
        '''calculate the ln(theta)'''
        
        self.ln_theta = np.log(self.theta)    
        
    # 20250718已验证：Python和Fortran的smth9实现计算方式、权重定义、边界处理、更新逻辑完全一致，无计算差异。by: Ms. Y. N. Yang
    def smth9(self, field, p=0.5, q=0.25):
        """
        九点加权平滑器，高效 NumPy 卷积实现。
        p: 十字方向（上下左右）权重系数
        q: 角点方向（四角）权重系数
        """
        kernel = np.array([
            [q / 4, p / 4, q / 4],
            [p / 4, -(p + q), p / 4],
            [q / 4, p / 4, q / 4]
        ])
        smooth = field
        for i in range(0, smooth[0,0,:].size):
            smooth[1:-2, 1:-2, i] = smooth[1:-2, 1:-2, i] + \
                convolve(field[:, :, i], kernel, mode='constant', cval=0.0)[1:-2, 1:-2]
        return smooth        
        
    def ready(self, xcyclic=False):
        self.xcyclic = xcyclic
        self.cal_lntheta()
        self.ln_theta_x = self.gradient_x(self.ln_theta)
        self.ln_theta_y = self.gradient_y(self.ln_theta)
        self.ln_theta_z = self.gradient_z(self.ln_theta)
        self.ln_theta_t = self.gradient_t(self.ln_theta)
        
        self.ln_theta_xx = self.gradient_xx(self.ln_theta)
        self.ln_theta_xy = self.gradient_xy(self.ln_theta)
        self.ln_theta_xz = self.gradient_xz(self.ln_theta)
        self.ln_theta_xt = self.gradient_xt(self.ln_theta)
        
        self.ln_theta_yx = self.gradient_yx(self.ln_theta)
        self.ln_theta_yy = self.gradient_yy(self.ln_theta)
        self.ln_theta_yz = self.gradient_yz(self.ln_theta)
        self.ln_theta_yt = self.gradient_yt(self.ln_theta)
        
        self.ln_theta_zx = self.gradient_zx(self.ln_theta)
        self.ln_theta_zy = self.gradient_zy(self.ln_theta)
        self.ln_theta_zz = self.gradient_zz(self.ln_theta)
        self.ln_theta_zt = self.gradient_zt(self.ln_theta)
        


        self.fields = np.stack([
            self.ln_theta_x,
            self.ln_theta_y,
            self.ln_theta_z,
            self.ln_theta_t,
            self.ln_theta_xx,
            self.ln_theta_xy,
            self.ln_theta_xz,
            self.ln_theta_xt,
            self.ln_theta_yx,
            self.ln_theta_yy,
            self.ln_theta_yz,
            self.ln_theta_yt,
            self.ln_theta_zx,
            self.ln_theta_zy,
            self.ln_theta_zz,
            self.ln_theta_zt,
        ], axis=-1)
        self.fields = self.fields.astype(self.all_dtype_)
        if self.xcyclic:
            self.fields = np.concatenate(
                [self.fields, self.fields[0:1, :, :, :]], axis=0)

    def output(self, ncfile):
        """
        将基本流场输出为 NetCDF4 文件，支持压缩、变量自动写入。
        """
        output_type = self.all_dtype_
        with Dataset(ncfile, 'w', format='NETCDF4') as ds:
            # 创建维度
            ds.createDimension('lon', self.nx)
            ds.createDimension('lat', self.ny)
            ds.createDimension('level', self.nz)
            ds.createDimension('t', self.nt)

            # 写入经纬度
            for name, data in zip(
                    ['lon', 'lat', 'level', 't'], [self.getlon(), self.getlat(), self.getlevel(), self.gett()]):
                var = ds.createVariable(name, output_type, (name,))
                var[:] = data
                var.units = 'degrees_east' if name == 'lon' else 'degrees_north' if name == 't' else 'second'

            # 统一写入4维变量

            field_map = {
                'theta': (self.theta, 'K'),
                'ln_theta': (self.ln_theta, 'None'),
                'ln_theta_x': (self.ln_theta_x, 'None'),
                'ln_theta_y': (self.ln_theta_y, 'None'),
                'ln_theta_z': (self.ln_theta_z, 'None'),
                'ln_theta_t': (self.ln_theta_t, 'None'),
                'ln_theta_xx': (self.ln_theta_xx, 'None'),
                'ln_theta_xy': (self.ln_theta_xy, 'None'),
                'ln_theta_xz': (self.ln_theta_xz, 'None'),
                'ln_theta_xt': (self.ln_theta_xt, 'None'),
                'ln_theta_yx': (self.ln_theta_yx, 'None'),
                'ln_theta_yy': (self.ln_theta_yy, 'None'),
                'ln_theta_yz': (self.ln_theta_yz, 'None'),
                'ln_theta_yt': (self.ln_theta_yt, 'None'),
                'ln_theta_zx': (self.ln_theta_zx, 'None'),
                'ln_theta_zy': (self.ln_theta_zy, 'None'),
                'ln_theta_zz': (self.ln_theta_zz, 'None'),
                'ln_theta_zt': (self.ln_theta_zt, 'None'),
            }

            for name, (data, unit) in field_map.items():
                var = ds.createVariable(
                    name, output_type, ('lon', 'lat', 'level', 't'), zlib=True, complevel=4)
                var[:, :, :, :] = data
                var.units = unit
        
    def cal_bf_point(self, lon, lat, zo, to):
        """
        在时空 (lon, lat, z, t) 上插值计算基本流场及其导数，并转换到局地直角坐标系下。
        返回在该点的一系列物理量。
        """
        lon = lon % (2 * pi)
        wrapX = True
        
        # in_range_indices = np.where(np.abs(lat) <= 0.5 * pi)[0]
        notin_range_indices = np.where(np.abs(lat) > 0.5 * pi)[0]
        
        xo = lon  # np.array([lon])
        xo[notin_range_indices] = np.nan
        # print('lon = ', lon)
        
        yo = lat  # np.array([lat])
        yo[notin_range_indices] = np.nan
        # print('lat = ', lat)
        
        

        interp_fields_ = np.ones(
            (self.fields.shape[-1], len(lat))) * np.nan
        interp_fields = batch_linint4(
            self.lon,
            self.lat,
            self.z,
            self.t,
            self.fields,
            xo,
            yo,
            zo,
            to,
            fo_missing=undef,
            xcyclic=wrapX,
            all_dtype=self.all_dtype_)
        
        interp_fields_[:, :] = np.transpose(interp_fields, (1, 0))
                
        (ln_theta_x,
         ln_theta_y,
         ln_theta_z,
         ln_theta_t,
         ln_theta_xx,
         ln_theta_xy,
         ln_theta_xz,
         ln_theta_xt,
         ln_theta_yx,
         ln_theta_yy,
         ln_theta_yz,
         ln_theta_yt,
         ln_theta_zx,
         ln_theta_zy,
         ln_theta_zz,
         ln_theta_zt) = interp_fields_
        
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
        ln_theta_zx = ln_theta_zx * mask / cos_phi
        ln_theta_zy = ln_theta_zy * mask
        ln_theta_zz = ln_theta_zz * mask
        ln_theta_zt = ln_theta_zt * mask
        
        f = 2 * omega * sin_phi

        return np.array([ln_theta_x, ln_theta_y, ln_theta_z, ln_theta_t,
                         ln_theta_xx, ln_theta_xy, ln_theta_xz, ln_theta_xt,
                         ln_theta_yx, ln_theta_yy, ln_theta_yz, ln_theta_yt,
                         ln_theta_zx, ln_theta_zy, ln_theta_zz, ln_theta_zt, f]
                        , dtype=self.all_dtype_)

def change_roots_order(kz, deg):
    '''
    一元二次方程根排序与筛选函数    
    kz: 根数组，包含两个根
    deg: 根的数量（对于二次方程应为2）
    '''
    if deg == 2:
        # 确保有两个有效根
        if not np.isnan(kz[0]) and not np.isnan(kz[1]):
            # 将正根放在前面
            if kz[0] < 0 and kz[1] >= 0:
                kz[0], kz[1] = kz[1], kz[0]
            
            # 如果两个根都是正的，将较小的放在前面
            elif kz[0] >= 0 and kz[1] >= 0 and kz[0] > kz[1]:
                kz[0], kz[1] = kz[1], kz[0]
            
            # 如果两个根都是负的，将较大的（绝对值较小的）放在前面
            elif kz[0] < 0 and kz[1] < 0 and kz[0] < kz[1]:
                kz[0], kz[1] = kz[1], kz[0]
    
    # 检查根的有效性
    # for i in range(2):
    #     if not np.isnan(kz[i]) and abs(kz[i]) > 100.:
    #         kz[i] = np.nan
    #         deg = deg - 1
    
    return kz, deg
        
def cal_kz(ln_theta_x, ln_theta_y, ln_theta_z, f, freq, zwn, mwn):
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
    kz_list = np.ones((len(ln_theta_x), 2)) * np.nan
    lens = np.ones((len(ln_theta_x))) - 1
    # if (freq**2-f**2) != 0:
    # 构建多项式系数（按 m^2 -> m^0 排列）
    # print('a2 = ',freq**2-f**2)
    # print('a1 = ',grav*(zwn * ln_theta_x + mwn * ln_theta_y) / rearth)
    # print('a0 = ',(zwn**2 + mwn**2) * (freq**2 - grav * ln_theta_z))
    # print('zwn = ',zwn)
    # print('mwn = ',mwn)
    # print('f = ',f)
    # print('freq = ',freq)
    # print('ln_theta_x = ',ln_theta_x)
    # print('ln_theta_y = ',ln_theta_y)
    # print('ln_theta_z = ',ln_theta_z)
    
    coeff_ = np.stack([freq**2-f**2,
                       grav*(zwn * ln_theta_x + mwn * ln_theta_y) / rearth,
                       (zwn**2 + mwn**2) * (freq**2 - grav * ln_theta_z)
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
            