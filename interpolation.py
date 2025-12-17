# interpolation.py - 线性插值模块
import numpy as np

def batch_linint4(xi, yi, zi, ti, fi, xo, yo, zo, to, fo_missing=np.nan,
                        xcyclic=False, all_dtype='float64'):
    """
    批量四线性插值函数：对多个点 (xo, yo, zo, to) 在四维网格 fi 上插值。
    
    参数:
        xi, yi, zi, ti - 一维网格坐标 (经度、纬度、对数气压、时间)
        fi          - 4D 数据，形状 (len(xi), len(yi), len(zi), len(ti))
        xo, yo, zo, to - 多个插值点，一维数组
        fo_missing  - 缺测值
        xcyclic     - 是否对 xi 做周期处理
        all_dtype   - 数据类型
    
    返回:
        values      - 插值后的值，形状 (len(xo),)
    """
    
    # print('xo = ', xo)
    # print('yo = ', yo)
    # print('zo = ', zo)
    # print('to = ', to)

    # 计算网格间距（x, y, t方向假设为均匀网格）
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    dt = ti[1] - ti[0]
    
    # 计算归一化索引（x, y, t方向）
    ilons = (xo - xi[0]) / dx
    # print(ilons)
    ilats = (yo - yi[0]) / dy
    itimes = (to - ti[0]) / dt
    
    # 对于非均匀的z方向，我们直接使用实际坐标值
    # 不需要计算归一化索引
    
    # 将fi重塑为 (1, nx, ny, nz, nt) 形状以适应四线性插值函数
    fi_reshaped = fi[None, :, :, :, :]
    
    # 使用支持非均匀垂向坐标的插值函数
    return quadlinear_interpolation(
        fi_reshaped, zi, ilons, ilats, zo, itimes, xcyclic=xcyclic, all_dtype=all_dtype)

def get_pixel_value_4d(imgs, x, y, z, t):
    '''
    从四维场中获取指定位置的值
    
    Parameters:
    -----------
    imgs: 四维场数据，shape=(batch, lon, lat, ln(pressure), time, vars)
    x: 经度索引，整数数组
    y: 纬度索引，整数数组
    z: 对数气压层索引，整数数组
    t: 时间索引，整数数组
    '''
    
    return imgs[0, x, y, z, t, :]

def quadlinear_interpolation(imgs, z_coords, x, y, z, t, xcyclic=True, all_dtype='float64'):
    '''
    四维场线性插值函数，支持非均匀垂向坐标
    Parameters:
    -----------
    imgs: 四维场数据，shape=(batch, lon, lat, pressure, time, vars)
    z_coords: 对数气压层的参考坐标
    x: 经度索引，浮点数数组
    y: 纬度索引，浮点数数组
    z: 对数气压层索引，浮点数数组
    t: 时间索引，浮点数数组
    xcyclic: 是否在经度方向使用循环边界条件
    all_dtype: 输出数据类型
    
    Returns:
    --------
    inter_img: 插值后的值，shape = (npoints, vars)
    
    Written by DeepSeek 2025.11.17 
    Edited  by DeepSeek 2025.12.11
    '''

    width, height, depth, duration = imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4]
    
    # 创建有效点掩码，标记非NaN的点
    valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    
    # 初始化输出数组
    n_points = len(x)
    n_vars = imgs.shape[-1] if len(imgs.shape) > 5 else 1
    inter_img = np.full((n_points, n_vars), np.nan, dtype=all_dtype)
    
    # 如果没有有效点，直接返回
    if not np.any(valid_mask):
        return inter_img
    
    # 只处理有效点
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    z_valid = z[valid_mask]
    
    
    # 判断z是否超出z_coords范围
    z_min = np.min(z_coords)
    z_max = np.max(z_coords)
    
    # 判断哪些点在z_coords范围内
    z_in_range_mask = (z_valid >= z_min) & (z_valid <= z_max)
    # Debug用
    # z_in_range_mask = (z_valid >= min(z_valid)) & (z_valid <= max(z_valid))

    # 将超出范围的点结果设为0
    inter_img_valid = np.zeros((len(z_valid), n_vars), dtype=all_dtype)
    inter_img_valid[:] = np.nan
    
    # 如果没有点在范围内，直接返回全0
    if not np.any(z_in_range_mask):
        inter_img[valid_mask] = inter_img_valid
        return inter_img.astype(all_dtype)
    
    # 提取范围内的点 只对范围内的点进行插值
    x_in_range = x_valid[z_in_range_mask]
    y_in_range = y_valid[z_in_range_mask]
    z_in_range = z_valid[z_in_range_mask]
    
    
    # 找到目标点周围的16个角点
    x0 = np.floor(x_in_range).astype('int32')
    x1 = x0 + 1
    y0 = np.floor(y_in_range).astype('int32')
    y1 = y0 + 1
    t0 = np.floor(t).astype('int32')
    t1 = t0 + 1
    
    # 对于非均匀的z方向，找到包含目标点的区间
    # 使用searchsorted找到z在z_coords中的位置
    z_indices = np.searchsorted(z_coords, z_in_range)
    z0 = np.clip(z_indices - 1, 0, depth - 1)
    z1 = np.clip(z_indices, 0, depth - 1)
    z1 = np.where(z1 == z0, np.clip(z0 + 1, 0, depth - 1), z1)
    
    # 确保索引在有效范围内
    if xcyclic:
        x0 = x0 % width
        x1 = x1 % width
    else:
        x0 = np.clip(x0, 0, width - 1)
        x1 = np.clip(x1, 0, width - 1)
        
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)
    t0 = np.clip(t0, 0, duration - 1)
    t1 = np.clip(t1, 0, duration - 1)
    
    # 获取16个角点的值
    c0000 = get_pixel_value_4d(imgs, x0, y0, z0, t0)
    c0001 = get_pixel_value_4d(imgs, x0, y0, z0, t1)
    c0010 = get_pixel_value_4d(imgs, x0, y0, z1, t0)
    c0011 = get_pixel_value_4d(imgs, x0, y0, z1, t1)
    c0100 = get_pixel_value_4d(imgs, x0, y1, z0, t0)
    c0101 = get_pixel_value_4d(imgs, x0, y1, z0, t1)
    c0110 = get_pixel_value_4d(imgs, x0, y1, z1, t0)
    c0111 = get_pixel_value_4d(imgs, x0, y1, z1, t1)
    c1000 = get_pixel_value_4d(imgs, x1, y0, z0, t0)
    c1001 = get_pixel_value_4d(imgs, x1, y0, z0, t1)
    c1010 = get_pixel_value_4d(imgs, x1, y0, z1, t0)
    c1011 = get_pixel_value_4d(imgs, x1, y0, z1, t1)
    c1100 = get_pixel_value_4d(imgs, x1, y1, z0, t0)
    c1101 = get_pixel_value_4d(imgs, x1, y1, z0, t1)
    c1110 = get_pixel_value_4d(imgs, x1, y1, z1, t0)
    c1111 = get_pixel_value_4d(imgs, x1, y1, z1, t1)
    
    # 计算相对距离
    xd = (x_in_range - x0)
    yd = (y_in_range - y0)
    td = (t - t0)
    
    # z方向使用非均匀坐标的相对距离
    z_coords_z0 = z_coords[z0]
    z_coords_z1 = z_coords[z1]
    
    denominator = z_coords_z1 - z_coords_z0
    zd = np.zeros_like(z_in_range, dtype=float)
    mask_nonzero = denominator != 0
    zd[mask_nonzero] = (z_in_range[mask_nonzero] - z_coords_z0[mask_nonzero]) / denominator[mask_nonzero]
    zd = np.clip(zd, 0, 1)
    
    # 计算权重
    w0000 = (1 - xd) * (1 - yd) * (1 - zd) * (1 - td)
    w0001 = (1 - xd) * (1 - yd) * (1 - zd) * td
    w0010 = (1 - xd) * (1 - yd) * zd * (1 - td)
    w0011 = (1 - xd) * (1 - yd) * zd * td
    w0100 = (1 - xd) * yd * (1 - zd) * (1 - td)
    w0101 = (1 - xd) * yd * (1 - zd) * td
    w0110 = (1 - xd) * yd * zd * (1 - td)
    w0111 = (1 - xd) * yd * zd * td
    w1000 = xd * (1 - yd) * (1 - zd) * (1 - td)
    w1001 = xd * (1 - yd) * (1 - zd) * td
    w1010 = xd * (1 - yd) * zd * (1 - td)
    w1011 = xd * (1 - yd) * zd * td
    w1100 = xd * yd * (1 - zd) * (1 - td)
    w1101 = xd * yd * (1 - zd) * td
    w1110 = xd * yd * zd * (1 - td)
    w1111 = xd * yd * zd * td
    
    # 加权求和
    inter_img_range = (c0000 * w0000[:, None] + c0001 * w0001[:, None] + 
                      c0010 * w0010[:, None] + c0011 * w0011[:, None] + 
                      c0100 * w0100[:, None] + c0101 * w0101[:, None] + 
                      c0110 * w0110[:, None] + c0111 * w0111[:, None] + 
                      c1000 * w1000[:, None] + c1001 * w1001[:, None] + 
                      c1010 * w1010[:, None] + c1011 * w1011[:, None] + 
                      c1100 * w1100[:, None] + c1101 * w1101[:, None] + 
                      c1110 * w1110[:, None] + c1111 * w1111[:, None])
    
    # 将范围内的点插值结果放入inter_img_valid
    inter_img_valid[z_in_range_mask] = inter_img_range.astype(all_dtype)
    
    # 将有效点的结果放回正确位置
    inter_img[valid_mask] = inter_img_valid
    
    return inter_img.astype(all_dtype)

def linint4_point(xi, yi, zi, ti, fi, xcyclic, xo, yo, zo, to,
                  fo_missing=np.nan, nopt=1):
    """
    四线性插值计算单点值
    Parameter:
      xi[0..nxi-1] - 自变量x坐标数组 (如经度), 要求单调递增
      yi[0..nyi-1] - 自变量y坐标数组 (如纬度), 要求单调递增
      zi[0..nzi-1] - 自变量对数p坐标数组 (如气压), 要求单调递增
      ti[0..nti-1] - 自变量t坐标数组 (如时间), 要求单调递增
      fi[nxi, nyi, nzi, nti] - 四维网格上的函数值数组
      xcyclic      - x方向是否周期 (True表示x是环状变量, 如经度)
      xo, yo, zo, to - 待插值的点坐标
      fo_missing   - 缺测值标记 (输出用)
      nopt         - 选项: =-1 时在缺测数据情况下采用距离加权平均，否则输出缺测
    返回:
      插值得到的函数值 (若无法插值则返回缺测标记 fo_missing)
    """
    
    nxi = len(xi)
    nyi = len(yi)
    nzi = len(zi)
    nti = len(ti)
    
    # 输入有效性检查
    ier = 0
    if nxi < 2 or nyi < 2 or nzi < 2 or nti < 2:
        ier = 1
    if ier != 0:
        return fo_missing
    
    # 检查坐标单调性
    if (not np.all(np.diff(xi) > 0) or 
        not np.all(np.diff(yi) > 0) or 
        not np.all(np.diff(zi) > 0) or 
        not np.all(np.diff(ti) > 0)):
        return fo_missing  # 数据不单调，返回缺测

    # 如果 x 方向周期性，则扩展xi和fi用于周期处理
    xi_arr = np.array(xi, dtype=float)
    yi_arr = np.array(yi, dtype=float)
    zi_arr = np.array(zi, dtype=float)
    ti_arr = np.array(ti, dtype=float)
    fi_arr = np.array(fi, dtype=float)
    
    if xcyclic:
        # 规范化 xo 落入 [xi[0], xi[-1] + dx) 范围
        period = (xi_arr[-1] - xi_arr[0]) + (xi_arr[1] - xi_arr[0])
        xo = ((xo - xi_arr[0]) % period) + xi_arr[0]
        # 构造扩展坐标数组 (在两端各添加一个点)
        dx = xi_arr[1] - xi_arr[0]
        xiw = np.empty(nxi + 2, dtype=float)
        xiw[0] = xi_arr[0] - dx
        xiw[-1] = xi_arr[-1] + dx
        xiw[1:-1] = xi_arr
        # 构造扩展值数组 (两端复制原数组最后和最前的面)
        fixw = np.empty((nxi + 2, nyi, nzi, nti), dtype=float)
        fixw[1:-1, :, :, :] = fi_arr
        fixw[0, :, :, :] = fi_arr[-1, :, :, :]   # 左端延拓：接续最后一个经度的数据
        fixw[-1, :, :, :] = fi_arr[0, :, :, :]   # 右端延拓：接续第一个经度的数据
        xi_use = xiw
        fi_use = fixw
        nxi_use = nxi + 2
    else:
        xi_use = xi_arr
        fi_use = fi_arr
        nxi_use = nxi

    # 在 xi_use 中查找 xo 所在的区间索引
    if xo < xi_use[0] or xo > xi_use[-1]:
        # xo 超出插值范围
        return fo_missing
    # 找到 xo 的相邻索引区间 [nx, nx+1)
    nx = int(np.searchsorted(xi_use, xo) - 1)
    if nx < 0:
        nx = 0
    if nx > nxi_use - 2:
        nx = nxi_use - 2

    # 在 yi 中查找 yo 所在的区间索引
    if yo < yi_arr[0] or yo > yi_arr[-1]:
        return fo_missing
    ny = int(np.searchsorted(yi_arr, yo) - 1)
    if ny < 0:
        ny = 0
    if ny > nyi - 2:
        ny = nyi - 2

    # 在 zi 中查找 zo 所在的区间索引
    if zo < zi_arr[0] or zo > zi_arr[-1]:
        return fo_missing
    nz_ = int(np.searchsorted(zi_arr, zo) - 1)
    if nz_ < 0:
        np_ = 0
    if nz_ > nzi - 2:
        nz_ = nzi - 2

    # 在 ti 中查找 to 所在的区间索引
    if to < ti_arr[0] or to > ti_arr[-1]:
        return fo_missing
    nt = int(np.searchsorted(ti_arr, to) - 1)
    if nt < 0:
        nt = 0
    if nt > nti - 2:
        nt = nti - 2

    # 取十六个顶点值，检查是否缺测
    f0000 = fi_use[nx, ny, np_, nt]
    f0001 = fi_use[nx, ny, np_, nt + 1]
    f0010 = fi_use[nx, ny, np_ + 1, nt]
    f0011 = fi_use[nx, ny, np_ + 1, nt + 1]
    f0100 = fi_use[nx, ny + 1, np_, nt]
    f0101 = fi_use[nx, ny + 1, np_, nt + 1]
    f0110 = fi_use[nx, ny + 1, np_ + 1, nt]
    f0111 = fi_use[nx, ny + 1, np_ + 1, nt + 1]
    f1000 = fi_use[nx + 1, ny, np_, nt]
    f1001 = fi_use[nx + 1, ny, np_, nt + 1]
    f1010 = fi_use[nx + 1, ny, np_ + 1, nt]
    f1011 = fi_use[nx + 1, ny, np_ + 1, nt + 1]
    f1100 = fi_use[nx + 1, ny + 1, np_, nt]
    f1101 = fi_use[nx + 1, ny + 1, np_, nt + 1]
    f1110 = fi_use[nx + 1, ny + 1, np_ + 1, nt]
    f1111 = fi_use[nx + 1, ny + 1, np_ + 1, nt + 1]
    
    # 检查是否有缺测值
    points = [f0000, f0001, f0010, f0011, f0100, f0101, f0110, f0111,
              f1000, f1001, f1010, f1011, f1100, f1101, f1110, f1111]
    
    if any(f == fo_missing for f in points):
        # 存在缺测值
        if nopt == -1:
            # 距离加权平均法近似
            # 这里简单采用周边非缺测值的平均作为替代
            vals = [f for f in points if f != fo_missing]
            return np.mean(vals) if vals else fo_missing
        else:
            return fo_missing

    # 进行四线性插值计算
    # 计算归一化位置 (相对于网格左下角前角的分数)
    tx = (xo - xi_use[nx]) / (xi_use[nx + 1] - xi_use[nx])  # x方向比例
    ty = (yo - yi_arr[ny]) / (yi_arr[ny + 1] - yi_arr[ny])  # y方向比例
    tp = (zo - zi_arr[np_]) / (zi_arr[np_ + 1] - zi_arr[np_])  # p方向比例
    tt = (to - ti_arr[nt]) / (ti_arr[nt + 1] - ti_arr[nt])  # t方向比例

    # 四线性插值公式
    # 先沿x方向插值
    c000 = f0000 * (1 - tx) + f1000 * tx
    c001 = f0001 * (1 - tx) + f1001 * tx
    c010 = f0010 * (1 - tx) + f1010 * tx
    c011 = f0011 * (1 - tx) + f1011 * tx
    c100 = f0100 * (1 - tx) + f1100 * tx
    c101 = f0101 * (1 - tx) + f1101 * tx
    c110 = f0110 * (1 - tx) + f1110 * tx
    c111 = f0111 * (1 - tx) + f1111 * tx
    
    # 再沿y方向插值
    c00 = c000 * (1 - ty) + c100 * ty
    c01 = c001 * (1 - ty) + c101 * ty
    c10 = c010 * (1 - ty) + c110 * ty
    c11 = c011 * (1 - ty) + c111 * ty
    
    # 再沿z方向插值
    c0 = c00 * (1 - tp) + c10 * tp
    c1 = c01 * (1 - tp) + c11 * tp
    
    # 最后沿t方向插值
    fo = c0 * (1 - tt) + c1 * tt

    return fo
