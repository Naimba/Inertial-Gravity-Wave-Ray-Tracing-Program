# interpolation.py - 线性插值模块
import numpy as np
import numba as nb


# ===============================
# 边界工具函数
# ===============================

def periodic_x(x, x0, x1):
    """周期经度"""
    L = x1 - x0
    return (x - x0) % L + x0


def clamp(x, x_bottom, x_top):
    """边界复制"""
    if x_bottom > x_top:
        return np.maximum(np.minimum(x, x_bottom), x_top)
    if x_bottom < x_top:
        return np.minimum(np.maximum(x, x_bottom), x_top)
\

def safe_shift(x, dx, x_bottom, x_top, periodic=False):
    """
    生成 x+dx 和 x-dx
    """
    xp = x + dx
    xm = x - dx

    if periodic:
        xp = periodic_x(xp, x_bottom, x_top)
        xm = periodic_x(xm, x_bottom, x_top)
    else:
        xp = clamp(xp, x_bottom, x_top)
        xm = clamp(xm, x_bottom, x_top)

    return xp, xm


# ===============================
# 四线性插值
# ===============================

# ===============================
# 单点插值（Numba核心）
# ===============================

@nb.njit(inline='always')
def interp4d_point(xi, yi, zi, ti, fi, xo, yo, zo, to, xcyclic):

    nx = xi.shape[0]
    ny = yi.shape[0]
    nz = zi.shape[0]
    nt = ti.shape[0]

    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    dt = ti[1] - ti[0]

    ix = (xo - xi[0]) / dx
    iy = (yo - yi[0]) / dy
    it = (to - ti[0]) / dt

    x0 = int(np.floor(ix))
    y0 = int(np.floor(iy))
    t0 = int(np.floor(it))

    # x
    if xcyclic:
        x0 = x0 % nx
        x1 = (x0 + 1) % nx
    else:
        if x0 < 0: x0 = 0
        if x0 > nx-1: x0 = nx-1
        if x1 < 0: x1 = 0
        if x1 > nx-1: x1 = nx-1

    # y
    y1 = y0 + 1

    if y0 < 0: y0 = 0
    if y0 > ny-1: y0 = ny-1
    if y1 < 0: y1 = 0
    if y1 > ny-1: y1 = ny-1
    # t
    # t0 = min(max(t0,0),nt-1)
    # t1 = min(max(t0+1,0),nt-1)
    t1 = t0 + 1
    xd = ix - x0
    yd = iy - y0
    td = it - t0
    
    if xd < 0: xd = 0.0
    if xd > 1: xd = 1.0

    if yd < 0: yd = 0.0
    if yd > 1: yd = 1.0

    # z（非均匀）
    k = np.searchsorted(zi, zo)

    z0 = min(max(k-1, 0), nz - 1)
    z1 = min(max(k, 0), nz - 1)

    if z1 == z0:
        z1 = min(z0 + 1, nz - 1)

    zc0 = zi[z0]
    zc1 = zi[z1]

    dz = zc1 - zc0
    zd = 0.0
    if dz != 0.0:
        zd = (zo - zc0)/dz

    if zd < 0: zd = 0
    if zd > 1: zd = 1

    wx0 = 1 - xd
    wx1 = xd
    wy0 = 1 - yd
    wy1 = yd
    wz0 = 1 - zd
    wz1 = zd
    wt0 = 1 - td
    wt1 = td

    f = 0.0

    # 16点展开
    f += fi[x0, y0, z0, t0] * wx0 * wy0 * wz0 * wt0
    f += fi[x0, y0, z0, t1] * wx0 * wy0 * wz0 * wt1
    f += fi[x0, y0, z1, t0] * wx0 * wy0 * wz1 * wt0
    f += fi[x0, y0, z1, t1] * wx0 * wy0 * wz1 * wt1

    f += fi[x0, y1, z0, t0] * wx0 * wy1 * wz0 * wt0
    f += fi[x0, y1, z0, t1] * wx0 * wy1 * wz0 * wt1
    f += fi[x0, y1, z1, t0] * wx0 * wy1 * wz1 * wt0
    f += fi[x0, y1, z1, t1] * wx0 * wy1 * wz1 * wt1

    f += fi[x1, y0, z0, t0] * wx1 * wy0 * wz0 * wt0
    f += fi[x1, y0, z0, t1] * wx1 * wy0 * wz0 * wt1
    f += fi[x1, y0, z1, t0] * wx1 * wy0 * wz1 * wt0
    f += fi[x1, y0, z1, t1] * wx1 * wy0 * wz1 * wt1

    f += fi[x1, y1, z0, t0] * wx1 * wy1 * wz0 * wt0
    f += fi[x1, y1, z0, t1] * wx1 * wy1 * wz0 * wt1
    f += fi[x1, y1, z1, t0] * wx1 * wy1 * wz1 * wt0
    f += fi[x1, y1, z1, t1] * wx1 * wy1 * wz1 * wt1

    return f

@nb.njit(parallel=True)
def interp4d_only(xi, yi, zi, ti, fi, xo, yo, zo, to, xcyclic=True):

    n = xo.shape[0]
    out = np.empty(n, dtype=np.float64)

    for i in nb.prange(n):

        if not (np.isfinite(xo[i]) and np.isfinite(yo[i]) and np.isfinite(zo[i])):
            out[i] = np.nan
            continue

        out[i] = interp4d_point(
            xi, yi, zi, ti, fi,
            xo[i], yo[i], zo[i], to,
            xcyclic
        )

    return out


# ===============================
# 插值 + 一阶二阶导数
# ===============================

def interp4d_value(xi, yi, zi, ti, fi, xo, yo, zo, to, xcyclic):

    xo = np.asarray(xo)
    yo = np.asarray(yo)
    zo = np.asarray(zo)

    n = xo.size
    f = np.full(n, np.nan)

    valid = (
        np.isfinite(xo) &
        np.isfinite(yo) &
        np.isfinite(zo) &
        (yo >= min(yi)) &
        (yo <= max(yi)) &
        (zo >= min(zi)) &
        (zo <= max(zi))
    )

    if not np.any(valid):
        return f

    xv = xo[valid]
    yv = yo[valid]
    zv = zo[valid]

    fv = interp4d_only(xi, yi, zi, ti, fi, xv, yv, zv, to, xcyclic)

    f[valid] = fv

    return f

def interp4d_grad(xi, yi, zi, ti, fi, xo, yo, zo, to, xcyclic=True):

    xo = np.asarray(xo)
    yo = np.asarray(yo)
    zo = np.asarray(zo)

    n = xo.size

    f  = np.full(n, np.nan)
    fx = np.full(n, np.nan)
    fy = np.full(n, np.nan)
    fz = np.full(n, np.nan)
    ft = np.full(n, np.nan)

    valid = (
        np.isfinite(xo) &
        np.isfinite(yo) &
        np.isfinite(zo) &
        (yo >= min(yi)) &
        (yo <= max(yi)) &
        (zo >= min(zi)) &
        (zo <= max(zi))
    )

    if not np.any(valid):
        return f, fx, fy, fz, ft

    xv = xo[valid]
    yv = yo[valid]
    zv = zo[valid]

    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    dt = ti[1] - ti[0]
    
    
    k  = np.searchsorted(zi, zv)
    z0 = np.clip(k-1, 0, len(zi)-1)
    z1 = np.clip(k,   0, len(zi)-1)

    dz = zi[z1] - zi[z0]
    # dz = np.maximum(dz,1e-12)

    fv = interp4d_only(xi, yi, zi, ti, fi, xv, yv, zv, to, xcyclic)

    xp,xm = safe_shift(xv, dx, xi[0], xi[-1], xcyclic)
    yp,ym = safe_shift(yv, dy, yi[0], yi[-1], False)
    zp,zm = safe_shift(zv, dz, zi[0], zi[-1], False)
    tp,tm = safe_shift(to, dt, ti[0], ti[-1], False)
      
    fxv = (interp4d_only(xi, yi, zi, ti, fi, xp, yv, zv, to, xcyclic)
          -interp4d_only(xi, yi, zi, ti, fi, xm, yv, zv, to, xcyclic))/(2 * dx)

    fyv = (interp4d_only(xi, yi, zi, ti, fi, xv, yp, zv, to, xcyclic)
          -interp4d_only(xi, yi, zi, ti, fi, xv, ym, zv, to, xcyclic))/(2 * dy)

    fzv = (interp4d_only(xi, yi, zi, ti, fi, xv, yv, zp, to, xcyclic)
          -interp4d_only(xi, yi, zi, ti, fi, xv, yv, zm, to, xcyclic))/(2 * dz)

    ftv = (interp4d_only(xi, yi, zi, ti, fi, xv, yv, zv, tp, xcyclic)
          -interp4d_only(xi, yi, zi, ti, fi, xv, yv, zv, tm, xcyclic))/(2 * dt)
    
    f[valid]  = fv
    fx[valid] = fxv
    fy[valid] = fyv
    fz[valid] = fzv
    ft[valid] = ftv

    return f, fx, fy, fz, ft

def interp4d_grad2(xi, yi, zi, ti, fi, xo, yo, zo, to, xcyclic=True):

    f,fx,fy,fz,ft = interp4d_grad(xi, yi, zi, ti, fi, xo, yo, zo, to, xcyclic)

    xo = np.asarray(xo)
    yo = np.asarray(yo)
    zo = np.asarray(zo)

    n = xo.size

    fxx = np.full(n, np.nan)
    fxy = np.full(n, np.nan)
    fxz = np.full(n, np.nan)
    fxt = np.full(n, np.nan)

    fyy = np.full(n, np.nan)
    fyz = np.full(n, np.nan)
    fyt = np.full(n, np.nan)

    fzz = np.full(n, np.nan)
    fzt = np.full(n, np.nan)

    valid = np.isfinite(f)

    if not np.any(valid):
        return (f, fx, fy, fz, ft, 
                fxx, fxy, fxz, fxt, 
                fxy, fyy, fyz, fyt, 
                fzz, fzt)

    xv = xo[valid]
    yv = yo[valid]
    zv = zo[valid]

    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    dt = ti[1] - ti[0]

    k  = np.searchsorted(zi, zv)
    z0 = np.clip(k-1, 0, len(zi)-1)
    z1 = np.clip(k,   0, len(zi)-1)

    dz = zi[z1] - zi[z0]
    # dz = np.maximum(dz,1e-12)

    xp, xm = safe_shift(xv, dx, xi[0], xi[-1], xcyclic)
    yp, ym = safe_shift(yv, dy, yi[0], yi[-1], False)
    zp, zm = safe_shift(zv, dz, zi[0], zi[-1], False)
    tp, tm = safe_shift(to, dt, ti[0], ti[-1], False)

    fv = f[valid]

    # ======================
    # 二阶导（主对角）
    # ======================

    fxxv = (interp4d_only(xi, yi, zi, ti, fi, xp, yv, zv, to, xcyclic)
           -2 * fv
           +interp4d_only(xi, yi, zi, ti, fi, xm, yv, zv, to, xcyclic))/(dx * dx)

    fyyv = (interp4d_only(xi, yi, zi, ti, fi, xv, yp, zv, to, xcyclic)
           -2 * fv
           +interp4d_only(xi, yi, zi, ti, fi, xv, ym, zv, to, xcyclic))/(dy * dy)

    fzzv = (interp4d_only(xi, yi, zi, ti, fi, xv, yv, zp, to, xcyclic)
           -2 * fv
           +interp4d_only(xi, yi, zi, ti, fi, xv, yv, zm, to, xcyclic))/(dz * dz)

    # ======================
    # 交叉导数
    # ======================

    fxyv = (interp4d_only(xi, yi, zi, ti, fi, xp, yp, zv, to, xcyclic)
           -interp4d_only(xi, yi, zi, ti, fi, xp, ym, zv, to, xcyclic)
           -interp4d_only(xi, yi, zi, ti, fi, xm, yp, zv, to, xcyclic)
           +interp4d_only(xi, yi, zi, ti, fi, xm, ym, zv, to, xcyclic))/(4 * dx * dy)

    fxzv = (interp4d_only(xi, yi, zi, ti, fi, xp, yv, zp, to, xcyclic)
           -interp4d_only(xi, yi, zi, ti, fi, xp, yv, zm, to, xcyclic)
           -interp4d_only(xi, yi, zi, ti, fi, xm, yv, zp, to, xcyclic)
           +interp4d_only(xi, yi, zi, ti, fi, xm, yv, zm, to, xcyclic))/(4 * dx * dz)

    fxtv = (interp4d_only(xi, yi, zi, ti, fi, xp, yv, zv, tp, xcyclic)
           -interp4d_only(xi, yi, zi, ti, fi, xp, yv, zv, tm, xcyclic)
           -interp4d_only(xi, yi, zi, ti, fi, xm, yv, zv, tp, xcyclic)
           +interp4d_only(xi, yi, zi, ti, fi, xm, yv, zv, tm, xcyclic))/(4 * dx * dt)

    fyzv = (interp4d_only(xi, yi, zi, ti, fi, xv, yp, zp, to, xcyclic)
           -interp4d_only(xi, yi, zi, ti, fi, xv, yp, zm, to, xcyclic)
           -interp4d_only(xi, yi, zi, ti, fi, xv, ym, zp, to, xcyclic)
           +interp4d_only(xi, yi, zi, ti, fi, xv, ym, zm, to, xcyclic))/(4 * dy * dz)

    fytv = (interp4d_only(xi, yi, zi, ti, fi, xv, yp, zv, tp, xcyclic)
           -interp4d_only(xi, yi, zi, ti, fi, xv, yp, zv, tm, xcyclic)
           -interp4d_only(xi, yi, zi, ti, fi, xv, ym, zv, tp, xcyclic)
           +interp4d_only(xi, yi, zi, ti, fi, xv, ym, zv, tm, xcyclic))/(4 * dy * dt)

    fztv = (interp4d_only(xi, yi, zi, ti, fi, xv, yv, zp, tp, xcyclic)
           -interp4d_only(xi, yi, zi, ti, fi, xv, yv, zp, tm, xcyclic)
           -interp4d_only(xi, yi, zi, ti, fi, xv, yv, zm, tp, xcyclic)
           +interp4d_only(xi, yi, zi, ti, fi, xv, yv, zm, tm, xcyclic))/(4 * dz * dt)

    # ======================
    # 写回
    # ======================

    fxx[valid] = fxxv
    fyy[valid] = fyyv
    fzz[valid] = fzzv

    fxy[valid] = fxyv
    fxz[valid] = fxzv
    fxt[valid] = fxtv

    fyz[valid] = fyzv
    fyt[valid] = fytv
    fzt[valid] = fztv

    return (f, fx, fy, fz, ft, 
            fxx, fxy, fxz, fxt, 
            fxy, fyy, fyz, fyt, 
            fzz, fzt)