# -*- coding: utf-8 -*-
"""
Only-flat-downstream roughness analysis.
核心要求：后续一切计算（重采样/曲率/采样/JRC）仅使用平整后的残差曲线点。
并通过预处理侧（平滑+RDP）自适应，令 JRC ∈ [0, 20]；不在结果端裁剪。
"""

import argparse
import logging
import math
import os
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, skeletonize

from .config import MASK_ARTIFACT_ROOT, ROUGHNESS_ARTIFACT_PATH

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ================== 配置 ==================
INPUT_FOLDER = str(MASK_ARTIFACT_ROOT / "roughness_analysis")
OUT_ROOT = str(ROUGHNESS_ARTIFACT_PATH)

CIRCUMFERENCE_MM = 94.25  # 圆周长（mm）
DEPTH_MM_PER_IMAGE = 500.0  # 每张图片对应的深度（mm），前三问不需要改，第四问改为1000.0

# 形态学闭运算的核大小和最小对象面积
CLOSE_KERNEL_PX = 3
MIN_OBJ_AREA_PX = 30
MAX_BRIDGE_GAP = 6

# 基线重采样（对平整曲线）
BASE_RESAMPLE_STEP_MM = 0.25  # 重采样步长（mm）

# 三种自适应采样初值
CURV_TARGET_N = 200
CURV_KAPPA_AMP = 10.0
GRAD_TARGET_N = 200
GRAD_AMP = 5.0
RDP_EPS_MM_INIT = 0.20  # RDP算法的初始容差（mm

# 仅在“预处理侧”控制 JRC 范围（0~20）
ADAPT_TARGET_RANGE = (0.0, 20.0)
ADAPT_MAX_ITERS = 8
SMOOTH_WIN_INIT = 5  # 采样点列的移动平均窗口（奇数）
SMOOTH_WIN_STEP = 4
SMOOTH_WIN_MAX = 101
RDP_EPS_STEP = 0.05
RDP_EPS_MAX = 1.00

MAX_POINTS_PER_CRACK = 5000

# ================== 目录 ==================
DIR_CLEAN = DIR_FLATTEN = DIR_COORDS = None
LOGGER = logging.getLogger(__name__)


def configure_output_dirs(out_root):
    global OUT_ROOT, DIR_CLEAN, DIR_FLATTEN, DIR_COORDS
    OUT_ROOT = str(Path(out_root))
    DIR_CLEAN = os.path.join(OUT_ROOT, "01_clean_binary")
    DIR_FLATTEN = os.path.join(OUT_ROOT, "02_desine_flatten")
    DIR_COORDS = os.path.join(OUT_ROOT, "03_sampling_on_flat")
    for directory in (OUT_ROOT, DIR_CLEAN, DIR_FLATTEN, DIR_COORDS):
        os.makedirs(directory, exist_ok=True)


# ================== 基础工具 ==================
def load_binary(path):
    """加载并预处理二值图像"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bw == 255) < np.mean(bw == 0):  # 线为白时翻转
        bw = cv2.bitwise_not(bw)
    return bw


def imwrite_unicode(path, image):
    suffix = Path(path).suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"图像编码失败：{path}")
    encoded.tofile(str(path))


def morph_close_and_clean(bw):
    """形态学闭运算与清理图像"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_PX, CLOSE_KERNEL_PX))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)
    fg = closed == 0
    fg = remove_small_objects(fg, min_size=MIN_OBJ_AREA_PX)
    fg = ndi.binary_fill_holes(fg)
    return np.where(fg, 0, 255).astype(np.uint8)


def find_endpoints(skel):
    """在骨架图中找到端点"""
    kernel = np.ones((3, 3), np.uint8)
    neighbor = cv2.filter2D((skel > 0).astype(np.uint8), -1, kernel)
    endpoints = np.logical_and(skel > 0, neighbor == 2)
    ys, xs = np.where(endpoints)
    return np.column_stack([xs, ys])


def bridge_small_gaps(cleaned):
    """桥接小的裂缝间隙"""
    bin_inv = (cleaned == 0).astype(np.uint8)
    skel = skeletonize(bin_inv).astype(np.uint8) * 255
    pts = find_endpoints(skel)
    if len(pts) < 2:
        return cleaned
    used = np.zeros(len(pts), dtype=bool)
    canvas = bin_inv.copy()
    for i in range(len(pts)):
        if used[i]:
            continue
        xi, yi = pts[i]
        d2 = np.sum((pts - pts[i]) ** 2, axis=1)
        d2[i] = 1e9
        j = int(np.argmin(d2))
        if used[j]:
            continue
        dist = math.sqrt(d2[j])
        if dist <= MAX_BRIDGE_GAP:
            xj, yj = pts[j]
            cv2.line(canvas, (xi, yi), (xj, yj), 1, 1)
            used[i] = used[j] = True
    return np.where(canvas > 0, 0, 255).astype(np.uint8)


def connected_components(cleaned):
    """提取连通区域"""
    lab = label(cleaned == 0, connectivity=2)
    props = regionprops(lab)
    return [(lab == p.label) for p in props if p.area >= MIN_OBJ_AREA_PX]


def extract_sorted_xy_from_mask(mask):
    """提取按 x 排序的坐标点"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None
    pts = np.column_stack([xs, ys])
    pts = np.unique(pts, axis=0)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    return pts[:, 0], pts[:, 1]


def img_coords_to_mm(xs_px, ys_px, H_img, W_img):
    """将像素坐标转换为毫米坐标"""
    xs_px = np.asarray(xs_px, dtype=np.float64)
    ys_px = np.asarray(ys_px, dtype=np.float64)
    mm_per_px_x = float(CIRCUMFERENCE_MM) / float(W_img)
    mm_per_px_y = float(DEPTH_MM_PER_IMAGE) / float(H_img)
    ys_px_bottom = (H_img - 1) - ys_px
    x_mm = xs_px * mm_per_px_x
    y_mm = ys_px_bottom * mm_per_px_y
    return x_mm, y_mm, mm_per_px_x, mm_per_px_y


def rescale_x_to_100mm(x_mm):
    """将x坐标重缩放到100mm内"""
    return x_mm * (100.0 / CIRCUMFERENCE_MM)


def preprocess_xy(xs_px, ys_px, H, W, max_points=MAX_POINTS_PER_CRACK, combine_by_x=True):
    """像素坐标转换到毫米坐标，并进行去重与排序"""
    x_mm, y_mm, _, _ = img_coords_to_mm(xs_px, ys_px, H, W)
    if combine_by_x:
        x_unique = np.unique(x_mm)
        x_new, y_new = [], []
        for xv in x_unique:
            sel = x_mm == xv
            y_new.append(np.median(y_mm[sel]))
            x_new.append(xv)
        x_mm = np.asarray(x_new)
        y_mm = np.asarray(y_new)
    order = np.argsort(x_mm)
    x_mm = x_mm[order]
    y_mm = y_mm[order]
    mask = np.isfinite(x_mm) & np.isfinite(y_mm)
    x_mm = x_mm[mask]
    y_mm = y_mm[mask]
    if len(x_mm) > max_points:
        idx = np.linspace(0, len(x_mm) - 1, max_points).astype(int)
        x_mm = x_mm[idx]
        y_mm = y_mm[idx]
    return x_mm, y_mm


# —— 稳健正弦拟合 + 趋势兜底（与前版一致） ——
def estimate_omega_by_fft_safe(xs, ys):
    """通过FFT估计正弦拟合的频率"""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 16:
        return None
    dx = np.diff(xs)
    if not np.all(dx > 0):
        return None
    if np.nanstd(ys) < 1e-9:
        return None
    d = float(np.median(dx))
    0
    if d <= 0 or not np.isfinite(d):
        return None
    y0 = ys - np.nanmean(ys)
    Y = np.fft.rfft(y0)
    freqs = np.fft.rfftfreq(len(y0), d=d)
    if len(freqs) < 3:
        return None
    k = np.argmax(np.abs(Y[1:])) + 1
    f = float(freqs[k])
    omega = 2.0 * np.pi * f if (np.isfinite(f) and f > 0) else None
    if omega is None:
        return None
    total_w = xs[-1] - xs[0]
    if total_w <= 0:
        return None
    period = (2 * np.pi) / omega
    if period > 2.0 * total_w or period < 4.0 * np.median(dx):
        return None
    return omega


def moving_average(y, k=31):
    k = max(3, int(k) | 1)
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    ker = np.ones(k) / k
    return np.convolve(ypad, ker, mode="valid")


def fit_trend_fallback(xs, ys, poly_deg=2):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    try:
        xmu = xs.mean()
        xstd = xs.std() if xs.std() > 0 else 1.0
        xn = (xs - xmu) / xstd
        deg = 2 if (poly_deg >= 2 and len(xs) >= 32) else 1
        coef = np.polyfit(xn, ys, deg=deg)
        ytrend = np.polyval(coef, xn)
        if np.all(np.isfinite(ytrend)):
            return ytrend
    except Exception:
        pass
    k = max(9, int(len(xs) * 0.03) | 1)
    return moving_average(ys, k=k)


def fit_sine_linear_robust(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    omega = estimate_omega_by_fft_safe(xs, ys)
    if omega is None:
        ytrend = fit_trend_fallback(xs, ys, poly_deg=2)
        return False, 0.0, 0.0, 0.0, ytrend, None
    M = np.column_stack([np.sin(omega * xs), np.cos(omega * xs), np.ones_like(xs)])
    mask = np.isfinite(M).all(axis=1) & np.isfinite(ys)
    M2 = M[mask]
    y2 = ys[mask]
    if len(y2) < 16:
        ytrend = fit_trend_fallback(xs, ys, poly_deg=2)
        return False, 0.0, 0.0, 0.0, ytrend, None
    try:
        coef, _, _, _ = np.linalg.lstsq(M2, y2, rcond=None)
        A, B, C = coef
        R = float(np.hypot(A, B))
        beta = float(np.arctan2(B, A))
        yfit = M @ coef
        if not np.all(np.isfinite(yfit)):
            raise ValueError
        r2_num = np.sum((yfit[mask] - y2) ** 2)
        r2_den = np.sum((y2 - np.mean(y2)) ** 2) + 1e-12
        r2 = 1.0 - r2_num / r2_den
        if r2 < 0.05 or R < 1e-6:
            ytrend = fit_trend_fallback(xs, ys, poly_deg=2)
            return False, 0.0, 0.0, 0.0, ytrend, None
        return True, R, beta, C, yfit, omega
    except Exception:
        ytrend = fit_trend_fallback(xs, ys, poly_deg=2)
        return False, 0.0, 0.0, 0.0, ytrend, None


def desine_project_robust(xs, ys, is_sine, R, beta, C, omega, yfit):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    yfit = np.asarray(yfit, dtype=float)
    y_flat = ys - yfit
    m = np.isfinite(xs) & np.isfinite(y_flat)
    return xs[m], y_flat[m]


# ================== “仅用平整曲线”的下游工具 ==================
def arclength(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    return np.concatenate([[0.0], np.cumsum(np.hypot(dx, dy))])


def resample_equal_arclength(x, y, step_mm=0.25):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return x.copy(), y.copy()
    s = arclength(x, y)
    if s[-1] <= 0:
        return x.copy(), y.copy()
    s_new = np.arange(0.0, s[-1], step_mm)
    return np.interp(s_new, s, x), np.interp(s_new, s, y)


def estimate_curvature(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return np.zeros_like(x)
    dx1 = np.gradient(x)
    dy1 = np.gradient(y)
    dx2 = np.gradient(dx1)
    dy2 = np.gradient(dy1)
    denom = (dx1 * dx1 + dy1 * dy1) ** 1.5
    kap = np.zeros_like(x)
    v = denom > 1e-12
    kap[v] = np.abs(dx1[v] * dy2[v] - dy1[v] * dx2[v]) / denom[v]
    return kap


def curvature_adaptive_sampling(x, y, n_target=200, kappa_amp=10.0):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n <= 2 or n_target >= n:
        return x.copy(), y.copy()
    k = estimate_curvature(x, y)
    w = 1.0 + kappa_amp * np.abs(k)
    w /= np.mean(w)
    s = np.cumsum((w[:-1] + w[1:]) * np.hypot(np.diff(x), np.diff(y)) * 0.5)
    s = np.concatenate([[0.0], s])
    s_t = np.linspace(0.0, s[-1], n_target)
    idx = np.unique(np.interp(s_t, s, np.arange(n)).round().astype(int))
    idx[0] = 0
    idx[-1] = n - 1
    return x[idx], y[idx]


def gradient_adaptive_sampling(x, y, n_target=200, grad_amp=5.0):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n <= 2 or n_target >= n:
        return x.copy(), y.copy()
    dx = np.gradient(x)
    dy = np.gradient(y)
    slope = np.abs(dy / np.maximum(dx, 1e-12))
    w = 1.0 + grad_amp * slope
    w /= np.mean(w)
    s = np.cumsum((w[:-1] + w[1:]) * np.hypot(np.diff(x), np.diff(y)) * 0.5)
    s = np.concatenate([[0.0], s])
    s_t = np.linspace(0.0, s[-1], n_target)
    idx = np.unique(np.interp(s_t, s, np.arange(n)).round().astype(int))
    idx[0] = 0
    idx[-1] = n - 1
    return x[idx], y[idx]

    # 5 自适应采样 + 等间距采样（在平整曲线上）
    # 说明：
    # - x_base, y_base 已经是等弧长 0.25 mm 的基线采样
    # - 若 BASE_RESAMPLE_STEP_MM 能整除 0.5 / 1.0，则直接切片抽样，避免二次插值误差
    # - 否则回退到对 (x_flat, y_flat) 重新做等弧长重采样到目标步长


def _uniform_sample_from_base(x_base, y_base, target_step_mm, base_step_mm):
    """优先用切片法从 0.25mm 基线抽样；不整除时回退重采样"""
    ratio = target_step_mm / float(base_step_mm)
    if abs(ratio - round(ratio)) < 1e-9 and ratio >= 1:
        k = int(round(ratio))
        xs = x_base[::k]
        ys = y_base[::k]
        # 确保最后一个点也包含
        if len(xs) == 0 or (xs[-1] != x_base[-1] or ys[-1] != y_base[-1]):
            xs = np.append(xs, x_base[-1])
            ys = np.append(ys, y_base[-1])
        return xs, ys
    else:
        return resample_equal_arclength(x_base, y_base, target_step_mm)


def max_dev_to_seg(x, y, i0, i1):
    x0, y0 = x[i0], y[i0]
    x1, y1 = x[i1], y[i1]
    vx, vy = x1 - x0, y1 - y0
    denom = math.hypot(vx, vy) + 1e-12
    idx = np.arange(i0 + 1, i1)
    if len(idx) == 0:
        return 0.0, None
    px = x[idx] - x0
    py = y[idx] - y0
    cross = np.abs(px * vy - py * vx) / denom
    k = int(np.argmax(cross))
    return float(cross[k]), int(idx[k])


def rdp(x, y, eps_mm=0.2):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n <= 2:
        return x.copy(), y.copy()
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        dev, imax = max_dev_to_seg(x, y, i0, i1)
        if imax is not None and dev > eps_mm:
            keep[imax] = True
            stack.append((i0, imax))
            stack.append((imax, i1))
    idx = np.where(keep)[0]
    return x[idx], y[idx]


def error_control_sampling(x, y, eps_mm=0.2):
    return rdp(x, y, eps_mm)


# JRC 公式（不裁剪），若超界再“预处理侧”自适应
def compute_Z2_JRC(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0, -10.37
    dx = np.diff(x)
    dy = np.diff(y)
    dx = np.maximum(dx, 1e-12)
    Z2 = float(np.sqrt(np.mean((dy / dx) ** 2)))
    JRC = 51.85 * (Z2**0.6) - 10.37
    return Z2, JRC


# 仅对“采样点列”做自适应预处理
def moving_avg_xy(x, y, k):
    k = max(3, int(k) | 1)
    if len(x) <= k:
        return x.copy(), y.copy()
    pad = k // 2
    # x 基本等距（等弧长/采样后），保留端点
    x_s = x.copy()
    ypad = np.pad(y, (pad, pad), mode="edge")
    ker = np.ones(k) / k
    y_s = np.convolve(ypad, ker, mode="valid")
    return x_s, y_s


# JRC 截断机制：确保 JRC 在 [0, 20] 范围内
def truncate_JRC(JRC, low=0.0, high=20.0):
    """对 JRC 值进行截断，使其始终在指定范围内"""
    return max(low, min(JRC, high))


# 仅对“采样点列”做自适应预处理（结合JRC控制采样）
def adapt_preprocess_for_range(
    x,
    y,
    jrc_low=ADAPT_TARGET_RANGE[0],
    jrc_high=ADAPT_TARGET_RANGE[1],
    max_iters=ADAPT_MAX_ITERS,
    smooth_win_init=SMOOTH_WIN_INIT,
    smooth_win_step=SMOOTH_WIN_STEP,
    smooth_win_max=SMOOTH_WIN_MAX,
    rdp_eps_init=RDP_EPS_MM_INIT,
    rdp_eps_step=RDP_EPS_STEP,
    rdp_eps_max=RDP_EPS_MAX,
):
    """
    仅作用于采样点列：
      高于上限 → 增强平滑/增大RDP ε（删细碎起伏，降低Z2→降JRC）
      低于下限 → 减弱预处理（尽量保留细节；这里主要在初值就偏弱）
    """
    xs, ys = x.copy(), y.copy()
    sw = smooth_win_init
    eps = rdp_eps_init

    for _ in range(max_iters):
        Z2, JRC = compute_Z2_JRC(xs, ys)

        if jrc_low <= JRC <= jrc_high:
            return xs, ys, Z2, JRC, sw, eps, True  # 命中区间

        if JRC > jrc_high:
            # 过高：增强预处理
            if sw < smooth_win_max:
                sw = min(sw + smooth_win_step, smooth_win_max)
                _, ys = moving_avg_xy(xs, ys, sw)
            if eps < rdp_eps_max:
                eps = min(eps + rdp_eps_step, rdp_eps_max)
                xs, ys = rdp(xs, ys, eps)
        else:
            # 过低：尽量减弱（若已经是初始/很弱，就直接退出）
            if sw > smooth_win_init or eps > rdp_eps_init:
                sw = max(smooth_win_init, sw - smooth_win_step)
                eps = max(rdp_eps_init, eps - rdp_eps_step)
                # 重新在原始采样点列上应用更弱的处理：这里简单退出（保持弱处理）
                break
            else:
                break

    # 返回最后一次（可能仍超界，供外层记录/诊断）
    Z2, JRC = compute_Z2_JRC(xs, ys)
    return xs, ys, Z2, JRC, sw, eps, (jrc_low <= JRC <= jrc_high)


# 在主流程中，将自适应调整与JRC控制结合
def run_roughness_analysis(input_folder=INPUT_FOLDER, out_root=OUT_ROOT):
    configure_output_dirs(out_root)
    files = sorted(Path(input_folder).rglob("*.png"))
    if not files:
        raise RuntimeError(f"Roughness mask directory contains no PNG files: {input_folder}")
    rows = []
    for file_path in files:
        name = file_path.stem.replace("_mask", "")
        LOGGER.info("Analyzing roughness mask: %s", name)
        bw = load_binary(str(file_path))
        cleaned = morph_close_and_clean(bw)
        bridged = bridge_small_gaps(cleaned)
        imwrite_unicode(os.path.join(DIR_CLEAN, f"{name}_clean.png"), bridged)
        height, width = bridged.shape

        for idx, component in enumerate(connected_components(bridged), start=1):
            xs_px, ys_px = extract_sorted_xy_from_mask(component)
            if xs_px is None or len(xs_px) < 2:
                continue
            x_mm, y_mm = preprocess_xy(
                xs_px,
                ys_px,
                height,
                width,
                max_points=MAX_POINTS_PER_CRACK,
                combine_by_x=True,
            )
            if len(x_mm) < 16:
                continue
            x_mm_100 = rescale_x_to_100mm(x_mm)
            is_sine, amplitude, phase, center, yfit, omega = fit_sine_linear_robust(x_mm_100, y_mm)
            x_flat, y_flat = desine_project_robust(
                x_mm_100, y_mm, is_sine, amplitude, phase, center, omega, yfit
            )
            if len(x_flat) < 4:
                continue
            x_base, y_base = resample_equal_arclength(x_flat, y_flat, BASE_RESAMPLE_STEP_MM)
            if len(x_base) < 4:
                continue

            fig = plt.figure(figsize=(6, 4))
            plt.plot(x_mm_100, y_mm, lw=0.6, label="original")
            plt.plot(x_mm_100, yfit, lw=1.0, label="fit (sine/trend)")
            plt.plot(x_flat, y_flat, lw=0.8, label="flattened residual")
            plt.legend()
            plt.xlabel("x (mm, width to 100)")
            plt.ylabel("y (mm)")
            plt.title(f"{name} crack#{idx} ({'sine' if is_sine else 'trend'})")
            plt.tight_layout()
            fig.savefig(os.path.join(DIR_FLATTEN, f"{name}_crack{idx}_flatten.png"), dpi=200)
            plt.close(fig)

            uniform_05 = _uniform_sample_from_base(x_base, y_base, 0.5, BASE_RESAMPLE_STEP_MM)
            uniform_10 = _uniform_sample_from_base(x_base, y_base, 1.0, BASE_RESAMPLE_STEP_MM)
            strategies = {
                "curvature": curvature_adaptive_sampling(
                    x_base, y_base, CURV_TARGET_N, CURV_KAPPA_AMP
                ),
                "gradient": gradient_adaptive_sampling(x_base, y_base, GRAD_TARGET_N, GRAD_AMP),
                "error_ctrl": error_control_sampling(x_base, y_base, RDP_EPS_MM_INIT),
                "uniform_0p5": uniform_05,
                "uniform_1p0": uniform_10,
            }

            for strategy_name, (xs0, ys0) in strategies.items():
                xs, ys, z2, jrc_raw, smooth_window, epsilon, in_range = adapt_preprocess_for_range(
                    xs0, ys0
                )
                jrc_bounded = truncate_JRC(jrc_raw)
                fig2 = plt.figure(figsize=(5, 4))
                plt.plot(x_base, y_base, lw=0.8, label="baseline")
                plt.scatter(xs, ys, s=10, label=f"{strategy_name} ({len(xs)} pts)")
                plt.xlabel("x (mm)")
                plt.ylabel("flattened y (mm)")
                plt.title(
                    f"{name} crack#{idx} | {strategy_name}\n"
                    f"Z2={z2:.4f}, raw JRC={jrc_raw:.2f}, bounded={jrc_bounded:.2f}"
                )
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(DIR_COORDS, f"{name}_crack{idx}_{strategy_name}.png"),
                    dpi=200,
                )
                plt.close(fig2)
                period = (
                    float(2 * np.pi / omega)
                    if omega is not None and np.isfinite(omega) and omega != 0.0
                    else np.nan
                )
                rows.append(
                    {
                        "图像名": name,
                        "裂缝编号": int(idx),
                        "方法": strategy_name,
                        "振幅R(mm)": float(amplitude),
                        "周期P(mm)": period,
                        "相位beta(rad)": float(phase),
                        "中心线位置C(mm)": float(center),
                        "JRC原始值": float(jrc_raw),
                        "JRC值(0-20截断)": float(jrc_bounded),
                        "原始采样点数": int(len(xs0)),
                        "最终采样点数": int(len(xs)),
                        "Z2": float(z2),
                        "平滑窗口": int(smooth_window),
                        "RDP阈值(mm)": float(epsilon),
                        "原始JRC是否在[0,20]": bool(in_range),
                    }
                )

    if not rows:
        raise RuntimeError("No valid roughness results were produced; inspect the masks")
    frame = pd.DataFrame(rows)
    csv_path = Path(OUT_ROOT) / "roughness_jrc_results.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    frame.to_excel(Path(OUT_ROOT) / "roughness_jrc_results.xlsx", index=False)
    LOGGER.info("Roughness analysis complete: rows=%s output=%s", len(frame), OUT_ROOT)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Fracture roughness and JRC analysis")
    parser.add_argument("--input", type=Path, default=Path(INPUT_FOLDER))
    parser.add_argument("--output", type=Path, default=Path(OUT_ROOT))
    args = parser.parse_args()
    run_roughness_analysis(args.input, args.output)


if __name__ == "__main__":
    main()
