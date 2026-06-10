
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import yaml

C0 = 299_792_458.0
EPS = 1e-12


# ============================================================
# Basic utilities
# ============================================================

def ri_from_complex(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex64)
    return np.stack([x.real.astype(np.float32), x.imag.astype(np.float32)], axis=-1)


def is_num(x) -> bool:
    return isinstance(x, (int, float, np.number))


def pick_float(rng: np.random.Generator, spec) -> float:
    if is_num(spec):
        return float(spec)
    if isinstance(spec, (list, tuple)):
        if len(spec) == 0:
            raise ValueError("empty numeric spec")
        if len(spec) == 1 and is_num(spec[0]):
            return float(spec[0])
        if len(spec) == 2 and is_num(spec[0]) and is_num(spec[1]):
            a, b = float(spec[0]), float(spec[1])
            lo, hi = (a, b) if a <= b else (b, a)
            return float(rng.uniform(lo, hi))
        return float(rng.choice(spec))
    raise TypeError(f"bad float spec: {spec!r}")


def pick_int(rng: np.random.Generator, spec) -> int:
    if isinstance(spec, (int, np.integer)):
        return int(spec)
    if isinstance(spec, (list, tuple)):
        if len(spec) == 1 and isinstance(spec[0], (int, np.integer)):
            return int(spec[0])
        if len(spec) == 2 and all(isinstance(x, (int, np.integer)) for x in spec):
            a, b = int(spec[0]), int(spec[1])
            lo, hi = (a, b) if a <= b else (b, a)
            return int(rng.integers(lo, hi + 1))
        return int(rng.choice(spec))
    raise TypeError(f"bad int spec: {spec!r}")


def pick_choice(rng: np.random.Generator, xs):
    xs = list(xs)
    if not xs:
        raise ValueError("empty choice list")
    return rng.choice(xs)


def safe_filename(s: str) -> str:
    return "".join([c if (c.isalnum() or c in "-_+.,") else "_" for c in str(s)])


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def stable_seed(*items: Any, modulo: int = 2**32 - 1) -> int:
    s = "|".join(map(str, items)).encode("utf-8")
    h = hashlib.blake2b(s, digest_size=8).digest()
    return int.from_bytes(h, "little") % modulo


def parse_float_spec(s: Optional[str], default: Optional[Any] = None):
    if s is None:
        return default
    s = str(s).strip()
    if "," in s:
        vals = [float(x.strip()) for x in s.split(",") if x.strip()]
        if len(vals) == 1:
            return vals[0]
        return vals
    return float(s)


def parse_types(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    xs = [x.strip() for x in str(s).split(",") if x.strip()]
    return xs if xs else []


def parse_float_list(s: Optional[str], default: Optional[List[float]] = None) -> Optional[List[float]]:
    if s is None:
        return default
    xs = [x.strip() for x in str(s).split(",") if x.strip()]
    if not xs:
        return []
    return [float(x) for x in xs]


# ============================================================
# Default config patch: keeps existing UAV proxy unchanged
# ============================================================

def interference_default_block() -> Dict[str, Any]:
    return {
        "enable_interference": True,
        "enable_noise": True,

        # These can be scalar or [min, max].
        "snr_db": [0.0, 20.0],
        "sir_db": [-10.0, 10.0],

        # Total SIR definition:
        #   10log10(mean(|target_array_signal|^2) / mean(|sum_all_interference_array_signal|^2))
        "sir_definition": "target_power_over_total_interference_power",

        # Which interference types are allowed to appear.
        # Empty or missing means all enabled components below.
        "enabled_interference_types": [
            "cw",
            "cw_drift",
            "wideband_block",
            "fullband_impulse",
            "fh_jammer",
            "chirp_sweep",
        ],

        # Overlap definitions:
        # binary:
        #   sum(Ms * Mi) / sum(Ms)
        # target_energy:
        #   sum(Ps * Mi) / sum(Ps)
        # In target_energy mode, SIR is still controlled separately afterwards.
        "tf_overlap": {
            "mode": "target_energy",       # none | binary | target_energy
            "range": [0.00, 0.05],
            "max_candidates": 256,
            "accept_tolerance": 0.005,
            "target_mask_threshold": 1e-8,
            "candidate_score": "range_distance",
        },

        # Per clip, sample a subset of enabled types.
        # If always_use_all_enabled_types=True, each enabled type appears once.
        "mixture_policy": {
            "always_use_all_enabled_types": False,
            "min_components": 1,
            "max_components": 3,
            "allow_repeated_types": True,
        },

        "spatial_policy": {
            "near_target_prob": 0.30,
            "near_target_delta_az_deg": [10.0, 25.0],
            "near_target_delta_el_deg": [3.0, 12.0],
            "random_az_range_deg": [0.0, 360.0],
            "random_el_range_deg": [0.0, 90.0],
        },

        "interference": {
            "components": [
                {
                    "type": "cw",
                    "enable": True,
                    "num_tones": [1, 3],
                    "tone_width_bins": [1, 2],
                    "time_duty": [0.80, 1.00],
                    "tone_drift_bins": [0, 2],
                    "allow_zero_drift_direction": True,
                },
                {
                    "type": "cw_drift",
                    "enable": True,
                    "num_tones": [1, 3],
                    "tone_width_bins": [1, 2],
                    "time_duty": [0.80, 1.00],
                    "tone_drift_bins": [2, 6],
                    "allow_zero_drift_direction": False,
                },
                {
                    "type": "wideband_block",
                    "enable": True,
                    "num_blocks": [1, 2],
                    "bandwidth_hz": [5.0e6, 30.0e6],
                    "duration_frames": [32, 512],
                },
                {
                    "type": "fullband_impulse",
                    "enable": True,
                    "num_impulses_per_clip": [1, 8],
                    "len_frames": [1, 8],
                    "full_band_ratio": [0.70, 1.00],
                },
                {
                    "type": "fh_jammer",
                    "enable": True,
                    "num_hops": [3, 12],
                    "tone_bw_hz": [0.8e6, 4.0e6],
                    "dwell_frames": [2, 16],
                    "hop_gap_frames": [0, 8],
                    "hopset_bw_hz": [20.0e6, 90.0e6],
                },
                {
                    "type": "chirp_sweep",
                    "enable": True,
                    "num_sweeps": [1, 2],
                    "duration_frames": [32, 256],
                    "sweep_bw_hz": [5.0e6, 50.0e6],
                    "tone_bw_hz": [0.8e6, 4.0e6],
                },
            ]
        },
    }


def exp7_default_block() -> Dict[str, Any]:
    return {
        "enable": False,
        "split_name": "test",
        "samples_per_bin": 0,
        "sep_bin_edges_deg": [],
        "component_types": [],
        "single_component_only": True,
        "max_doa_trials": 4096,
        "near_target_threshold_deg": 25.0,
    }


def exp6_default_block() -> Dict[str, Any]:
    return {
        "enable": False,
        "split_name": "test",
        "samples_per_type": 0,
        "component_types": [],
        "single_component_only": True,
    }


def apply_interference_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(cfg)
    base = interference_default_block()
    old = cfg.get("interference_noise", {})
    cfg["interference_noise"] = deep_update(base, old if isinstance(old, dict) else {})
    exp6_base = exp6_default_block()
    exp6_old = cfg.get("exp6_controlled_jammer_type", {})
    cfg["exp6_controlled_jammer_type"] = deep_update(
        exp6_base, exp6_old if isinstance(exp6_old, dict) else {}
    )
    exp7_base = exp7_default_block()
    exp7_old = cfg.get("exp7_controlled_separation", {})
    cfg["exp7_controlled_separation"] = deep_update(
        exp7_base, exp7_old if isinstance(exp7_old, dict) else {}
    )
    return cfg


# ============================================================
# Array model
# ============================================================

def uca_positions(M: int, radius_m: float) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, M, endpoint=False)
    return np.stack(
        [radius_m * np.cos(ang), radius_m * np.sin(ang), np.zeros_like(ang)], axis=1
    ).astype(np.float64)


def unit_vec_from_az_el(az_deg: float, el_deg: float) -> np.ndarray:
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    return np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
        dtype=np.float64,
    )


def az_el_from_unit_vec(u: np.ndarray) -> Tuple[float, float]:
    u = np.asarray(u, dtype=np.float64).reshape(3)
    x, y, z = u
    az = float(np.rad2deg(np.arctan2(y, x)) % 360.0)
    el = float(np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0))))
    return az, el


def separation_3d_deg(az1: float, el1: float, az2: float, el2: float) -> float:
    u = unit_vec_from_az_el(az1, el1)
    v = unit_vec_from_az_el(az2, el2)
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(dot)))


def orthonormal_basis_around(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.float64).reshape(3)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64) if abs(u[2]) < 0.99 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    e1 = np.cross(ref, u)
    n1 = float(np.linalg.norm(e1))
    if n1 < 1e-12:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        e1 = np.cross(ref, u)
        n1 = float(np.linalg.norm(e1))
    if n1 < 1e-12:
        raise RuntimeError("failed to build tangent basis for controlled separation sampling")
    e1 /= n1
    e2 = np.cross(u, e1)
    e2 /= max(float(np.linalg.norm(e2)), 1e-12)
    return e1, e2


def steering_vectors(
    pos: np.ndarray,
    freqs_rf: np.ndarray,
    az_deg: float,
    el_deg: float,
    mode: str,
    fc_hz: float,
) -> np.ndarray:
    u = unit_vec_from_az_el(az_deg, el_deg)
    proj = pos @ u
    if mode == "center_freq":
        k = 2.0 * np.pi * fc_hz / C0
        a = np.exp(-1j * k * proj)[None, :]
        return np.repeat(a, len(freqs_rf), axis=0).astype(np.complex64)
    k = 2.0 * np.pi * freqs_rf / C0
    return np.exp(-1j * (k[:, None] * proj[None, :])).astype(np.complex64)


# ============================================================
# Target UAV proxy source generation
# ============================================================

def build_time_axis_s(cfg: Dict[str, Any]) -> np.ndarray:
    Nt = int(cfg["stft"]["T"])
    duration_s = float(cfg["stft"].get("observation_duration_ms", 50.0)) * 1e-3
    return np.linspace(0.0, duration_s, Nt, endpoint=False, dtype=np.float64)


def build_freq_axis_hz(cfg: Dict[str, Any]) -> np.ndarray:
    Nf = int(cfg["stft"]["F"])
    f_min_hz = float(cfg["stft"]["f_min_hz"])
    f_max_hz = float(cfg["stft"]["f_max_hz"])
    df = (f_max_hz - f_min_hz) / Nf
    return f_min_hz + (np.arange(Nf, dtype=np.float64) + 0.5) * df


def gen_ofdm_source(rng, Nt, Nf, f_axis_hz, time_axis_s, cfg):
    bw = float(pick_choice(rng, cfg.get("bw_choices_hz", [1e7, 2e7])))
    duty = pick_float(rng, cfg.get("duty", [0.7, 0.95]))
    on_ratio = pick_float(rng, cfg.get("burst_on_ratio", [0.6, 1.0]))
    center_off = pick_float(rng, cfg.get("center_offset_ratio", [-0.1, 0.1]))

    fmin, fmax = float(f_axis_hz[0]), float(f_axis_hz[-1])
    band = fmax - fmin
    bw = min(bw, band)
    center = fmin + 0.5 * band + center_off * (0.5 * band)
    occ_f = (f_axis_hz >= center - 0.5 * bw) & (f_axis_hz <= center + 0.5 * bw)

    total_duration_s = float(time_axis_s[-1] - time_axis_s[0] + (time_axis_s[1] - time_axis_s[0]))
    default_period_s = max(1e-6, 0.01 * total_duration_s)
    burst_period_ms = cfg.get("burst_period_ms", [default_period_s * 1e3, default_period_s * 1e3])
    period_s = 1e-3 * pick_float(rng, burst_period_ms)
    dt = float(time_axis_s[1] - time_axis_s[0])
    period_bins = max(1, int(round(period_s / dt)))
    on_bins = max(1, int(round(on_ratio * period_bins)))

    occ_t = np.zeros((Nt,), dtype=bool)
    start = int(rng.integers(0, period_bins))
    for t in range(Nt):
        occ_t[t] = ((t - start) % period_bins) < on_bins
    cur = float(occ_t.mean())
    if cur > 1e-9:
        occ_t = occ_t & (rng.random(Nt) < min(1.0, duty / cur))

    S = (rng.standard_normal((Nt, Nf)) + 1j * rng.standard_normal((Nt, Nf))).astype(np.complex64)
    S *= occ_t[:, None].astype(np.float32)
    S *= occ_f[None, :].astype(np.float32)
    p = np.mean(np.abs(S) ** 2)
    return S / np.sqrt(p) if p > 0 else S


def gen_fhss_source(rng, Nt, Nf, f_axis_hz, time_axis_s, cfg):
    duty = pick_float(rng, cfg.get("duty", [0.3, 0.9]))
    hopset_bw = float(pick_float(rng, cfg.get("hopset_bw_hz", 8e7)))
    tone_bw = float(pick_float(rng, cfg.get("tone_bw_hz", 1.2e6)))
    dwell_ms = float(pick_float(rng, cfg.get("dwell_ms", 2.0)))
    hop_int_ms = float(pick_float(rng, cfg.get("hop_interval_ms", 8.0)))
    hop_step = float(pick_float(rng, cfg.get("hop_step_hz", 3.2e6)))
    center_off = pick_float(rng, cfg.get("center_offset_ratio", [-0.2, 0.2]))

    fmin, fmax = float(f_axis_hz[0]), float(f_axis_hz[-1])
    band = fmax - fmin
    hopset_bw = min(hopset_bw, band)
    base_center = fmin + 0.5 * band + center_off * (0.5 * band)
    hs_lo = base_center - 0.5 * hopset_bw
    hs_hi = base_center + 0.5 * hopset_bw
    f0 = float(rng.uniform(hs_lo, hs_hi))

    dwell_s = dwell_ms * 1e-3
    hop_int_s = hop_int_ms * 1e-3
    dt = float(time_axis_s[1] - time_axis_s[0])

    S = np.zeros((Nt, Nf), dtype=np.complex64)
    hop_on = {}
    for t in range(Nt):
        time_s = t * dt
        hop_idx = int(math.floor(time_s / max(1e-12, hop_int_s)))
        if hop_idx not in hop_on:
            hop_on[hop_idx] = rng.random() <= duty
        if not hop_on[hop_idx]:
            continue
        local = time_s - hop_idx * hop_int_s
        if local > dwell_s:
            continue
        fh = f0 + hop_idx * hop_step
        if hopset_bw > 0:
            fh = hs_lo + ((fh - hs_lo) % hopset_bw)
        occ_f = (f_axis_hz >= fh - 0.5 * tone_bw) & (f_axis_hz <= fh + 0.5 * tone_bw)
        if not np.any(occ_f):
            continue
        S[t, occ_f] = (
            rng.standard_normal(np.sum(occ_f)) + 1j * rng.standard_normal(np.sum(occ_f))
        ).astype(np.complex64)

    p = np.mean(np.abs(S) ** 2)
    return S / np.sqrt(p) if p > 0 else S


def build_enabled_proxy_names(cfg: Dict[str, Any]) -> List[str]:
    explicit = list(cfg.get("dataset_plan", {}).get("proxy_names", []))
    if explicit:
        return [str(x) for x in explicit]
    type_probs = cfg.get("uav_proxy", {}).get("type_probs", {})
    if type_probs:
        return [str(k) for k in type_probs.keys()]
    return [str(x["name"]) for x in cfg.get("uav_proxy", {}).get("proxy_types", [])]


def find_proxy_cfg(cfg: Dict[str, Any], proxy_name: str) -> Dict[str, Any]:
    for item in cfg["uav_proxy"]["proxy_types"]:
        if item["name"] == proxy_name:
            return dict(item)
    raise KeyError(f"proxy_type not found: {proxy_name}")


def generate_target_signal_scalar(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    proxy_name: str,
    Nt: int,
    Nf: int,
    f_axis: np.ndarray,
    time_axis_s: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    proxy_cfg = find_proxy_cfg(cfg, proxy_name)
    components = [dict(c) for c in proxy_cfg.get("components", [])]

    S_total = np.zeros((Nt, Nf), dtype=np.complex64)
    for c in components:
        if c["kind"] == "ofdm":
            S_total += float(c.get("weight", 1.0)) * gen_ofdm_source(rng, Nt, Nf, f_axis, time_axis_s, c)
        else:
            raise ValueError(f"Unsupported proxy component kind: {c['kind']}")

    fh_cfg = cfg.get("uav_auxiliary", {}).get("fhss_global", {})
    fh_present = False
    if bool(fh_cfg.get("enable", False)) and rng.random() <= float(fh_cfg.get("presence_prob", 0.0)):
        fh_present = True
        S_total += float(pick_float(rng, fh_cfg.get("weight", [0.10, 0.25]))) * gen_fhss_source(
            rng, Nt, Nf, f_axis, time_axis_s, fh_cfg
        )

    p = np.mean(np.abs(S_total) ** 2)
    if p > 0:
        S_total /= np.sqrt(p)
    return S_total.astype(np.complex64), bool(fh_present)


# ============================================================
# Interference mask/source generation
# ============================================================

def hz_to_bins(freq_bw_hz: float, f_axis_hz: np.ndarray) -> int:
    if len(f_axis_hz) < 2:
        return 1
    df = float(np.median(np.diff(f_axis_hz)))
    return max(1, int(round(float(freq_bw_hz) / max(df, EPS))))


def mark_rect(mask: np.ndarray, t0: int, t1: int, f0: int, f1: int) -> None:
    Nt, Nf = mask.shape
    t0 = max(0, min(Nt, int(t0)))
    t1 = max(0, min(Nt, int(t1)))
    f0 = max(0, min(Nf, int(f0)))
    f1 = max(0, min(Nf, int(f1)))
    if t1 > t0 and f1 > f0:
        mask[t0:t1, f0:f1] = True


def mask_cw(rng, Nt, Nf, f_axis_hz, cfg):
    mask = np.zeros((Nt, Nf), dtype=bool)
    n_tones = pick_int(rng, cfg.get("num_tones", [1, 3]))
    width_spec = cfg.get("tone_width_bins", [1, 2])
    time_duty = pick_float(rng, cfg.get("time_duty", [0.8, 1.0]))
    drift_bins = pick_int(rng, cfg.get("tone_drift_bins", [0, 2]))
    allow_zero_drift_direction = bool(cfg.get("allow_zero_drift_direction", True))
    t_on = rng.random(Nt) < time_duty

    for _ in range(n_tones):
        f_start = int(rng.integers(0, Nf))
        width = max(1, pick_int(rng, width_spec))
        if drift_bins <= 0:
            drift_dir = 0
        else:
            drift_dir = int(rng.choice([-1, 0, 1] if allow_zero_drift_direction else [-1, 1]))
        for t in range(Nt):
            if not t_on[t]:
                continue
            f_center = f_start + drift_dir * int(round(drift_bins * t / max(1, Nt - 1)))
            f_center %= Nf
            mark_rect(mask, t, t + 1, f_center - width // 2, f_center - width // 2 + width)
    return mask


def mask_wideband_block(rng, Nt, Nf, f_axis_hz, cfg):
    mask = np.zeros((Nt, Nf), dtype=bool)
    n_blocks = pick_int(rng, cfg.get("num_blocks", [1, 2]))
    for _ in range(n_blocks):
        dur = min(Nt, max(1, pick_int(rng, cfg.get("duration_frames", [32, Nt]))))
        t0 = int(rng.integers(0, max(1, Nt - dur + 1)))
        bw_bins = min(Nf, hz_to_bins(pick_float(rng, cfg.get("bandwidth_hz", [5e6, 30e6])), f_axis_hz))
        f0 = int(rng.integers(0, max(1, Nf - bw_bins + 1)))
        mark_rect(mask, t0, t0 + dur, f0, f0 + bw_bins)
    return mask


def mask_fullband_impulse(rng, Nt, Nf, f_axis_hz, cfg):
    mask = np.zeros((Nt, Nf), dtype=bool)
    n_imp = pick_int(rng, cfg.get("num_impulses_per_clip", [1, 8]))
    for _ in range(n_imp):
        length = min(Nt, max(1, pick_int(rng, cfg.get("len_frames", [1, 8]))))
        ratio = min(1.0, max(0.0, pick_float(rng, cfg.get("full_band_ratio", [0.7, 1.0]))))
        bw_bins = min(Nf, max(1, int(round(ratio * Nf))))
        t0 = int(rng.integers(0, max(1, Nt - length + 1)))
        f0 = int(rng.integers(0, max(1, Nf - bw_bins + 1)))
        mark_rect(mask, t0, t0 + length, f0, f0 + bw_bins)
    return mask


def mask_fh_jammer(rng, Nt, Nf, f_axis_hz, cfg):
    mask = np.zeros((Nt, Nf), dtype=bool)
    n_hops = pick_int(rng, cfg.get("num_hops", [3, 12]))
    tone_bins = min(Nf, hz_to_bins(pick_float(rng, cfg.get("tone_bw_hz", [0.8e6, 4e6])), f_axis_hz))
    hopset_bins = min(Nf, hz_to_bins(pick_float(rng, cfg.get("hopset_bw_hz", [20e6, 90e6])), f_axis_hz))
    dwell_spec = cfg.get("dwell_frames", [2, 16])
    gap_spec = cfg.get("hop_gap_frames", [0, 8])
    hs0 = int(rng.integers(0, max(1, Nf - hopset_bins + 1)))
    t = int(rng.integers(0, max(1, Nt // 8 + 1)))
    for _ in range(n_hops):
        if t >= Nt:
            break
        dwell = max(1, pick_int(rng, dwell_spec))
        f0 = hs0 + int(rng.integers(0, max(1, hopset_bins - tone_bins + 1)))
        mark_rect(mask, t, min(Nt, t + dwell), f0, f0 + tone_bins)
        t += dwell + max(0, pick_int(rng, gap_spec))
    return mask


def mask_chirp_sweep(rng, Nt, Nf, f_axis_hz, cfg):
    mask = np.zeros((Nt, Nf), dtype=bool)
    n_sw = pick_int(rng, cfg.get("num_sweeps", [1, 2]))
    tone_bins = min(Nf, hz_to_bins(pick_float(rng, cfg.get("tone_bw_hz", [0.8e6, 4e6])), f_axis_hz))
    sweep_bins = min(Nf, hz_to_bins(pick_float(rng, cfg.get("sweep_bw_hz", [5e6, 50e6])), f_axis_hz))
    for _ in range(n_sw):
        dur = min(Nt, max(2, pick_int(rng, cfg.get("duration_frames", [32, 256]))))
        t0 = int(rng.integers(0, max(1, Nt - dur + 1)))
        f_start = int(rng.integers(0, Nf))
        direction = int(rng.choice([-1, 1]))
        for i in range(dur):
            alpha = i / max(1, dur - 1)
            fc = int(round(f_start + direction * alpha * sweep_bins)) % Nf
            f_lo = fc - tone_bins // 2
            f_hi = f_lo + tone_bins
            # Handle wrap-around.
            if f_lo < 0:
                mark_rect(mask, t0 + i, t0 + i + 1, f_lo + Nf, Nf)
                mark_rect(mask, t0 + i, t0 + i + 1, 0, f_hi)
            elif f_hi > Nf:
                mark_rect(mask, t0 + i, t0 + i + 1, f_lo, Nf)
                mark_rect(mask, t0 + i, t0 + i + 1, 0, f_hi - Nf)
            else:
                mark_rect(mask, t0 + i, t0 + i + 1, f_lo, f_hi)
    return mask


MASK_GENERATORS = {
    "cw": mask_cw,
    "cw_drift": mask_cw,
    "wideband_block": mask_wideband_block,
    "fullband_impulse": mask_fullband_impulse,
    "fh_jammer": mask_fh_jammer,
    "chirp_sweep": mask_chirp_sweep,
}


def random_phase_signal_on_mask(rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    z = np.zeros(mask.shape, dtype=np.complex64)
    n = int(mask.sum())
    if n <= 0:
        return z
    vals = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    z[mask] = vals
    p = np.mean(np.abs(z) ** 2)
    if p > 0:
        z /= np.sqrt(p)
    return z


def get_component_cfg_map(in_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    comps = in_cfg.get("interference", {}).get("components", [])
    out = {}
    for c in comps:
        if bool(c.get("enable", True)):
            out[str(c["type"])] = dict(c)
    return out


def choose_component_types(rng: np.random.Generator, in_cfg: Dict[str, Any]) -> List[str]:
    c_map = get_component_cfg_map(in_cfg)
    enabled = list(in_cfg.get("enabled_interference_types", []))
    if not enabled:
        enabled = list(c_map.keys())
    enabled = [t for t in enabled if t in c_map and t in MASK_GENERATORS]
    if not enabled:
        raise ValueError("No valid enabled interference types.")

    pol = in_cfg.get("mixture_policy", {})
    if bool(pol.get("always_use_all_enabled_types", False)):
        return list(enabled)

    mn = int(pol.get("min_components", 1))
    mx = int(pol.get("max_components", max(1, len(enabled))))
    mn = max(1, min(mn, len(enabled) if not bool(pol.get("allow_repeated_types", True)) else max(mn, 1)))
    mx = max(mn, mx)
    n = int(rng.integers(mn, mx + 1))
    if bool(pol.get("allow_repeated_types", True)):
        return [str(rng.choice(enabled)) for _ in range(n)]
    n = min(n, len(enabled))
    return [str(x) for x in rng.choice(enabled, size=n, replace=False)]


def compute_overlap_metrics(target_mask: np.ndarray, target_power: np.ndarray, interf_mask: np.ndarray) -> Dict[str, float]:
    Ms = target_mask.astype(bool)
    Mi = interf_mask.astype(bool)
    Ps = np.asarray(target_power, dtype=np.float64)
    binary_den = max(1, int(Ms.sum()))
    energy_den = float(Ps.sum() + EPS)

    binary = float(np.logical_and(Ms, Mi).sum() / binary_den)
    target_energy = float((Ps * Mi.astype(np.float64)).sum() / energy_den)

    return {
        "binary": binary,
        "target_energy": target_energy,
    }


def range_distance(x: float, lo: float, hi: float) -> float:
    if lo <= x <= hi:
        return 0.0
    return min(abs(x - lo), abs(x - hi))


def sample_interference_candidate(
    rng: np.random.Generator,
    in_cfg: Dict[str, Any],
    Nt: int,
    Nf: int,
    f_axis: np.ndarray,
) -> Tuple[List[Tuple[str, np.ndarray]], np.ndarray]:
    c_map = get_component_cfg_map(in_cfg)
    types = choose_component_types(rng, in_cfg)
    comps: List[Tuple[str, np.ndarray]] = []
    agg = np.zeros((Nt, Nf), dtype=bool)
    for typ in types:
        cfg = c_map[typ]
        mask = MASK_GENERATORS[typ](rng, Nt, Nf, f_axis, cfg)
        if not np.any(mask):
            continue
        comps.append((typ, mask))
        agg |= mask
    return comps, agg


def sample_interference_with_overlap(
    rng: np.random.Generator,
    in_cfg: Dict[str, Any],
    Nt: int,
    Nf: int,
    f_axis: np.ndarray,
    target_mask: np.ndarray,
    target_power: np.ndarray,
    overlap_range: Sequence[float],
    overlap_mode: str,
) -> Tuple[List[Tuple[str, np.ndarray]], Dict[str, float], bool]:
    ov_cfg = in_cfg.get("tf_overlap", {})
    max_candidates = int(ov_cfg.get("max_candidates", 256))
    tol = float(ov_cfg.get("accept_tolerance", 0.005))

    lo, hi = float(overlap_range[0]), float(overlap_range[1])
    lo, hi = max(0.0, min(lo, hi)), min(1.0, max(lo, hi))

    if overlap_mode == "none":
        comps, agg = sample_interference_candidate(rng, in_cfg, Nt, Nf, f_axis)
        metrics = compute_overlap_metrics(target_mask, target_power, agg)
        return comps, metrics, True

    if overlap_mode not in ("binary", "target_energy"):
        raise ValueError(f"Unsupported overlap_mode={overlap_mode}")

    best = None
    best_score = float("inf")
    accepted = False

    for _ in range(max_candidates):
        comps, agg = sample_interference_candidate(rng, in_cfg, Nt, Nf, f_axis)
        metrics = compute_overlap_metrics(target_mask, target_power, agg)
        x = metrics[overlap_mode]
        score = range_distance(x, lo, hi)
        if score < best_score:
            best_score = score
            best = (comps, metrics)
        if score <= tol or (lo <= x <= hi):
            accepted = True
            return comps, metrics, accepted

    if best is None:
        comps, agg = sample_interference_candidate(rng, in_cfg, Nt, Nf, f_axis)
        metrics = compute_overlap_metrics(target_mask, target_power, agg)
        return comps, metrics, False
    return best[0], best[1], accepted


def sample_interference_doa(
    rng: np.random.Generator,
    target_az: float,
    target_el: float,
    spatial_cfg: Dict[str, Any],
) -> Tuple[float, float, bool]:
    if rng.random() < float(spatial_cfg.get("near_target_prob", 0.0)):
        da = pick_float(rng, spatial_cfg.get("near_target_delta_az_deg", [10, 25]))
        de = pick_float(rng, spatial_cfg.get("near_target_delta_el_deg", [3, 12]))
        da *= float(rng.choice([-1.0, 1.0]))
        de *= float(rng.choice([-1.0, 1.0]))
        az = (target_az + da) % 360.0
        el = float(np.clip(target_el + de, 0.0, 90.0))
        return az, el, True
    az = pick_float(rng, spatial_cfg.get("random_az_range_deg", [0, 360]))
    el = pick_float(rng, spatial_cfg.get("random_el_range_deg", [0, 90]))
    return float(az % 360.0), float(np.clip(el, 0.0, 90.0)), False


def sample_controlled_interference_doa(
    rng: np.random.Generator,
    target_az: float,
    target_el: float,
    sep_range_deg: Sequence[float],
    max_trials: int = 4096,
) -> Tuple[float, float, float]:
    lo, hi = float(sep_range_deg[0]), float(sep_range_deg[1])
    lo = max(0.0, min(lo, hi))
    hi = min(180.0, max(lo, hi))
    if hi <= lo:
        raise ValueError(f"invalid controlled separation range: {sep_range_deg}")

    u = unit_vec_from_az_el(target_az, target_el)
    e1, e2 = orthonormal_basis_around(u)

    for _ in range(max_trials):
        sep_deg = float(rng.uniform(lo, hi))
        sep_rad = np.deg2rad(sep_deg)
        phi = float(rng.uniform(0.0, 2.0 * np.pi))
        tangential = np.cos(phi) * e1 + np.sin(phi) * e2
        v = np.cos(sep_rad) * u + np.sin(sep_rad) * tangential
        v /= max(float(np.linalg.norm(v)), 1e-12)
        az_i, el_i = az_el_from_unit_vec(v)
        if 0.0 <= el_i <= 90.0:
            sep3d = separation_3d_deg(target_az, target_el, az_i, el_i)
            return az_i, el_i, sep3d

    raise RuntimeError(
        f"failed to sample controlled interference DOA in range {sep_range_deg} "
        f"for target ({target_az:.3f}, {target_el:.3f}) after {max_trials} trials"
    )


def build_interference_array_signal(
    rng: np.random.Generator,
    cfg: Dict[str, Any],
    comps: List[Tuple[str, np.ndarray]],
    target_az: float,
    target_el: float,
    pos: np.ndarray,
    freqs_rf: np.ndarray,
    fc_hz: float,
    steer_mode: str,
    forced_component_doa: Optional[Tuple[float, float]] = None,
    forced_component_sep3d_deg: Optional[float] = None,
    force_near_target: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    Nt, Nf = comps[0][1].shape if comps else (int(cfg["stft"]["T"]), int(cfg["stft"]["F"]))
    M = pos.shape[0]
    Xi = np.zeros((Nt, Nf, M), dtype=np.complex64)
    scalar_power = np.zeros((Nt, Nf), dtype=np.float32)
    agg_mask = np.zeros((Nt, Nf), dtype=bool)
    meta_comps: List[Dict[str, Any]] = []

    spatial_cfg = cfg["interference_noise"].get("spatial_policy", {})
    for idx, (typ, mask) in enumerate(comps):
        z = random_phase_signal_on_mask(rng, mask)
        if forced_component_doa is not None:
            az_i, el_i = forced_component_doa
            if forced_component_sep3d_deg is None:
                sep3d_deg = separation_3d_deg(target_az, target_el, az_i, el_i)
            else:
                sep3d_deg = float(forced_component_sep3d_deg)
            if force_near_target is None:
                near_thr = float(cfg.get("exp7_controlled_separation", {}).get("near_target_threshold_deg", 25.0))
                near = bool(sep3d_deg <= near_thr)
            else:
                near = bool(force_near_target)
        else:
            az_i, el_i, near = sample_interference_doa(rng, target_az, target_el, spatial_cfg)
            sep3d_deg = separation_3d_deg(target_az, target_el, az_i, el_i)
        A_i = steering_vectors(pos, freqs_rf, az_i, el_i, steer_mode, fc_hz)
        Xi_i = z[:, :, None] * A_i[None, :, :]
        Xi += Xi_i.astype(np.complex64)
        scalar_power += (np.abs(z) ** 2).astype(np.float32)
        agg_mask |= mask
        meta_comps.append({
            "index": idx,
            "type": typ,
            "az_deg": float(az_i),
            "el_deg": float(el_i),
            "near_target": bool(near),
            "sep3d_deg": float(sep3d_deg),
            "tf_occupancy": float(mask.mean()),
        })

    return Xi, scalar_power, meta_comps


# ============================================================
# Noise, errors, SCM
# ============================================================

def apply_array_errors(rng, X, cfg):
    M = X.shape[-1]
    gain_db = rng.normal(0.0, float(cfg.get("gain_std_db", 0.0)), size=(M,))
    phase_deg = rng.normal(0.0, float(cfg.get("phase_std_deg", 0.0)), size=(M,))
    g = (10.0 ** (gain_db / 20.0)) * np.exp(1j * np.deg2rad(phase_deg))
    return X * g[None, None, :]


def compute_quicklook_scm_timeavg(X: np.ndarray) -> np.ndarray:
    Nt, Nf, M = X.shape
    R = np.zeros((Nt, M, M), dtype=np.complex64)
    for t in range(Nt):
        Xt = X[t, :, :]
        R[t] = (Xt.T @ Xt.conj()) / max(1, Nf)
    return R


def scale_to_sir(Xs: np.ndarray, Xi: np.ndarray, sir_db: float) -> Tuple[np.ndarray, float]:
    Ps = float(np.mean(np.abs(Xs) ** 2) + EPS)
    Pi = float(np.mean(np.abs(Xi) ** 2) + EPS)
    Pi_target = Ps / (10.0 ** (float(sir_db) / 10.0))
    scale = math.sqrt(Pi_target / Pi) if Pi > 0 else 0.0
    Xi2 = Xi * np.float32(scale)
    eff = 10.0 * math.log10(Ps / float(np.mean(np.abs(Xi2) ** 2) + EPS))
    return Xi2.astype(np.complex64), float(eff)


def make_noise(rng: np.random.Generator, shape: Tuple[int, int, int]) -> np.ndarray:
    N = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    p = float(np.mean(np.abs(N) ** 2) + EPS)
    return (N / math.sqrt(p)).astype(np.complex64)


def scale_noise_to_snr(Xs: np.ndarray, N: np.ndarray, snr_db: float) -> Tuple[np.ndarray, float]:
    Ps = float(np.mean(np.abs(Xs) ** 2) + EPS)
    Pn = float(np.mean(np.abs(N) ** 2) + EPS)
    Pn_target = Ps / (10.0 ** (float(snr_db) / 10.0))
    scale = math.sqrt(Pn_target / Pn) if Pn > 0 else 0.0
    N2 = N * np.float32(scale)
    eff = 10.0 * math.log10(Ps / float(np.mean(np.abs(N2) ** 2) + EPS))
    return N2.astype(np.complex64), float(eff)


# ============================================================
# Dataset plan helpers
# ============================================================

def build_snr_values(cfg: Dict[str, Any]) -> List[float]:
    scan = cfg.get("dataset_plan", {}).get("snr_scan", None)
    if isinstance(scan, dict):
        start = int(scan.get("start_db", -20))
        stop = int(scan.get("stop_db", 20))
        step = int(scan.get("step_db", 5))
        return [float(v) for v in range(start, stop + 1, step)]

    spec = cfg.get("interference_noise", {}).get("snr_db", [0.0, 20.0])
    if isinstance(spec, list) and len(spec) == 2 and all(is_num(x) for x in spec):
        return [float(spec[0]), float(spec[1])] if float(spec[0]) != float(spec[1]) else [float(spec[0])]
    if isinstance(spec, list):
        return [float(x) for x in spec]
    return [float(spec)]


def build_sir_values(cfg: Dict[str, Any]) -> List[float]:
    scan = cfg.get("dataset_plan", {}).get("sir_scan", None)
    if isinstance(scan, dict):
        start = int(scan.get("start_db", -10))
        stop = int(scan.get("stop_db", 10))
        step = int(scan.get("step_db", 5))
        return [float(v) for v in range(start, stop + 1, step)]
    spec = cfg.get("interference_noise", {}).get("sir_db", [-10.0, 10.0])
    if isinstance(spec, list) and len(spec) == 2 and all(is_num(x) for x in spec):
        return [float(spec[0]), float(spec[1])] if float(spec[0]) != float(spec[1]) else [float(spec[0])]
    if isinstance(spec, list):
        return [float(x) for x in spec]
    return [float(spec)]


def build_overlap_ranges(cfg: Dict[str, Any]) -> List[List[float]]:
    plan = cfg.get("dataset_plan", {})
    if "overlap_ranges" in plan:
        return [[float(a), float(b)] for a, b in plan["overlap_ranges"]]
    ov = cfg.get("interference_noise", {}).get("tf_overlap", {})
    r = ov.get("range", [0.0, 0.05])
    if len(r) == 2 and is_num(r[0]) and is_num(r[1]):
        return [[float(r[0]), float(r[1])]]
    return [[float(a), float(b)] for a, b in r]


def build_split_counts(cfg: Dict[str, Any]) -> Dict[str, int]:
    plan = cfg.get("dataset_plan", {})
    base_n = int(plan.get("base_N", 1))
    out: Dict[str, int] = {}
    for split_name in ["train", "val", "test"]:
        multiplier = int(plan.get("splits", {}).get(split_name, {}).get("multiplier", 1))
        out[split_name] = base_n * multiplier
    return out


def build_exp7_plan(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    exp7_cfg = cfg.get("exp7_controlled_separation", {})
    if not bool(exp7_cfg.get("enable", False)):
        return None

    edges = [float(x) for x in exp7_cfg.get("sep_bin_edges_deg", [])]
    if len(edges) < 2:
        raise ValueError("exp7_controlled_separation.sep_bin_edges_deg must contain at least two edges")
    if any(edges[i + 1] <= edges[i] for i in range(len(edges) - 1)):
        raise ValueError("exp7_controlled_separation.sep_bin_edges_deg must be strictly increasing")

    samples_per_bin = int(exp7_cfg.get("samples_per_bin", 0))
    if samples_per_bin <= 0:
        raise ValueError("exp7_controlled_separation.samples_per_bin must be > 0 when exp7 mode is enabled")

    split_name = str(exp7_cfg.get("split_name", "test")).strip().lower()
    if split_name not in ("train", "val", "test"):
        raise ValueError("exp7_controlled_separation.split_name must be one of train/val/test")

    component_types = [str(x).strip() for x in exp7_cfg.get("component_types", []) if str(x).strip()]
    if not component_types:
        component_types = list(cfg.get("interference_noise", {}).get("enabled_interference_types", []))
    if not component_types:
        raise ValueError("exp7 controlled mode requires at least one interference component type")

    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        label = f"sep{int(round(lo))}-{int(round(hi))}deg"
        bins.append({
            "range_deg": [float(lo), float(hi)],
            "label": label,
        })

    return {
        "split_name": split_name,
        "samples_per_bin": samples_per_bin,
        "bins": bins,
        "component_types": component_types,
        "single_component_only": bool(exp7_cfg.get("single_component_only", True)),
        "max_doa_trials": int(exp7_cfg.get("max_doa_trials", 4096)),
        "near_target_threshold_deg": float(exp7_cfg.get("near_target_threshold_deg", 25.0)),
    }


def build_exp6_plan(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    exp6_cfg = cfg.get("exp6_controlled_jammer_type", {})
    if not bool(exp6_cfg.get("enable", False)):
        return None

    samples_per_type = int(exp6_cfg.get("samples_per_type", 0))
    if samples_per_type <= 0:
        raise ValueError("exp6_controlled_jammer_type.samples_per_type must be > 0 when exp6 mode is enabled")

    split_name = str(exp6_cfg.get("split_name", "test")).strip().lower()
    if split_name not in ("train", "val", "test"):
        raise ValueError("exp6_controlled_jammer_type.split_name must be one of train/val/test")

    component_types = [str(x).strip() for x in exp6_cfg.get("component_types", []) if str(x).strip()]
    if not component_types:
        component_types = list(cfg.get("interference_noise", {}).get("enabled_interference_types", []))
    if not component_types:
        raise ValueError("exp6 controlled mode requires at least one interference component type")

    return {
        "split_name": split_name,
        "samples_per_type": samples_per_type,
        "component_types": component_types,
        "single_component_only": bool(exp6_cfg.get("single_component_only", True)),
    }


# ============================================================
# Clip generation
# ============================================================

def gen_one_clip_with_interference(
    cfg: Dict[str, Any],
    base_seed: int,
    forced_proxy_name: str,
    forced_snr_db: float,
    forced_sir_db: float,
    overlap_range: Sequence[float],
    overlap_mode: str,
    split_name: str,
    sample_index_within_bucket: int,
    condition_tag: str,
    exp6_component_type: Optional[str] = None,
    exp7_sep_range_deg: Optional[Sequence[float]] = None,
    exp7_sep_label: str = "",
    exp7_component_type: Optional[str] = None,
) -> Dict[str, Any]:
    array = cfg["array"]
    stft = cfg["stft"]
    in_cfg = cfg["interference_noise"]

    Nt = int(stft["T"])
    Nf = int(stft["F"])
    time_axis_s = build_time_axis_s(cfg)
    f_axis = build_freq_axis_hz(cfg)

    fc_hz = float(array["center_freq_hz"])
    freqs_rf = f_axis + fc_hz
    pos = uca_positions(int(array["M"]), float(array["radius_m"]))
    steer_mode = str(array.get("steering", {}).get("mode", "per_bin"))

    # Target/noise seed intentionally excludes SIR and overlap,
    # so target and noise can stay fixed across interference sweeps.
    target_seed = stable_seed(base_seed, "target", split_name, forced_proxy_name, forced_snr_db, sample_index_within_bucket)
    noise_seed = stable_seed(base_seed, "noise", split_name, forced_proxy_name, forced_snr_db, sample_index_within_bucket)
    interf_seed = stable_seed(
        base_seed, "interference", split_name, forced_proxy_name, forced_snr_db,
        forced_sir_db, overlap_mode, tuple(overlap_range), sample_index_within_bucket, condition_tag
    )

    rng_target = np.random.default_rng(target_seed)
    rng_noise = np.random.default_rng(noise_seed)
    rng_interf = np.random.default_rng(interf_seed)

    doa = cfg["scene"]["doa"]
    az = float(pick_float(rng_target, doa.get("az_range_deg", [0, 360])))
    el = float(pick_float(rng_target, doa.get("el_range_deg", [0, 60])))
    A = steering_vectors(pos, freqs_rf, az, el, steer_mode, fc_hz)

    S_total, fh_present = generate_target_signal_scalar(
        cfg, rng_target, forced_proxy_name, Nt, Nf, f_axis, time_axis_s
    )
    target_power = (np.abs(S_total) ** 2).astype(np.float32)
    target_thr = float(in_cfg.get("tf_overlap", {}).get("target_mask_threshold", 1e-8))
    target_mask = target_power > target_thr

    Xs = (S_total[:, :, None] * A[None, :, :]).astype(np.complex64)

    exp6_mode = exp6_component_type is not None
    exp7_mode = exp7_sep_range_deg is not None
    if exp6_mode and exp7_mode:
        raise ValueError("exp6 controlled jammer-type mode and exp7 controlled separation mode cannot be enabled together")

    exp6_cfg = cfg.get("exp6_controlled_jammer_type", {})
    exp7_cfg = cfg.get("exp7_controlled_separation", {})
    in_cfg_for_clip = dict(in_cfg)
    if exp6_mode or exp7_mode:
        in_cfg_for_clip = deep_update({}, in_cfg)
        forced_type = exp6_component_type if exp6_mode else exp7_component_type
        if forced_type:
            in_cfg_for_clip["enabled_interference_types"] = [str(forced_type)]
        single_component_only = bool(
            exp6_cfg.get("single_component_only", True) if exp6_mode else exp7_cfg.get("single_component_only", True)
        )
        if single_component_only:
            in_cfg_for_clip["mixture_policy"] = {
                "always_use_all_enabled_types": False,
                "min_components": 1,
                "max_components": 1,
                "allow_repeated_types": False,
            }

    if not bool(in_cfg_for_clip.get("enable_interference", True)):
        Xi = np.zeros_like(Xs)
        interf_scalar_power = np.zeros((Nt, Nf), dtype=np.float32)
        interf_mask = np.zeros((Nt, Nf), dtype=bool)
        overlap_metrics = compute_overlap_metrics(target_mask, target_power, interf_mask)
        overlap_accepted = True
        component_meta = []
        sir_eff = float("inf")
    else:
        comps, overlap_metrics, overlap_accepted = sample_interference_with_overlap(
            rng_interf, in_cfg_for_clip, Nt, Nf, f_axis, target_mask, target_power,
            overlap_range=overlap_range, overlap_mode=overlap_mode
        )
        forced_component_doa = None
        forced_component_sep3d_deg = None
        forced_near_target = None
        if exp7_mode:
            if len(comps) == 0:
                raise RuntimeError("exp7 controlled separation mode sampled zero interference components")
            if bool(exp7_cfg.get("single_component_only", True)) and len(comps) != 1:
                raise RuntimeError(
                    "exp7 controlled separation mode requires exactly one interference component; "
                    f"got {len(comps)}"
                )
            forced_component_doa = sample_controlled_interference_doa(
                rng_interf,
                az,
                el,
                exp7_sep_range_deg,
                max_trials=int(exp7_cfg.get("max_doa_trials", 4096)),
            )[:2]
            forced_component_sep3d_deg = separation_3d_deg(
                az, el, forced_component_doa[0], forced_component_doa[1]
            )
            forced_near_target = bool(
                forced_component_sep3d_deg <= float(exp7_cfg.get("near_target_threshold_deg", 25.0))
            )
        interf_mask = np.zeros((Nt, Nf), dtype=bool)
        for _, m in comps:
            interf_mask |= m

        if len(comps) == 0 or not np.any(interf_mask):
            Xi = np.zeros_like(Xs)
            interf_scalar_power = np.zeros((Nt, Nf), dtype=np.float32)
            component_meta = []
            sir_eff = float("inf")
        else:
            Xi, interf_scalar_power, component_meta = build_interference_array_signal(
                rng_interf,
                cfg,
                comps,
                az,
                el,
                pos,
                freqs_rf,
                fc_hz,
                steer_mode,
                forced_component_doa=forced_component_doa,
                forced_component_sep3d_deg=forced_component_sep3d_deg,
                force_near_target=forced_near_target,
            )
            Xi, sir_eff = scale_to_sir(Xs, Xi, float(forced_sir_db))
            # Scalar power after total-SIR scaling. Scaling factor can be inferred from Xi.
            # For analysis, recompute array-averaged interference TF power.
            interf_scalar_power = np.mean(np.abs(Xi) ** 2, axis=-1).astype(np.float32)

    if bool(in_cfg.get("enable_noise", True)):
        N = make_noise(rng_noise, Xs.shape)
        N, snr_eff = scale_noise_to_snr(Xs, N, float(forced_snr_db))
    else:
        N = np.zeros_like(Xs)
        snr_eff = float("inf")

    X_clean = Xs
    X_interf = Xi
    X_noise = N
    X = (Xs + Xi + N).astype(np.complex64)

    # Apply the same receiver errors to the final observation. Keep separated components unmodified
    # so SNR/SIR metadata remains exactly defined by the synthetic mixture before receiver errors.
    rng_err = np.random.default_rng(stable_seed(base_seed, "array_error", split_name, forced_proxy_name, forced_snr_db, sample_index_within_bucket))
    X = apply_array_errors(rng_err, X, cfg.get("errors", {}).get("per_clip_random", {})).astype(np.complex64)

    duration_ms = float(stft.get("observation_duration_ms", 50.0))
    df_hz = float((float(stft["f_max_hz"]) - float(stft["f_min_hz"])) / Nf)
    lo, hi = float(overlap_range[0]), float(overlap_range[1])

    Ps = float(np.mean(np.abs(Xs) ** 2) + EPS)
    Pi = float(np.mean(np.abs(Xi) ** 2) + EPS)
    Pn = float(np.mean(np.abs(N) ** 2) + EPS)

    meta = {
        "dataset_mode": "signal_noise_interference_tf512",
        "split": split_name,
        "sample_index_within_bucket": int(sample_index_within_bucket),
        "condition_tag": condition_tag,
        "proxy_type": forced_proxy_name,
        "fhss_present": bool(fh_present),
        "Nt": Nt,
        "Nf": Nf,
        "observation_duration_ms": duration_ms,
        "df_hz": df_hz,
        "center_freq_hz": fc_hz,
        "f_min_hz": float(stft["f_min_hz"]),
        "f_max_hz": float(stft["f_max_hz"]),
        "steering_mode": steer_mode,
        "az": float(az),
        "el": float(el),

        "target_seed": int(target_seed),
        "noise_seed": int(noise_seed),
        "interference_seed": int(interf_seed),

        "snr_db_target": float(forced_snr_db),
        "snr_db_effective": float(snr_eff),
        "sir_db_target_total": float(forced_sir_db),
        "sir_db_effective_total": float(sir_eff),
        "sir_definition": str(in_cfg_for_clip.get("sir_definition", "target_power_over_total_interference_power")),
        "target_power_mean": Ps,
        "interference_power_mean": Pi,
        "noise_power_mean": Pn,

        "overlap_mode": str(overlap_mode),
        "overlap_target_range": [lo, hi],
        "overlap_binary_actual": float(overlap_metrics["binary"]),
        "overlap_target_energy_actual": float(overlap_metrics["target_energy"]),
        "overlap_accepted": bool(overlap_accepted),
        "target_tf_occupancy": float(target_mask.mean()),
        "interference_tf_occupancy": float(interf_mask.mean()),

        "enabled_interference_types": list(in_cfg_for_clip.get("enabled_interference_types", [])),
        "component_types": [c["type"] for c in component_meta],
        "components": component_meta,
        "exp6_controlled_jammer_type": {
            "enabled": bool(exp6_mode),
            "component_type": str(exp6_component_type or ""),
        },
        "exp7_controlled_separation": {
            "enabled": bool(exp7_mode),
            "sep_range_deg": [float(exp7_sep_range_deg[0]), float(exp7_sep_range_deg[1])] if exp7_mode else [],
            "sep_label": str(exp7_sep_label),
            "component_type": str(exp7_component_type or ""),
        },
    }

    return {
        "X": X,
        "X_clean": X_clean,
        "X_interf": X_interf,
        "X_noise": X_noise,
        "valid_tf": np.ones((Nt, Nf), dtype=np.uint8),
        "target_mask": target_mask.astype(np.uint8),
        "interference_mask": interf_mask.astype(np.uint8),
        "signal_tf_power": target_power.astype(np.float32),
        "interf_tf_power": interf_scalar_power.astype(np.float32),
        "freqs_rf": freqs_rf.astype(np.float32),
        "time_axis_s": time_axis_s.astype(np.float32),
        "az_gt": np.full((Nt,), az, dtype=np.float32),
        "el_gt": np.full((Nt,), el, dtype=np.float32),
        "proxy_type": forced_proxy_name,
        "split": split_name,
        "snr_db": float(forced_snr_db),
        "sir_db": float(forced_sir_db),
        "overlap_range": [lo, hi],
        "overlap_mode": str(overlap_mode),
        "meta": meta,
    }


def write_h5(path: str, clip: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    X = np.asarray(clip["X"], dtype=np.complex64)
    valid_tf = np.asarray(clip["valid_tf"], dtype=np.uint8)
    freqs_rf = np.asarray(clip["freqs_rf"], dtype=np.float32)
    time_axis_s = np.asarray(clip["time_axis_s"], dtype=np.float32)
    az_gt = np.asarray(clip["az_gt"], dtype=np.float32)
    el_gt = np.asarray(clip["el_gt"], dtype=np.float32)

    out_cfg = cfg.get("output", {})
    write_quicklook_scm = bool(out_cfg.get("write_quicklook_scm_tf", False))
    write_clean_parts = bool(out_cfg.get("write_clean_parts", False))

    with h5py.File(path, "w") as f:
        f.attrs["proxy_type"] = clip["proxy_type"]
        f.attrs["split"] = clip["split"]
        f.attrs["snr_db"] = float(clip["snr_db"])
        f.attrs["sir_db"] = float(clip["sir_db"])
        f.attrs["overlap_mode"] = clip["overlap_mode"]
        f.attrs["overlap_range"] = json.dumps(clip["overlap_range"])
        f.attrs["dataset_mode"] = "signal_noise_interference_tf512"
        f.attrs["meta_json"] = json.dumps(clip["meta"], ensure_ascii=False)
        f.attrs["observation_duration_ms"] = float(clip["meta"]["observation_duration_ms"])
        f.attrs["Nt"] = int(clip["meta"]["Nt"])
        f.attrs["Nf"] = int(clip["meta"]["Nf"])
        f.attrs["df_hz"] = float(clip["meta"]["df_hz"])

        f.create_dataset("X_tf", data=ri_from_complex(X), compression="gzip")
        if bool(out_cfg.get("write_X_stft_alias", True)):
            f.create_dataset("X_stft", data=ri_from_complex(X), compression="gzip")

        if write_clean_parts:
            f.create_dataset("X_signal_tf", data=ri_from_complex(clip["X_clean"]), compression="gzip")
            f.create_dataset("X_interference_tf", data=ri_from_complex(clip["X_interf"]), compression="gzip")
            f.create_dataset("X_noise_tf", data=ri_from_complex(clip["X_noise"]), compression="gzip")

        f.create_dataset("valid_tf", data=valid_tf, compression="gzip")
        f.create_dataset("target_mask", data=np.asarray(clip["target_mask"], dtype=np.uint8), compression="gzip")
        f.create_dataset("interference_mask", data=np.asarray(clip["interference_mask"], dtype=np.uint8), compression="gzip")
        f.create_dataset("signal_tf_power", data=np.asarray(clip["signal_tf_power"], dtype=np.float32), compression="gzip")
        f.create_dataset("interf_tf_power", data=np.asarray(clip["interf_tf_power"], dtype=np.float32), compression="gzip")
        f.create_dataset("freqs_hz", data=freqs_rf, compression="gzip")
        f.create_dataset("time_axis_s", data=time_axis_s, compression="gzip")

        if write_quicklook_scm:
            R = compute_quicklook_scm_timeavg(X)
            valid_t = np.ones((X.shape[0],), dtype=np.uint8)
            f.create_dataset("scm_tf", data=ri_from_complex(R), compression="gzip")
            f.create_dataset("scm_tf_valid", data=valid_t, compression="gzip")

        g = f.create_group("labels")
        g.create_dataset("az_gt", data=az_gt, compression="gzip")
        g.create_dataset("el_gt", data=el_gt, compression="gzip")


# ============================================================
# Main
# ============================================================

def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = apply_interference_defaults(cfg)

    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.snr_db is not None:
        cfg["interference_noise"]["snr_db"] = parse_float_spec(args.snr_db)
        cfg.setdefault("dataset_plan", {}).pop("snr_scan", None)
    if args.sir_db is not None:
        cfg["interference_noise"]["sir_db"] = parse_float_spec(args.sir_db)
        cfg.setdefault("dataset_plan", {}).pop("sir_scan", None)
    if args.overlap_mode is not None:
        cfg["interference_noise"]["tf_overlap"]["mode"] = str(args.overlap_mode)
    if args.overlap_range is not None:
        cfg["interference_noise"]["tf_overlap"]["range"] = parse_float_spec(args.overlap_range)
        cfg.setdefault("dataset_plan", {}).pop("overlap_ranges", None)
    if args.enabled_interference_types is not None:
        cfg["interference_noise"]["enabled_interference_types"] = parse_types(args.enabled_interference_types)
    if args.num_clips is not None:
        n = int(args.num_clips)
        cfg.setdefault("dataset_plan", {})["base_N"] = n
        cfg["dataset_plan"]["splits"] = {"train": {"multiplier": 1}, "val": {"multiplier": 0}, "test": {"multiplier": 0}}
    if args.T is not None:
        cfg["stft"]["T"] = int(args.T)
    if args.F is not None:
        cfg["stft"]["F"] = int(args.F)
    if args.write_clean_parts is not None:
        cfg.setdefault("output", {})["write_clean_parts"] = bool(args.write_clean_parts)
    if args.write_scm is not None:
        cfg.setdefault("output", {})["write_quicklook_scm_tf"] = bool(args.write_scm)
    if args.exp6_enable:
        cfg.setdefault("exp6_controlled_jammer_type", {})["enable"] = True
    if args.exp6_split_name is not None:
        cfg.setdefault("exp6_controlled_jammer_type", {})["split_name"] = str(args.exp6_split_name)
    if args.exp6_samples_per_type is not None:
        cfg.setdefault("exp6_controlled_jammer_type", {})["samples_per_type"] = int(args.exp6_samples_per_type)
    if args.exp6_component_types is not None:
        cfg.setdefault("exp6_controlled_jammer_type", {})["component_types"] = parse_types(args.exp6_component_types)
    if args.exp6_single_component_only is not None:
        cfg.setdefault("exp6_controlled_jammer_type", {})["single_component_only"] = bool(args.exp6_single_component_only)
    if args.exp7_enable:
        cfg.setdefault("exp7_controlled_separation", {})["enable"] = True
    if args.exp7_split_name is not None:
        cfg.setdefault("exp7_controlled_separation", {})["split_name"] = str(args.exp7_split_name)
    if args.exp7_samples_per_bin is not None:
        cfg.setdefault("exp7_controlled_separation", {})["samples_per_bin"] = int(args.exp7_samples_per_bin)
    if args.exp7_sep_bin_edges_deg is not None:
        cfg.setdefault("exp7_controlled_separation", {})["sep_bin_edges_deg"] = parse_float_list(args.exp7_sep_bin_edges_deg)
    if args.exp7_component_types is not None:
        cfg.setdefault("exp7_controlled_separation", {})["component_types"] = parse_types(args.exp7_component_types)
    if args.exp7_single_component_only is not None:
        cfg.setdefault("exp7_controlled_separation", {})["single_component_only"] = bool(args.exp7_single_component_only)
    if args.exp7_max_doa_trials is not None:
        cfg.setdefault("exp7_controlled_separation", {})["max_doa_trials"] = int(args.exp7_max_doa_trials)
    if args.exp7_near_target_threshold_deg is not None:
        cfg.setdefault("exp7_controlled_separation", {})["near_target_threshold_deg"] = float(args.exp7_near_target_threshold_deg)

    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate TF512 UAV proxy observations with controllable interference, total SIR, SNR, and TF overlap."
    )
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--snr_db", type=str, default=None, help="Scalar or range, e.g. '10' or '0,20'.")
    ap.add_argument("--sir_db", type=str, default=None, help="Scalar or range, e.g. '-5' or '-10,5'.")
    ap.add_argument("--overlap_mode", type=str, default=None, choices=["none", "binary", "target_energy"])
    ap.add_argument("--overlap_range", type=str, default=None, help="Range like '0,0.05'.")
    ap.add_argument("--enabled_interference_types", type=str, default=None,
                    help="Comma separated: cw,wideband_block,fullband_impulse,fh_jammer,chirp_sweep,cw_drift")
    ap.add_argument("--num_clips", type=int, default=None,
                    help="Override dataset_plan to create N train clips per proxy/SNR/SIR/overlap condition.")
    ap.add_argument("--T", type=int, default=None, help="Debug override for time frames.")
    ap.add_argument("--F", type=int, default=None, help="Debug override for freq bins.")
    ap.add_argument("--write_clean_parts", type=int, default=None, help="1 to store X_signal_tf/X_interference_tf/X_noise_tf.")
    ap.add_argument("--write_scm", type=int, default=None, help="1 to store time-averaged scm_tf.")
    ap.add_argument("--exp6_enable", action="store_true", help="Enable controlled jammer-type exp6 test-set generation.")
    ap.add_argument("--exp6_split_name", type=str, default=None, help="Split name to use in exp6 mode, typically test.")
    ap.add_argument("--exp6_samples_per_type", type=int, default=None, help="Samples per jammer type in exp6 mode.")
    ap.add_argument("--exp6_component_types", type=str, default=None, help="Comma-separated interference component types for exp6 mode.")
    ap.add_argument("--exp6_single_component_only", type=int, default=None, help="1 to force exactly one interference component in exp6 mode.")
    ap.add_argument("--exp7_enable", action="store_true", help="Enable controlled-separation exp7 test-set generation.")
    ap.add_argument("--exp7_split_name", type=str, default=None, help="Split name to use in exp7 mode, typically test.")
    ap.add_argument("--exp7_samples_per_bin", type=int, default=None, help="Samples per separation bin in exp7 mode.")
    ap.add_argument("--exp7_sep_bin_edges_deg", type=str, default=None, help="Comma-separated separation bin edges in degrees, e.g. '0,5,10,20,40,180'.")
    ap.add_argument("--exp7_component_types", type=str, default=None, help="Comma-separated interference component types for exp7 mode.")
    ap.add_argument("--exp7_single_component_only", type=int, default=None, help="1 to force exactly one interference component in exp7 mode.")
    ap.add_argument("--exp7_max_doa_trials", type=int, default=None, help="Max resampling trials for controlled interference DOA.")
    ap.add_argument("--exp7_near_target_threshold_deg", type=float, default=None, help="Threshold to mark controlled jammer as near target.")
    ap.add_argument("--quiet", action="store_true")

    args = ap.parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, args)

    out_dir = str(cfg["out_dir"])
    prefix = str(cfg.get("clip_prefix", "clip"))
    base_seed = int(cfg.get("seed", 0))

    enabled_proxy_names = build_enabled_proxy_names(cfg)
    split_counts = build_split_counts(cfg)
    snr_values = build_snr_values(cfg)
    sir_values = build_sir_values(cfg)
    overlap_ranges = build_overlap_ranges(cfg)
    overlap_mode = str(cfg["interference_noise"].get("tf_overlap", {}).get("mode", "target_energy"))
    exp6_plan = build_exp6_plan(cfg)
    exp7_plan = build_exp7_plan(cfg)
    if exp6_plan is not None and exp7_plan is not None:
        raise ValueError("exp6 and exp7 controlled modes cannot both be enabled in the same generation run")

    os.makedirs(out_dir, exist_ok=True)

    total = 0
    if exp6_plan is None and exp7_plan is None:
        for split_name in ["train", "val", "test"]:
            repeat = int(split_counts.get(split_name, 0))
            if repeat <= 0:
                continue
            total += len(enabled_proxy_names) * len(snr_values) * len(sir_values) * len(overlap_ranges) * repeat
    elif exp6_plan is not None:
        total = (
            len(enabled_proxy_names)
            * len(snr_values)
            * len(sir_values)
            * len(overlap_ranges)
            * len(exp6_plan["component_types"])
            * int(exp6_plan["samples_per_type"])
        )
    else:
        total = (
            len(enabled_proxy_names)
            * len(snr_values)
            * len(sir_values)
            * len(overlap_ranges)
            * len(exp7_plan["bins"])
            * len(exp7_plan["component_types"])
            * int(exp7_plan["samples_per_bin"])
        )

    if not args.quiet:
        msg = (
            "Planned generation:\n"
            f"  proxies={enabled_proxy_names}\n"
            f"  snr_values={snr_values}\n"
            f"  sir_values={sir_values}\n"
            f"  overlap_mode={overlap_mode}\n"
            f"  overlap_ranges={overlap_ranges}\n"
            f"  enabled_interference_types={cfg['interference_noise'].get('enabled_interference_types', [])}\n"
        )
        if exp6_plan is None and exp7_plan is None:
            msg += f"  split_counts={split_counts}\n"
        elif exp6_plan is not None:
            msg += (
                f"  exp6_split={exp6_plan['split_name']}\n"
                f"  exp6_component_types={exp6_plan['component_types']}\n"
                f"  exp6_samples_per_type={exp6_plan['samples_per_type']}\n"
            )
        else:
            msg += (
                f"  exp7_split={exp7_plan['split_name']}\n"
                f"  exp7_sep_bins={[b['range_deg'] for b in exp7_plan['bins']]}\n"
                f"  exp7_component_types={exp7_plan['component_types']}\n"
                f"  exp7_samples_per_bin={exp7_plan['samples_per_bin']}\n"
            )
        msg += f"  total_clips={total}\n  out_dir={out_dir}"
        print(msg)

    written = 0
    duration_ms = int(round(float(cfg["stft"].get("observation_duration_ms", 50.0))))

    if exp7_plan is None:
        if exp6_plan is None:
            for split_name in ["train", "val", "test"]:
                repeat = int(split_counts.get(split_name, 0))
                if repeat <= 0:
                    continue
                split_dir = os.path.join(out_dir, split_name)
                os.makedirs(split_dir, exist_ok=True)

                for proxy_name in enabled_proxy_names:
                    for snr_db in snr_values:
                        for sir_db in sir_values:
                            for ov_range in overlap_ranges:
                                ov_lo, ov_hi = float(ov_range[0]), float(ov_range[1])
                                condition_tag = f"snr{snr_db:+g}_sir{sir_db:+g}_ov{ov_lo:.3f}-{ov_hi:.3f}_{overlap_mode}"
                                for sample_idx in range(repeat):
                                    clip = gen_one_clip_with_interference(
                                        cfg=cfg,
                                        base_seed=base_seed,
                                        forced_proxy_name=proxy_name,
                                        forced_snr_db=float(snr_db),
                                        forced_sir_db=float(sir_db),
                                        overlap_range=ov_range,
                                        overlap_mode=overlap_mode,
                                        split_name=split_name,
                                        sample_index_within_bucket=sample_idx,
                                        condition_tag=condition_tag,
                                    )

                                    snr_tag = f"snr{snr_db:+g}dB"
                                    sir_tag = f"sir{sir_db:+g}dB"
                                    ov_tag = f"ov{ov_lo:.3f}-{ov_hi:.3f}"
                                    fname = (
                                        f"{prefix}_{split_name}_{safe_filename(proxy_name)}_"
                                        f"{safe_filename(snr_tag)}_{safe_filename(sir_tag)}_"
                                        f"{safe_filename(overlap_mode)}_{safe_filename(ov_tag)}_"
                                        f"{sample_idx:05d}_{duration_ms}ms.h5"
                                    )
                                    path = os.path.join(split_dir, fname)
                                    write_h5(path, clip, cfg)
                                    written += 1

                                    if not args.quiet and (written % max(1, min(50, total)) == 0 or written == total):
                                        m = clip["meta"]
                                        print(
                                            f"[{written}/{total}] {path} | "
                                            f"SNR={m['snr_db_effective']:.3f} dB, "
                                            f"SIR={m['sir_db_effective_total']:.3f} dB, "
                                            f"ov_bin={m['overlap_binary_actual']:.4f}, "
                                            f"ov_E={m['overlap_target_energy_actual']:.4f}, "
                                            f"accepted={m['overlap_accepted']}"
                                        )
        else:
            split_name = str(exp6_plan["split_name"])
            split_dir = os.path.join(out_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            for proxy_name in enabled_proxy_names:
                for snr_db in snr_values:
                    for sir_db in sir_values:
                        for ov_range in overlap_ranges:
                            ov_lo, ov_hi = float(ov_range[0]), float(ov_range[1])
                            for component_type in exp6_plan["component_types"]:
                                condition_tag = (
                                    f"snr{snr_db:+g}_sir{sir_db:+g}_ov{ov_lo:.3f}-{ov_hi:.3f}_{overlap_mode}_"
                                    f"{component_type}"
                                )
                                for sample_idx in range(int(exp6_plan["samples_per_type"])):
                                    clip = gen_one_clip_with_interference(
                                        cfg=cfg,
                                        base_seed=base_seed,
                                        forced_proxy_name=proxy_name,
                                        forced_snr_db=float(snr_db),
                                        forced_sir_db=float(sir_db),
                                        overlap_range=ov_range,
                                        overlap_mode=overlap_mode,
                                        split_name=split_name,
                                        sample_index_within_bucket=sample_idx,
                                        condition_tag=condition_tag,
                                        exp6_component_type=str(component_type),
                                    )

                                    snr_tag = f"snr{snr_db:+g}dB"
                                    sir_tag = f"sir{sir_db:+g}dB"
                                    ov_tag = f"ov{ov_lo:.3f}-{ov_hi:.3f}"
                                    fname = (
                                        f"{prefix}_{split_name}_{safe_filename(proxy_name)}_"
                                        f"{safe_filename(snr_tag)}_{safe_filename(sir_tag)}_"
                                        f"{safe_filename(overlap_mode)}_{safe_filename(ov_tag)}_"
                                        f"{safe_filename(component_type)}_"
                                        f"{sample_idx:05d}_{duration_ms}ms.h5"
                                    )
                                    path = os.path.join(split_dir, fname)
                                    write_h5(path, clip, cfg)
                                    written += 1

                                    if not args.quiet and (written % max(1, min(50, total)) == 0 or written == total):
                                        m = clip["meta"]
                                        exp6_meta = m.get("exp6_controlled_jammer_type", {})
                                        print(
                                            f"[{written}/{total}] {path} | "
                                            f"SNR={m['snr_db_effective']:.3f} dB, "
                                            f"SIR={m['sir_db_effective_total']:.3f} dB, "
                                            f"ov_bin={m['overlap_binary_actual']:.4f}, "
                                            f"jammer={exp6_meta.get('component_type', '')}"
                                        )
    else:
        split_name = str(exp7_plan["split_name"])
        split_dir = os.path.join(out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for proxy_name in enabled_proxy_names:
            for snr_db in snr_values:
                for sir_db in sir_values:
                    for ov_range in overlap_ranges:
                        ov_lo, ov_hi = float(ov_range[0]), float(ov_range[1])
                        for bin_item in exp7_plan["bins"]:
                            sep_range = bin_item["range_deg"]
                            sep_label = str(bin_item["label"])
                            for component_type in exp7_plan["component_types"]:
                                condition_tag = (
                                    f"snr{snr_db:+g}_sir{sir_db:+g}_ov{ov_lo:.3f}-{ov_hi:.3f}_{overlap_mode}_"
                                    f"{sep_label}_{component_type}"
                                )
                                for sample_idx in range(int(exp7_plan["samples_per_bin"])):
                                    clip = gen_one_clip_with_interference(
                                        cfg=cfg,
                                        base_seed=base_seed,
                                        forced_proxy_name=proxy_name,
                                        forced_snr_db=float(snr_db),
                                        forced_sir_db=float(sir_db),
                                        overlap_range=ov_range,
                                        overlap_mode=overlap_mode,
                                        split_name=split_name,
                                        sample_index_within_bucket=sample_idx,
                                        condition_tag=condition_tag,
                                        exp7_sep_range_deg=sep_range,
                                        exp7_sep_label=sep_label,
                                        exp7_component_type=str(component_type),
                                    )

                                    snr_tag = f"snr{snr_db:+g}dB"
                                    sir_tag = f"sir{sir_db:+g}dB"
                                    ov_tag = f"ov{ov_lo:.3f}-{ov_hi:.3f}"
                                    sep_tag = f"sep{float(sep_range[0]):.1f}-{float(sep_range[1]):.1f}deg"
                                    fname = (
                                        f"{prefix}_{split_name}_{safe_filename(proxy_name)}_"
                                        f"{safe_filename(snr_tag)}_{safe_filename(sir_tag)}_"
                                        f"{safe_filename(overlap_mode)}_{safe_filename(ov_tag)}_"
                                        f"{safe_filename(sep_tag)}_{safe_filename(component_type)}_"
                                        f"{sample_idx:05d}_{duration_ms}ms.h5"
                                    )
                                    path = os.path.join(split_dir, fname)
                                    write_h5(path, clip, cfg)
                                    written += 1

                                    if not args.quiet and (written % max(1, min(50, total)) == 0 or written == total):
                                        m = clip["meta"]
                                        exp7_meta = m.get("exp7_controlled_separation", {})
                                        print(
                                            f"[{written}/{total}] {path} | "
                                            f"SNR={m['snr_db_effective']:.3f} dB, "
                                            f"SIR={m['sir_db_effective_total']:.3f} dB, "
                                            f"ov_bin={m['overlap_binary_actual']:.4f}, "
                                            f"sep={exp7_meta.get('sep_label', '')}, "
                                            f"jammer={exp7_meta.get('component_type', '')}"
                                        )

    if not args.quiet:
        print("Done.")


if __name__ == "__main__":
    main()
