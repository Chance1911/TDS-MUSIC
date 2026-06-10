from __future__ import annotations
import argparse
import json
import math
import os
from typing import Any, Dict, List

import h5py
import numpy as np
import yaml

C0 = 299_792_458.0


def ri_from_complex(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex64)
    return np.stack([x.real.astype(np.float32), x.imag.astype(np.float32)], axis=-1)


def is_num(x) -> bool:
    return isinstance(x, (int, float, np.number))


def pick_float(rng: np.random.Generator, spec) -> float:
    if is_num(spec):
        return float(spec)
    if isinstance(spec, (list, tuple)):
        if len(spec) == 2 and is_num(spec[0]) and is_num(spec[1]):
            a, b = float(spec[0]), float(spec[1])
            lo, hi = (a, b) if a <= b else (b, a)
            return float(rng.uniform(lo, hi))
        return float(rng.choice(spec))
    raise TypeError(f"bad float spec: {spec}")


def pick_choice(rng: np.random.Generator, xs):
    return rng.choice(list(xs))


def safe_filename(s: str) -> str:
    return "".join([c if (c.isalnum() or c in "-_") else "_" for c in s])


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


# ---------- array ----------
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


def steering_vectors(
    pos: np.ndarray,
    freqs_rf: np.ndarray,
    az_deg: float,
    el_deg: float,
    mode: str,
    fc_hz: float,
) -> np.ndarray:
    u = unit_vec_from_az_el(az_deg, el_deg)
    proj = pos @ u  # [M]
    if mode == "center_freq":
        k = 2.0 * np.pi * fc_hz / C0
        a = np.exp(-1j * k * proj)[None, :]
        return np.repeat(a, len(freqs_rf), axis=0).astype(np.complex64)
    k = 2.0 * np.pi * freqs_rf / C0
    return np.exp(-1j * (k[:, None] * proj[None, :])).astype(np.complex64)


# ---------- sources on unified TF grid ----------
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

    # time occupancy on the unified 50 ms / Nt grid
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


def apply_array_errors(rng, X, cfg):
    M = X.shape[-1]
    gain_db = rng.normal(0.0, float(cfg.get("gain_std_db", 0.0)), size=(M,))
    phase_deg = rng.normal(0.0, float(cfg.get("phase_std_deg", 0.0)), size=(M,))
    g = (10.0 ** (gain_db / 20.0)) * np.exp(1j * np.deg2rad(phase_deg))
    return X * g[None, None, :]


def compute_quicklook_scm_timeavg(X: np.ndarray) -> np.ndarray:
    """Optional quicklook SCM: average over frequency for each time bin, shape [Nt,M,M]."""
    Nt, Nf, M = X.shape
    R = np.zeros((Nt, M, M), dtype=np.complex64)
    for t in range(Nt):
        Xt = X[t, :, :]
        R[t] = (Xt.T @ Xt.conj()) / max(1, Nf)
    return R


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


def build_snr_values(cfg: Dict[str, Any]) -> List[float]:
    scan = cfg.get("dataset_plan", {}).get("snr_scan", {})
    start = int(scan.get("start_db", -20))
    stop = int(scan.get("stop_db", 20))
    step = int(scan.get("step_db", 2))
    return [float(v) for v in range(start, stop + 1, step)]


def build_split_counts(cfg: Dict[str, Any]) -> Dict[str, int]:
    plan = cfg.get("dataset_plan", {})
    base_n = int(plan.get("base_N", 1))
    out: Dict[str, int] = {}
    for split_name in ["train", "val", "test"]:
        multiplier = int(plan.get("splits", {}).get(split_name, {}).get("multiplier", 1))
        out[split_name] = base_n * multiplier
    return out


def build_time_axis_s(cfg: Dict[str, Any]) -> np.ndarray:
    Nt = int(cfg["stft"]["T"])
    duration_s = float(cfg["stft"]["observation_duration_ms"]) * 1e-3
    return np.linspace(0.0, duration_s, Nt, endpoint=False, dtype=np.float64)


def build_freq_axis_hz(cfg: Dict[str, Any]) -> np.ndarray:
    Nf = int(cfg["stft"]["F"])
    f_min_hz = float(cfg["stft"]["f_min_hz"])
    f_max_hz = float(cfg["stft"]["f_max_hz"])
    # use bin centers for clearer physical meaning
    df = (f_max_hz - f_min_hz) / Nf
    return (f_min_hz + (np.arange(Nf, dtype=np.float64) + 0.5) * df)


def gen_one_clip_signal_noise_only(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    forced_proxy_name: str,
    forced_snr_db: float,
    split_name: str,
    sample_index_within_bucket: int,
) -> Dict[str, Any]:
    array = cfg["array"]
    stft = cfg["stft"]
    Nt = int(stft["T"])
    Nf = int(stft["F"])
    time_axis_s = build_time_axis_s(cfg)
    f_axis = build_freq_axis_hz(cfg)

    fc_hz = float(array["center_freq_hz"])
    freqs_rf = f_axis + fc_hz
    pos = uca_positions(int(array["M"]), float(array["radius_m"]))
    steer_mode = str(array.get("steering", {}).get("mode", "per_bin"))

    doa = cfg["scene"]["doa"]
    az = float(pick_float(rng, doa.get("az_range_deg", [0, 360])))
    el = float(pick_float(rng, doa.get("el_range_deg", [0, 60])))
    A = steering_vectors(pos, freqs_rf, az, el, steer_mode, fc_hz)  # [Nf,M]

    proxy_cfg = find_proxy_cfg(cfg, forced_proxy_name)
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

    Xs = (S_total[:, :, None] * A[None, :, :]).astype(np.complex64)  # [Nt,Nf,M]

    in_cfg = cfg["interference_noise"]
    if bool(in_cfg.get("enable_interference", False)):
        raise ValueError("This generator is for signal+noise only. Set enable_interference=False.")
    if not bool(in_cfg.get("enable_noise", True)):
        raise ValueError("This generator expects enable_noise=True.")

    N = (rng.standard_normal(Xs.shape) + 1j * rng.standard_normal(Xs.shape)).astype(np.complex64)
    pn = np.mean(np.abs(N) ** 2)
    if pn > 0:
        N /= np.sqrt(pn)

    Ps = float(np.mean(np.abs(Xs) ** 2) + 1e-12)
    snr_db = float(forced_snr_db)
    Pn_target = Ps / (10.0 ** (snr_db / 10.0))
    Pn_cur = float(np.mean(np.abs(N) ** 2) + 1e-12)
    N *= np.sqrt(Pn_target / Pn_cur)

    X = (Xs + N).astype(np.complex64)
    X = apply_array_errors(rng, X, cfg.get("errors", {}).get("per_clip_random", {}))

    valid_tf = np.ones((Nt, Nf), dtype=np.uint8)

    Ps_after = float(np.mean(np.abs(Xs) ** 2) + 1e-12)
    Pn_after = float(np.mean(np.abs(N) ** 2) + 1e-12)
    snr_eff_db = 10.0 * np.log10(Ps_after / Pn_after) if Pn_after > 0 else float("inf")

    duration_ms = float(stft["observation_duration_ms"])
    df_hz = float((float(stft["f_max_hz"]) - float(stft["f_min_hz"])) / Nf)

    meta = {
        "dataset_mode": "signal_plus_noise_only_tf512",
        "split": split_name,
        "sample_index_within_bucket": int(sample_index_within_bucket),
        "proxy_type": forced_proxy_name,
        "fhss_present": bool(fh_present),
        "Nt": Nt,
        "Nf": Nf,
        "observation_duration_ms": duration_ms,
        "df_hz": df_hz,
        "snr_db": float(snr_db),
        "snr_eff_db": float(snr_eff_db),
        "az": az,
        "el": el,
        "center_freq_hz": fc_hz,
        "f_min_hz": float(stft["f_min_hz"]),
        "f_max_hz": float(stft["f_max_hz"]),
        "steering_mode": steer_mode,
    }
    return {
        "X": X,
        "valid_tf": valid_tf,
        "freqs_rf": freqs_rf,
        "time_axis_s": time_axis_s,
        "az_gt": np.full((Nt,), az, dtype=np.float32),
        "el_gt": np.full((Nt,), el, dtype=np.float32),
        "proxy_type": forced_proxy_name,
        "split": split_name,
        "snr_db": float(snr_db),
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

    write_quicklook_scm = bool(cfg.get("output", {}).get("write_quicklook_scm_tf", False))

    with h5py.File(path, "w") as f:
        f.attrs["proxy_type"] = clip["proxy_type"]
        f.attrs["split"] = clip["split"]
        f.attrs["snr_db"] = float(clip["snr_db"])
        f.attrs["dataset_mode"] = "signal_plus_noise_only_tf512"
        f.attrs["meta_json"] = json.dumps(clip["meta"], ensure_ascii=False)
        f.attrs["observation_duration_ms"] = float(clip["meta"]["observation_duration_ms"])
        f.attrs["Nt"] = int(clip["meta"]["Nt"])
        f.attrs["Nf"] = int(clip["meta"]["Nf"])
        f.attrs["df_hz"] = float(clip["meta"]["df_hz"])

        # unified raw observation model
        f.create_dataset("X_tf", data=ri_from_complex(X), compression="gzip")
        # optional backward-compat alias
        if bool(cfg.get("output", {}).get("write_X_stft_alias", True)):
            f.create_dataset("X_stft", data=ri_from_complex(X), compression="gzip")

        f.create_dataset("valid_tf", data=valid_tf, compression="gzip")
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate unified 512x512x8x2 complex TF observations.")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_dir = str(cfg["out_dir"])
    prefix = str(cfg.get("clip_prefix", "clip"))
    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)

    enabled_proxy_names = build_enabled_proxy_names(cfg)
    split_counts = build_split_counts(cfg)
    snr_values = build_snr_values(cfg)
    os.makedirs(out_dir, exist_ok=True)

    total = sum(len(enabled_proxy_names) * len(snr_values) * repeat for repeat in split_counts.values())
    print(
        f"Planned generation: proxies={enabled_proxy_names}, snr_values={snr_values}, "
        f"split_counts={split_counts}, total_clips={total}"
    )

    written = 0
    duration_ms = int(round(float(cfg["stft"]["observation_duration_ms"])))
    for split_name in ["train", "val", "test"]:
        repeat = split_counts[split_name]
        split_dir = os.path.join(out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for proxy_name in enabled_proxy_names:
            for snr_db in snr_values:
                snr_tag = f"snr{int(snr_db):+03d}dB" if float(snr_db).is_integer() else f"snr{snr_db:+.1f}dB"
                for sample_idx in range(repeat):
                    clip = gen_one_clip_signal_noise_only(
                        cfg=cfg,
                        rng=rng,
                        forced_proxy_name=proxy_name,
                        forced_snr_db=snr_db,
                        split_name=split_name,
                        sample_index_within_bucket=sample_idx,
                    )
                    fname = (
                        f"{prefix}_{split_name}_{safe_filename(proxy_name)}_{snr_tag}_"
                        f"{sample_idx:05d}_{duration_ms}ms.h5"
                    )
                    path = os.path.join(split_dir, fname)
                    write_h5(path, clip, cfg)
                    written += 1
                    if written % max(1, min(50, total)) == 0 or written == total:
                        print(f"[{written}/{total}] wrote {path}")
    print("Done.")


if __name__ == "__main__":
    main()
