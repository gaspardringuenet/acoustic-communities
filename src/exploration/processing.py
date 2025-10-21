import numpy as np
import xarray as xr


def ds_to_sv(ds: xr.Dataset,
             sv_threshold: float = -150):
    sv = ds["Sv"]
    mask = sv.min(dim="channel") > sv_threshold
    return sv.where(mask)


def filter_depth (sv: xr.DataArray,
                  max_depth: float = 200):
    return sv.sel(depth=slice(0, max_depth))


def get_ch_list(sv: xr.DataArray,
                max_freq: float = 200,
                ch_ref: float = 38):
    return sv['channel'].values[(sv['channel'].values != ch_ref) & (sv['channel'].values <= max_freq)]


def compute_differences(sv: xr.DataArray,
                        ch_ref: int = 38,
                        ch_list: list[int] = [70, 120, 200]):
    sv_ref = sv.sel(channel=ch_ref)
    return {
        ch: sv.sel(channel=ch) - sv_ref
        for ch in ch_list
    }


def flatten_valid(*arrays):
    flats = [arr.values.ravel() for arr in arrays]
    mask = np.all([~np.isnan(f) for f in flats], axis=0)
    return [f[mask] for f in flats]


def sample_safe(*arrays, n_samples: int):
    assert all(len(arr) == len(arrays[0]) for arr in arrays[1:]), "Arrays should have the same length."
    if n_samples > len(arrays[0]):
        print(f"n_samples ({n_samples}) > array length ({len(arrays[0])}), keeping all elements...")
    n = min(len(arrays[0]), n_samples)
    idx = np.random.choice(len(arrays[0]), n)
    return [arr[idx] for arr in arrays]