import numpy as np
import xarray as xr

from ..data.io import load_survey_ds, print_file_infos
from ..data.data_config import BASE_DIR
from ..exploration.processing import ds_to_sv
from ..visualization.visualize_echograms import plot_sv_rgb

EI = "amazomix_3pings1m"
SV_THRESHOLD = -120  
frame = 100
start_index = 348984              # time index as displayed in Matecho
end_index = start_index + frame   # added frame length

# Import whole survey file
print(f"Lazy loading {EI} echointegration netCDF file...")
ds = load_survey_ds(survey=EI)
print("Loading complete.")

# Extract the `Sv` variable as `xarray.DataArray` and apply a threshold
print(f"\nConverting xarray.Dataset to xarray.DataArray for 'Sv' variable, and applying {SV_THRESHOLD} dB threshold...")
sv = ds_to_sv(ds, sv_threshold=SV_THRESHOLD)
print("Complete.")

# Create simplistic layer mask
print(f"Adding a 'layer' coordinate to `sv` to act as a mask...")
borders = [40,
           192,
           241,
           252,
           283,
           298,
           339,
           439,
           499] # FSSL borders identified visually on echogram

depth = sv["depth"]
time = sv["time"]

assert np.all(np.diff(depth.values) > 0), "Depth coordinate must be ascending for np.digitize"
borders_full = np.concatenate([[float(depth[0])], borders, [float(depth[-1])]]) # Build full borders (include min and max of depth)

layer_idx = np.digitize(depth, borders_full) - 1  # subtract 1 for 0-based layer indices

layer_arr = np.full((len(time), len(depth)), np.nan)
layer_arr[start_index:start_index + frame, :] = layer_idx

layer_idx_da = xr.DataArray(    # 3Make a new DataArray for the layer index (aligned with depth)
    layer_arr,
    coords={"time": sv.time, "depth": sv.depth},
    dims=["time", "depth"],
    name="layer"
)

sv = sv.assign_coords(layer=layer_idx_da)   # Assign as coordinate
print("Layers added.")

# Plot rgb
outpath = BASE_DIR / "notebooks/02-gr-delta-sv-dist-by-layers/output/figures/test"
time_slice = slice(start_index-50, end_index+50)

i_zmax = np.argmin(np.abs(depth.values - 300))
depth_slice = slice(0, i_zmax)

print(f"\nPlotting echograms for time slice ({time_slice.start}-{time_slice.stop}) and depth slice () to folder:\n{outpath}")
print()

plot_sv_rgb(
    sv,
    outpath=outpath,
    ei=EI,
    time_slice=time_slice,
    depth_slice=depth_slice,
    rgb_freqs=[38, 70, 120],
    class_var="layer",
    mask_alpha=0.
)

plot_sv_rgb(
    sv,
    outpath=outpath,
    ei=EI,
    time_slice=time_slice,
    depth_slice=depth_slice,
    rgb_freqs=38,
    class_var="layer",
    mask_alpha=0.
)


""" print()

plot_sv_rgb(
    sv,
    outpath=outpath,
    ei=EI,
    time_slice=time_slice,
)
print()

plot_sv_rgb(
    sv,
    outpath=outpath,
    ei=EI,
    time_slice=time_slice,
    depth_slice=slice(200, 700)
)
print()

plot_sv_rgb(
    sv,
    outpath=outpath,
    ei=EI,
    time_slice=time_slice,
    vmin=-70.,
    vmax=-55.
)
 """


