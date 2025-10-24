import argparse
from pathlib import Path
import numpy as np
from datetime import datetime

from .data_config import BASE_DIR
from .io import load_survey_ds, print_file_infos

def main(ei, 
         t_start, 
         t_end,
         z_min, 
         z_max, 
         save_folder: Path):
    
    save_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Lazy loading {ei} echointegration netCDF file...")
    ds = load_survey_ds(survey=ei, chunks={"time": 1000, "depth": 100}) # lazy loading whole survey echointegration
    print("Loading complete.")

    print("\nExtracting file info...")
    print_file_infos(ds)

    # Convert datatime strings to numpy.datetime64
    t_start_dt = np.datetime64(datetime.fromisoformat(t_start))
    t_end_dt = np.datetime64(datetime.fromisoformat(t_end))

    # Nearest scalar endpoints
    time_start_sel = ds['time'].sel(time=t_start_dt, method="nearest")
    time_end_sel = ds['time'].sel(time=t_end_dt, method="nearest")

    # Depth endpoints (optional if z_min/z_max might not match exactly)
    depth_start_sel = ds['depth'].sel(depth=z_min, method="nearest")
    depth_end_sel = ds['depth'].sel(depth=z_max, method="nearest")

    # Subset
    ds_subset = ds.sel(
        time=slice(time_start_sel, time_end_sel),
        depth=slice(depth_start_sel, depth_end_sel)
    )

    print("\nNew xarray.Dataset info:")
    print_file_infos(ds_subset)

    # Get the exact indices of the subset for clarity in the file name
    time_index_start = int(np.where(ds['time'].values == ds_subset['time'].values[0])[0][0])
    time_index_end = int(np.where(ds['time'].values == ds_subset['time'].values[-1])[0][0]) + 1

    depth_index_start = int(np.where(ds['depth'].values == ds_subset['depth'].values[0])[0][0])
    depth_index_end = int(np.where(ds['depth'].values == ds_subset['depth'].values[-1])[0][0]) + 1

    print(f"\nSaving subset to netCDF format...")
    file_name = f"{ei}_subset_T{time_index_start}-{time_index_end}_Z{depth_index_start}-{depth_index_end}.nc"
    ds_subset.to_netcdf(save_folder / file_name)
    print(f"\nSaved file to {save_folder / file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subset echosounder netCDF data.")
    parser.add_argument("--ei", required=True, help="Echointegration name (see data_config.py)")
    parser.add_argument("--t_start", required=True, help="Start time (ISO format, e.g. 2021-08-29T12:00:00)")
    parser.add_argument("--t_end", required=True, help="End time (ISO format)")
    parser.add_argument("--z_min", type=float, default=0., help="Minimum depth in meters")
    parser.add_argument("--z_max", type=float, default=9999., help="Maximum depth in meters")
    parser.add_argument("--save_folder", type=Path, default=BASE_DIR / "data/interim/acoustic_data_subsets/", help="Folder to save the data subset")

    args = parser.parse_args()
    main(args.ei, args.t_start, args.t_end, args.z_min, args.z_max, args.save_folder)

# Example call
# From project root:
# $ python -m src.data.create_subset --ei "amazomix_3pings1m" --t_start "2021-09-09T00:00:00" --t_end "2021-09-09T23:59:59" --save_folder data/interim/acoustic_data_subsets/