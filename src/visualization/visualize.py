import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def plot_sv_rgb(
    sv: xr.DataArray,
    outfile: str = "Sv_RGB_38_70_120kHz.png",
    figsize: tuple = (18, 8),
    vmin: float = -90,
    vmax: float = -50,
    time_slice: slice = slice(None),
    depth_slice: slice = slice(None),
):
    """
    Plot Sv as an RGB image using 38 kHz (R), 70 kHz (G), 120 kHz (B) channels.

    Parameters
    ----------
    sv : xarray.DataArray
        The Sv DataArray with dims (channel, time, depth).
    outfile : str
        Path to save the output figure (PNG).
    figsize : tuple
        Size of the figure (width, height).
    vmin, vmax : float
        Color scale limits (in dB).
    time_slice, depth_slice : slice
        Slices for time and depth dimensions.
    """
    import matplotlib.pyplot as plt

    # Channel frequencies for RGB
    rgb_freqs = [38, 70, 120]
    rgb_idx = []
    for f in rgb_freqs:
        idx = np.where(np.isclose(sv.channel.values, f))[0]
        if len(idx) == 0:
            raise ValueError(f"Channel {f} kHz not found in sv.channel.")
        rgb_idx.append(idx[0])

    # Select and normalize Sv for each channel
    sv_rgb = []
    for i in rgb_idx:
        arr = sv.isel(channel=i, time=time_slice, depth=depth_slice).T.values
        arr = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
        arr = np.nan_to_num(arr, nan=0.0)
        sv_rgb.append(arr)
    # Stack to (depth, time, 3)
    rgb_img = np.stack(sv_rgb, axis=-1)

    # Prepare axes for plotting (like plot_sv_channels_faceted)
    time_vals = sv["time"].values[time_slice]
    depth_vals = sv["depth"].values[depth_slice]

    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(
        rgb_img,
        aspect="auto",
        origin="lower",
        extent=[
            time_vals[0], time_vals[-1],
            depth_vals[0], depth_vals[-1]
        ]
    )
    ax.invert_yaxis()
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Sv RGB composite: R=38kHz, G=70kHz, B=120kHz")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"RGB Sv image saved as '{outfile}'")