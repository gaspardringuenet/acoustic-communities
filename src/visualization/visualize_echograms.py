import os
import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
from skimage import measure  # for contours/outlines

def overlay_masks(ax: matplotlib.axes.Axes,
                  mask_da,
                  time_vals,
                  depth_vals,
                  rgb_img=None,
                  alpha: float = 0.35,
                  cmap: str = "turbo",
                  draw_edges: bool = True,
                  darken_background: bool = True):
    """
    Overlay class masks on top of an echogram with natural color inside layers,
    darkened Sv outside, and crisp outlines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis from echogram figure.
    mask_da : xr.DataArray
        2D DataArray (time, depth) with integer class labels.
    time_vals, depth_vals : np.ndarray
        Coordinates for extent alignment.
    rgb_img : np.ndarray
        The echogram RGB image used as base (depth x time x 3).
    alpha : float
        Transparency of layer overlay.
    cmap : str
        Colormap for continuous layer colors.
    draw_edges : bool
        Draw contour outlines around layer regions.
    darken_background : bool
        Dim the Sv intensity outside layers.
    """
    mask = mask_da.T.values  # depth x time
    mask = np.nan_to_num(mask, nan=-1)  # -1 means background
    unique_classes = np.unique(mask[mask >= 0])
    n_classes = len(unique_classes)

    cmap_obj = plt.cm.get_cmap(cmap, n_classes)
    class_colors = {cls: cmap_obj(i / max(1, n_classes - 1)) for i, cls in enumerate(unique_classes)}

    # --- Step 1: Darken background if requested
    if darken_background and rgb_img is not None:
        # Create a dimmed copy of Sv everywhere
        dim_factor = 0.6
        rgb_dim = rgb_img * dim_factor

        # Build a composite: dim everywhere, keep bright inside masks
        mask_binary = (mask >= 0).astype(float)[..., np.newaxis]
        rgb_composite = rgb_dim * (1 - mask_binary) + rgb_img * mask_binary

        ax.imshow(
            rgb_composite,
            aspect="auto",
            origin="upper",
            extent=[time_vals[0], time_vals[-1], depth_vals[-1], depth_vals[0]]
        )
    else:
        # Default: just show Sv image (already plotted in base function)
        pass

    # --- Step 2: Plot outlines and subtle layer tint (optional)
    time_numeric = mdates.date2num(time_vals)

    for cls in unique_classes:
        color = class_colors[cls]
        cls_mask = np.where(mask == cls, 1, np.nan)

        # Optional faint color tint overlay inside each class
        ax.imshow(cls_mask,
                  cmap=matplotlib.colors.ListedColormap([color]),
                  alpha=alpha,
                  aspect="auto",
                  origin="upper",
                  extent=[time_vals[0], time_vals[-1], depth_vals[-1], depth_vals[0]])

        if draw_edges:
            contours = measure.find_contours(np.nan_to_num(cls_mask, nan=0), level=0.5)
            for contour in contours:
                x_idx = contour[:, 1]
                y_idx = contour[:, 0]
                x = np.interp(x_idx + 0.5, [0, mask.shape[1]], [time_numeric[0], time_numeric[-1]])
                y = np.interp(y_idx + 0.5, [0, mask.shape[0]], [depth_vals[0], depth_vals[-1]])
                ax.plot_date(x, y, color="white", lw=2*1.2, fmt="-", alpha=0.8)
                ax.plot_date(x, y, color=color, lw=2*0.8, fmt="-", alpha=0.9)

    # --- Step 3: Legend
    handles = [
        plt.Line2D([0], [0], color=class_colors[cls], lw=4, label=f"Layer {int(cls)}")
        for cls in unique_classes
    ]
    ax.legend(handles=handles, title="Layers", loc="upper right", fontsize=9)


def plot_sv_rgb(sv: xr.DataArray,
                outpath: str = "output/image",
                ei: str = None,
                figsize: tuple = (18, 8),
                vmin: float = -90,
                vmax: float = -50,
                rgb_freqs: list[int] | int = [38, 70, 120],
                time_slice: slice = slice(None),
                depth_slice: slice = slice(None),
                class_var: str = None,
                mask_alpha: float = None):
    """
    Plot Sv as an RGB image using 38 kHz (R), 70 kHz (G), 120 kHz (B) channels

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
    rgb_freqs : list[int]
        List of the frequency channels in order [R, G, B]. If only one channel is given as int, a greyscale image is plotted.
    time_slice, depth_slice : slice
        Slices for time and depth dimensions. Use indices.
    class_var : str
        Name of the xr.DataArray variable containing the class index. If not none, classes will be plotted on top of the echogram as transparent masks.
    """

    # RGB (or grayscale) setup
    if isinstance(rgb_freqs, int):
        print(f"Only one frequency ({rgb_freqs} kHz) given - plotting greyscale image.")
        single_channel = True
        file_desc = f"greyscale_{rgb_freqs}kHz"
        rgb_freqs = [rgb_freqs] * 3
    else:
        assert len(rgb_freqs) == 3, "RGB plot needs 3 frequencies as input"
        single_channel = False
        file_desc = f"RGB_{rgb_freqs[0]}_{rgb_freqs[1]}_{rgb_freqs[2]}kHz"

    # Finding channels indices
    rgb_idx = []
    for f in rgb_freqs:
        idx = np.where(np.isclose(sv.channel.values, f))[0]
        if len(idx) == 0:
            raise ValueError(f"Channel {f} kHz not found in sv.channel.")
        rgb_idx.append(idx[0])

    # Normalize and stack Sv for each channel
    sv_rgb = []
    for i in rgb_idx:
        arr = sv.isel(channel=i, time=time_slice, depth=depth_slice).T.values
        arr = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
        arr = np.nan_to_num(arr, nan=0.0)
        sv_rgb.append(arr)
    rgb_img = np.stack(sv_rgb, axis=-1)

    # Prepare axes for plotting
    time_vals = sv["time"].values[time_slice]
    depth_vals = sv["depth"].values[depth_slice]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(
        rgb_img,
        aspect="auto",
        origin="upper",
        extent=[
            time_vals[0], time_vals[-1],
            depth_vals[-1], depth_vals[0]
        ]
    )
    ax.invert_yaxis()
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Depth (m)")

    title = f"Sv RGB composite: R={rgb_freqs[0]}kHz, G={rgb_freqs[1]}kHz, B={rgb_freqs[2]}kHz" if not single_channel else f"Sv greyscale (channel={rgb_freqs[0]}kHz)"
    ax.set_title(title)

    # Optional class overlay
    if class_var is not None:
        if class_var in sv.coords:
            mask_da = sv[class_var].isel(time=time_slice, depth=depth_slice)
        else:
            raise ValueError(f"class_var '{class_var}' not found in sv coordinates.")
        overlay_masks(ax,
                      mask_da,
                      time_vals,
                      depth_vals,
                      rgb_img=rgb_img,
                      alpha=mask_alpha if mask_alpha is not None else 0.3)

    # Create output file path
    ei = f"{ei}_" if ei else ""
    class_var = f"_{class_var}" if class_var else ""

    t0, t1 = time_slice.start or 0, time_slice.stop or rgb_img.shape[1]
    z0, z1 = depth_slice.start or 0, depth_slice.stop or rgb_img.shape[0]

    outfile = os.path.join(
        outpath,
        f"{ei}Sv_echogram_{file_desc}_T{t0}-{t1}_Z{z0}-{z1}_clip{vmin}-{vmax}dB{class_var}.png"
    )

    # Save figure
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"RGB Sv image saved as '{outfile}'")