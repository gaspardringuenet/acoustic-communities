import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr

from ..data.data_config import BASE_DIR

def plot_survey_map(ds: xr.Dataset,
                    outfile: str,
                    bbox: tuple[float, float, float, float] | str = (-53, -31, -10, 6), # Tuple of float (xmin, xmax, ymin, ymax) or "auto"
                    margin: float = 1.):

    lats = ds['latitude'].values
    lons = ds['longitude'].values
    legs = ds['leg'].values

    if bbox == "auto":
        bbox = (lons.min()-margin, lons.max()+margin, lats.min()-margin, lats.max()+margin)
    else:
        bbox = (bbox[0]-margin, bbox[1]+margin, bbox[2]-margin, bbox[3]+margin)

    # Create figure and axis with a PlateCarree projection
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Using NOAA ETOPO1 data
    # Download: https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/bedrock/grid_registered/netcdf/
    etopo = xr.open_dataset(BASE_DIR / "data/external/spatial/bathymetry/ETOPO1_Bed_g_gmt4.grd")
    bathy = etopo['z']

    # Subset bathymetry to your region
    bathy_region = bathy.sel(x=slice(bbox[0], bbox[1]),
                             y=slice(bbox[2], bbox[3]))

    # Plot bathymetry
    ax.contourf(
        bathy_region['x'], bathy_region['y'], bathy_region,
        levels=np.arange(-6000, 6000, 500),
        cmap='Grays', transform=ccrs.PlateCarree()
    )

    # Add land and coastlines
    ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)

    # Optionally add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Plot vessel tracks grouped by leg
    for leg in np.unique(legs):
        mask = legs == leg
        leg_num = leg.split("_")[-1][3:]
        ax.plot(
            lons[mask], lats[mask],
            label=f'Vessel track LEG {leg_num}',
            linewidth=0.7,
            transform=ccrs.PlateCarree()
        )

    # Set map extent around the vessel track
    ax.set_extent([bbox[0], bbox[1], bbox[2], bbox[3]])

    # Add title and legend
    plt.title('')
    plt.legend()
    plt.savefig(outfile, bbox_inches='tight')
    plt.close();


def plot_sv_channels_faceted(sv: xr.DataArray,
                             outfile: str = "Sv_all_channels_faceted.png",
                             cmap: str = "inferno",
                             figsize: tuple = (20, 12),
                             vmin: float | None = None,
                             vmax: float | None = None,
                             sample_pixels: int | None = 2_000_000,
                             verbose: bool = False):
    """
    Plot all Sv channels as facets (subplots) with a single shared colorbar.
    Efficient for very large xarray DataArrays by subsampling.

    Parameters
    ----------
    sv : xarray.DataArray
        The Sv DataArray with dims (channel, time, depth).
    outfile : str
        Path to save the output figure (PNG).
    cmap : str
        Matplotlib colormap.
    figsize : tuple
        Size of the full figure (width, height).
    vmin, vmax : float or None
        Optional fixed colorbar limits (in dB).
    sample_pixels : int or None
        If set, subsample the data to have at most this many pixels per channel for plotting and color scaling.
    """
    n_channels = sv.sizes["channel"]
    fig, axes = plt.subplots(
        n_channels, 1,
        figsize=figsize,
        constrained_layout=True,
        sharex=True
    )
    if n_channels == 1:
        axes = [axes]

    # Subsampling for efficiency
    time_dim = sv.sizes["time"]
    depth_dim = sv.sizes["depth"]
    max_pixels = sample_pixels if sample_pixels is not None else time_dim * depth_dim
    stride_t = max(1, int(np.ceil(np.sqrt((time_dim * depth_dim) / max_pixels))))
    stride_d = stride_t

    if verbose:
        print(f"Subsampling for plotting: every {stride_t} time steps and every {stride_d} depth levels.")

    # Use a small chunk for vmin/vmax calculation
    if vmin is None or vmax is None:
        sv_sample = sv.isel(
            time=slice(0, time_dim, stride_t),
            depth=slice(0, depth_dim, stride_d)
        ).values
        if vmin is None:
            vmin = float(np.nanpercentile(sv_sample, 2))
        if vmax is None:
            vmax = float(np.nanpercentile(sv_sample, 98))

    imgs = []
    for i, ax in enumerate(axes):
        freq = sv.channel.values[i]
        sv_ch = sv.isel(channel=i)
        # Subsample for plotting
        sv_ch_plot = sv_ch.isel(
            time=slice(0, time_dim, stride_t),
            depth=slice(0, depth_dim, stride_d)
        )
        img = ax.pcolormesh(
            sv_ch_plot["time"].values,
            sv_ch_plot["depth"].values,
            sv_ch_plot.T.values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        ax.invert_yaxis()
        ax.set_title(f"Channel {freq:.0f} kHz", fontsize=12)
        ax.set_ylabel("Depth (m)")
        imgs.append(img)

    axes[-1].set_xlabel("Time (UTC)")

    # Add a single colorbar for all subplots
    cbar = fig.colorbar(imgs[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label("Sv (dB)")

    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    plt.close(fig)
    if verbose:
        print(f"\nFigure saved as '{outfile}'")


def plot_sv_rgb(sv: xr.DataArray,
                outpath: str = "output/image",
                ei: str = None,
                figsize: tuple = (18, 8),
                vmin: float = -90,
                vmax: float = -50,
                rgb_freqs: list[int] | int = [38, 70, 120],
                time_slice: slice = slice(None),
                depth_slice: slice = slice(None),
                class_var: str = None):
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
        origin="lower",
        extent=[
            time_vals[0], time_vals[-1],
            depth_vals[0], depth_vals[-1]
        ]
    )
    ax.invert_yaxis()
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Depth (m)")

    title = f"Sv RGB composite: R={rgb_freqs[0]}kHz, G={rgb_freqs[1]}kHz, B={rgb_freqs[2]}kHz" if not single_channel else f"Sv greyscale (channel={rgb_freqs[0]}kHz)"
    ax.set_title(title)

    # Create output file path
    ei = f"{ei}_" if ei else ""
    class_var = f"_{class_var}_" if class_var else ""

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


def plot_3d_scatter(diff_70_38, diff_120_38, diff_200_38, 
                    outfile: str,
                    figsize: tuple[int, int] = None,
                    diff_70_38_lim: tuple[float, float] = None,
                    diff_120_38_lim: tuple[float, float] = None,
                    diff_200_38_lim: tuple[float, float] = None):
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')

    ax.scatter(xs=diff_70_38, ys=diff_120_38, zs=diff_200_38,
               s=1,
               alpha=1e-2,
               c="black")

    ax.set_xlabel('ΔSv (70 - 38 kHz) [dB]')
    ax.set_ylabel('ΔSv (120 - 38 kHz) [dB]')
    ax.set_zlabel('ΔSv (200 - 38 kHz) [dB]')

    ax.set_box_aspect(None, zoom=0.85) # make sure the z-axis title is readable

    if diff_70_38_lim:
        ax.set_xlim(diff_70_38_lim[0], diff_70_38_lim[1])
    if diff_120_38_lim:
        ax.set_ylim(diff_120_38_lim[0], diff_120_38_lim[1])
    if diff_200_38_lim: 
        ax.set_zlim(diff_200_38_lim[0], diff_200_38_lim[1])

    plt.savefig(outfile, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_hexbin_2d(diff_70_38, diff_120_38,
                   outfile: str,
                   figsize: tuple[int, int] = None,
                   diff_70_38_lim: tuple[float, float] = None,
                   diff_120_38_lim: tuple[float, float] = None):

    fig, ax = plt.subplots(figsize=figsize)

    hb = ax.hexbin(x=diff_70_38,
                   y=diff_120_38,
                   gridsize=500,
                   bins="log",
                   cmap="inferno")

    ax.set_xlabel('ΔSv (70 - 38 kHz) [dB]')
    ax.set_ylabel('ΔSv (120 - 38 kHz) [dB]')

    if diff_70_38_lim:
        ax.set_xlim(diff_70_38_lim[0], diff_70_38_lim[1])
    if diff_120_38_lim:
        ax.set_ylim(diff_120_38_lim[0], diff_120_38_lim[1])

    cb = fig.colorbar(hb, ax=ax, label='counts')

    plt.savefig(outfile, bbox_inches='tight')
    plt.close()


def plot_hist(diff_70_38,
              outfile: str,
              figsize: tuple[int, int] = None,
              diff_70_38_lim: tuple[float, float] = None):

    fig, ax = plt.subplots(figsize=figsize)

    hb = ax.hist(x=diff_70_38,
                 bins=500,
                 color="black")

    ax.set_xlabel('ΔSv (70 - 38 kHz) [dB]')
    ax.set_ylabel('ΔSv (120 - 38 kHz) [dB]')

    if diff_70_38_lim:
        ax.set_xlim(diff_70_38_lim[0], diff_70_38_lim[1])

    plt.savefig(outfile, bbox_inches='tight')
    plt.close()


def plot_sv_dist_combined(img_0, img_1, img_2,
                          outfile: str,
                          figsize: tuple[int, int] = (18, 5)):
    # Create a row of 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    images = [img_0, img_1, img_2]
    labels = ['(a)', '(b)', '(c)']

    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img)
        ax.axis('off')  # Hide axes
        ax.set_title(label, loc='left', fontsize=14, pad=10)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def plot_3d_scatter_interactive(diff_70_38, diff_120_38, diff_200_38, 
                                diff_70_38_lim: tuple[float, float] = None,
                                diff_120_38_lim: tuple[float, float] = None,
                                diff_200_38_lim: tuple[float, float] = None):

    df = pd.DataFrame({
        'ΔSv (70 - 38 kHz) [dB]': diff_70_38,
        'ΔSv (120 - 38 kHz) [dB]': diff_120_38,
        'ΔSv (200 - 38 kHz) [dB]': diff_200_38
    })

    fig = px.scatter_3d(df, x='ΔSv (70 - 38 kHz) [dB]', y='ΔSv (120 - 38 kHz) [dB]', z='ΔSv (200 - 38 kHz) [dB]',
                        #size=1,
                        #opacity=1e-2,
                        #color="black",
                        range_x=diff_70_38_lim,
                        range_y=diff_120_38_lim,
                        range_z=diff_200_38_lim)
    
    fig.update_traces(marker=dict(
        size=1,
        opacity=0.1,
        color="black"
    ))

    fig.show()