
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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
    etopo = xr.open_dataset("../data/external/spatial/bathymetry/ETOPO1_Bed_g_gmt4.grd")
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
                             sample_pixels: int | None = 2_000_000):
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
    print(f"\nFigure saved as '{outfile}'")


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
    plt.close();


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
    plt.close();


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
    plt.close();


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