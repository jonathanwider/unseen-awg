import cartopy.crs as ccrs
import matplotlib as mpl
import numpy as np
import xarray as xr

from unseen_awg.utils import is_no_jump


def transition_init_time_plot(ax_joint, ax_top, ax_right, traj, only_jumps=False):
    # Data preparation (same as original)
    d_min = 0
    d_max = 366
    bins = np.arange(d_min - 0.5, d_max + 1.5)

    if traj.sizes.get("seed", 0):
        x = []
        y = []
        for s in traj.seed:
            c_traj = traj.sel(seed=s)
            if only_jumps:
                y.append(
                    (
                        c_traj.isel(out_time=slice(1, None))
                        .isel(out_time=~is_no_jump(c_traj).squeeze())
                        .init_time.dt.dayofyear.load()
                        .data
                    )
                )
                x.append(
                    (
                        c_traj.isel(out_time=slice(None, -1))
                        .isel(out_time=~is_no_jump(c_traj).squeeze())
                        .init_time.dt.dayofyear.load()
                        .data
                    )
                )
            else:
                y.append(
                    c_traj.isel(out_time=slice(1, None))
                    .init_time.dt.dayofyear.load()
                    .data
                )
                x.append(
                    c_traj.isel(out_time=slice(None, -1))
                    .init_time.dt.dayofyear.load()
                    .data
                )
        x = np.concatenate(x)
        y = np.concatenate(y)
    else:
        if only_jumps:
            y = (
                traj.isel(out_time=slice(1, None))
                .isel(out_time=~is_no_jump(traj).squeeze())
                .init_time.dt.dayofyear.load()
                .data
            )
            x = (
                traj.isel(out_time=slice(None, -1))
                .isel(out_time=~is_no_jump(traj).squeeze())
                .init_time.dt.dayofyear.load()
                .data
            )
        else:
            y = traj.isel(out_time=slice(1, None)).init_time.dt.dayofyear.load().data
            x = traj.isel(out_time=slice(None, -1)).init_time.dt.dayofyear.load().data

    # Create figure with gridspec for joint plot layout

    # Main scatter plot
    ax_joint.scatter(x, y, s=1, color="C0")
    ax_joint.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax_joint.set_xlabel(r"day of year $t_{init}$, step i")
    ax_joint.set_ylabel(r"day of year $t_{init}$, step i+1")
    ax_joint.set_xlim(d_min, d_max)
    ax_joint.set_ylim(d_min, d_max)
    ax_joint.set_aspect("equal")
    ax_joint.set_xticks(np.arange(d_min, d_max, 100))
    ax_joint.set_yticks(np.arange(d_min, d_max, 100))
    ax_joint.figure.canvas.draw()

    # Get the actual position of the joint plot after aspect ratio adjustment
    joint_pos = ax_joint.get_position()

    # Align top marginal with joint plot x-axis
    top_pos = ax_top.get_position()
    ax_top.set_position([joint_pos.x0, top_pos.y0, joint_pos.width, top_pos.height])

    # Align right marginal with joint plot y-axis
    right_pos = ax_right.get_position()
    ax_right.set_position(
        [right_pos.x0, joint_pos.y0, right_pos.width, joint_pos.height]
    )

    # Top marginal histogram (filled)
    ax_top.hist(x, bins=bins, color="C0", edgecolor="C0", histtype="stepfilled")
    ax_top.set_xlim(d_min, d_max)
    ax_top.tick_params(labelbottom=False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Right marginal histogram (filled)
    ax_right.hist(
        y,
        bins=bins,
        color="C0",
        edgecolor="C0",
        orientation="horizontal",
        histtype="stepfilled",
    )
    ax_right.set_ylim(d_min, d_max)
    ax_right.tick_params(labelleft=False)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    # Style adjustments to match seaborn
    for ax in [ax_joint, ax_top, ax_right]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")

    ax_right.sharey(ax_joint)
    ax_top.sharex(ax_joint)


def transition_lead_time_plot(
    ax_joint,
    ax_top,
    ax_right,
    traj,
    lt_max=44,
    only_jumps=False,
    use_log_cnorm_in_joint_plot=True,
):
    # c0_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    # Create custom colormap from transparent to C0
    # custom_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    #    "transparent_to_c0",
    #    [
    #        (0, 0, 0, 0),  # Transparent (RGBA)
    #        mpl.colors.to_rgba(c0_color),
    #    ],  # C0 color with full opacity
    # )
    # Data preparation (same as original)
    if traj.sizes.get("seed", 0):
        x = []
        y = []
        for s in traj.seed:
            c_traj = traj.sel(seed=s)
            if only_jumps:
                y.append(
                    (
                        c_traj.isel(out_time=slice(1, None))
                        .isel(out_time=~is_no_jump(c_traj).squeeze())
                        .lead_time.load()
                        .data
                        / np.timedelta64(1, "D")
                    ).astype("int")
                )
                x.append(
                    (
                        c_traj.isel(out_time=slice(None, -1))
                        .isel(out_time=~is_no_jump(c_traj).squeeze())
                        .lead_time.load()
                        .data
                        / np.timedelta64(1, "D")
                    ).astype("int")
                )
            else:
                y.append(
                    (
                        c_traj.isel(out_time=slice(1, None)).lead_time.load().data
                        / np.timedelta64(1, "D")
                    ).astype("int")
                )
                x.append(
                    (
                        c_traj.isel(out_time=slice(None, -1)).lead_time.load().data
                        / np.timedelta64(1, "D")
                    ).astype("int")
                )
        x = np.concatenate(x)
        y = np.concatenate(y)
    else:
        if only_jumps:
            y = (
                traj.isel(out_time=slice(1, None))
                .isel(out_time=~is_no_jump(traj).squeeze())
                .lead_time.load()
                .data
                / np.timedelta64(1, "D")
            ).astype("int")
            x = (
                traj.isel(out_time=slice(None, -1))
                .isel(out_time=~is_no_jump(traj).squeeze())
                .lead_time.load()
                .data
                / np.timedelta64(1, "D")
            ).astype("int")
        else:
            y = (
                traj.isel(out_time=slice(1, None)).lead_time.load().data
                / np.timedelta64(1, "D")
            ).astype("int")
            x = (
                traj.isel(out_time=slice(None, -1)).lead_time.load().data
                / np.timedelta64(1, "D")
            ).astype("int")

    bins = np.arange(-0.5, lt_max + 1.5, 1)

    # Create figure with gridspec for joint plot layout

    # Main 2D histogram
    norm = mpl.colors.LogNorm() if use_log_cnorm_in_joint_plot else None
    counts, xedges, yedges, im = ax_joint.hist2d(
        x, y, bins=[bins, bins], cmap="Blues", alpha=0.8, norm=norm
    )
    # ax_joint.grid(None)

    # Add grid lines at each bin edge
    for edge in bins:
        ax_joint.axvline(
            edge, color="k", linestyle="-", linewidth=0.5, alpha=0.5, zorder=10
        )
        ax_joint.axhline(
            edge, color="k", linestyle="-", linewidth=0.5, alpha=0.5, zorder=10
        )

    ax_joint.set_xlabel(r"$t_{lead}$, step i")
    ax_joint.set_ylabel(r"$t_{lead}$, step i+1")
    ax_joint.set_xlim(-0.5, lt_max + 0.5)
    ax_joint.set_ylim(-0.5, lt_max + 0.5)
    ax_joint.set_aspect("equal")
    ax_joint.grid(None)

    ax_joint.figure.canvas.draw()

    # Get the actual position of the joint plot after aspect ratio adjustment
    joint_pos = ax_joint.get_position()

    # Align top marginal with joint plot x-axis
    top_pos = ax_top.get_position()
    ax_top.set_position([joint_pos.x0, top_pos.y0, joint_pos.width, top_pos.height])

    # Align right marginal with joint plot y-axis
    right_pos = ax_right.get_position()
    ax_right.set_position(
        [right_pos.x0, joint_pos.y0, right_pos.width, joint_pos.height]
    )

    # Top marginal histogram (stepfilled style)
    ax_top.hist(x, bins=bins, histtype="stepfilled", alpha=0.7, color="C0")

    # Add vertical grid lines to top marginal
    for edge in bins:
        ax_top.axvline(
            edge, color="k", linestyle="-", linewidth=0.5, alpha=0.5, zorder=10
        )

    ax_top.set_xlim(-0.5, lt_max + 0.5)
    ax_top.tick_params(labelbottom=False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.set_xticks([])
    ax_top.grid(None)

    # Right marginal histogram (stepfilled style)
    ax_right.hist(
        y,
        bins=bins,
        histtype="stepfilled",
        alpha=0.7,
        color="C0",
        orientation="horizontal",
    )

    # Add horizontal grid lines to right marginal
    for edge in bins:
        ax_right.axhline(
            edge, color="k", linestyle="-", linewidth=0.5, alpha=0.5, zorder=10
        )

    ax_right.set_ylim(-0.5, lt_max + 0.5)
    ax_right.tick_params(labelleft=False)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.set_yticks([])
    ax_right.grid(None)
    # Style adjustments
    for ax in [ax_joint, ax_top, ax_right]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")


def transition_valid_time_plot(ax_joint, ax_top, ax_right, traj, only_jumps=False):
    # Data preparation (same as original)
    d_min = 0
    d_max = 366
    bins = np.arange(d_min - 0.5, d_max + 1.5)

    if traj.sizes.get("seed", 0):
        x = []
        y = []
        for s in traj.seed:
            c_traj = traj.sel(seed=s)
        if only_jumps:
            y.append(
                (
                    c_traj.isel(out_time=slice(1, None))
                    .isel(out_time=~is_no_jump(c_traj).squeeze())
                    .init_time
                    + c_traj.isel(out_time=slice(1, None))
                    .isel(out_time=~is_no_jump(c_traj).squeeze())
                    .lead_time
                ).dt.dayofyear.load()
            )
            x.append(
                (
                    c_traj.isel(out_time=slice(None, -1))
                    .isel(out_time=~is_no_jump(c_traj).squeeze())
                    .init_time
                    + c_traj.isel(out_time=slice(None, -1))
                    .isel(out_time=~is_no_jump(c_traj).squeeze())
                    .lead_time
                ).dt.dayofyear.load()
            )
        else:
            y.append(
                (
                    c_traj.isel(out_time=slice(1, None)).init_time
                    + c_traj.isel(out_time=slice(1, None)).lead_time
                ).dt.dayofyear.load()
            )
            x.append(
                (
                    c_traj.isel(out_time=slice(None, -1)).init_time
                    + c_traj.isel(out_time=slice(None, -1)).lead_time
                ).dt.dayofyear.load()
            )
        x = np.concatenate(x)
        y = np.concatenate(y)
    else:
        if only_jumps:
            y = (
                traj.isel(out_time=slice(1, None))
                .isel(out_time=~is_no_jump(traj).squeeze())
                .init_time.dt.dayofyear.load()
                .data
            )
            x = (
                traj.isel(out_time=slice(None, -1))
                .isel(out_time=~is_no_jump(traj).squeeze())
                .init_time.dt.dayofyear.load()
                .data
            )
        else:
            y = traj.isel(out_time=slice(1, None)).init_time.dt.dayofyear.load().data
            x = traj.isel(out_time=slice(None, -1)).init_time.dt.dayofyear.load().data

    # Create figure with gridspec for joint plot layout

    # Main scatter plot
    ax_joint.scatter(x, y, s=1, color="C0")
    ax_joint.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax_joint.set_xlabel(r"day of year $t_{valid}$, step i")
    ax_joint.set_ylabel(r"day of year $t_{valid}$, step i+1")
    ax_joint.set_xlim(d_min, d_max)
    ax_joint.set_ylim(d_min, d_max)
    ax_joint.set_aspect("equal")
    ax_joint.figure.canvas.draw()
    ax_joint.set_xticks(np.arange(d_min, d_max, 100))
    ax_joint.set_yticks(np.arange(d_min, d_max, 100))

    # Get the actual position of the joint plot after aspect ratio adjustment
    joint_pos = ax_joint.get_position()

    # Align top marginal with joint plot x-axis
    top_pos = ax_top.get_position()
    ax_top.set_position([joint_pos.x0, top_pos.y0, joint_pos.width, top_pos.height])

    # Align right marginal with joint plot y-axis
    right_pos = ax_right.get_position()
    ax_right.set_position(
        [right_pos.x0, joint_pos.y0, right_pos.width, joint_pos.height]
    )

    # Top marginal histogram (filled)
    ax_top.hist(x, bins=bins, color="C0", edgecolor="C0", histtype="stepfilled")
    ax_top.set_xlim(d_min, d_max)
    ax_top.tick_params(labelbottom=False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Right marginal histogram (filled)
    ax_right.hist(
        y,
        bins=bins,
        color="C0",
        edgecolor="C0",
        orientation="horizontal",
        histtype="stepfilled",
    )
    ax_right.set_ylim(d_min, d_max)
    ax_right.tick_params(labelleft=False)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    # Style adjustments to match seaborn
    for ax in [ax_joint, ax_top, ax_right]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")


# https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
def add_headers(
    fig,
    row_headers=None,
    col_headers=None,
    cbar_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs,
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()
    if cbar_headers is not None:
        axes = [a for a in axes if a.get_ylabel() not in cbar_headers]
    else:
        axes = list(axes)
    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )


def map_plot_without_frame_with_bounds(ax, da: xr.DataArray, **plot_kwargs):
    dlat = abs(da.latitude[1] - da.latitude[0])
    dlon = abs(da.longitude[1] - da.longitude[0])

    lat_max = da.latitude.max() + dlat / 2
    lat_min = da.latitude.min() - dlat / 2
    lon_max = da.longitude.max() + dlon / 2
    lon_min = da.longitude.min() - dlon / 2

    # black frame around region with data
    ax.plot(
        [lon_min, lon_max, lon_max, lon_min, lon_min],
        [lat_min, lat_min, lat_max, lat_max, lat_min],
        color="black",
        linewidth=1,
        transform=ccrs.PlateCarree(),  # remove this line to get straight lines
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    m = da.plot(ax=ax, transform=ccrs.PlateCarree(), **plot_kwargs)

    ax.coastlines(resolution="50m", linewidth=1, color="k")

    return m


def contourf_plot_without_frame_with_bounds(ax, da: xr.DataArray, **plot_kwargs):
    # dlat = abs(da.latitude[1] - da.latitude[0])
    # dlon = abs(da.longitude[1] - da.longitude[0])

    lat_max = da.latitude.max()
    lat_min = da.latitude.min()
    lon_max = da.longitude.max()
    lon_min = da.longitude.min()

    # black frame around region with data
    ax.plot(
        [lon_min, lon_max, lon_max, lon_min, lon_min],
        [lat_min, lat_min, lat_max, lat_max, lat_min],
        color="black",
        transform=ccrs.PlateCarree(),  # remove this line to get straight lines
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    da.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), **plot_kwargs)

    ax.coastlines()


def add_contours(
    ax,
    da: xr.DataArray,
    major_levels,
    minor_levels,
    use_contour_labels=True,
    linewidth_major=1,
    linewidth_minor=0.5,
    **plot_kwargs,
):
    da.squeeze().plot.contour(
        ax=ax,
        levels=minor_levels,
        linewidths=linewidth_minor,
        transform=ccrs.PlateCarree(),
        colors="k",
        **plot_kwargs,
    )
    contours = da.squeeze().plot.contour(
        ax=ax,
        levels=major_levels,
        linewidths=linewidth_major,
        transform=ccrs.PlateCarree(),
        colors="k",
        **plot_kwargs,
    )
    if use_contour_labels:
        ax.clabel(contours, contours.levels)


def add_label_to_axes(
    ax,
    label,
    ax_xpos=0.05,
    ax_ypos=0.95,
    ha="left",
    va="top",
    edgecolor="white",
    **font_kwargs,
):
    ax.text(
        ax_xpos,
        ax_ypos,
        label,
        ha=ha,
        va=va,
        transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor=edgecolor, boxstyle="round4"),
        **font_kwargs,
    )
