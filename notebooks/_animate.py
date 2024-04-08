import pandas as pd
import numpy as np

import hvplot
import hvplot.pandas  # noqa
import holoviews as hv
import holoviews.operation.datashader as hd
import datashader as ds
import datashader.transfer_functions as tf
from datashader.bundling import directly_connect_edges
from holoviews.selection import link_selections
import matplotlib.pyplot as plt

from tqdm import tqdm
from subprocess import PIPE, run


def build_webdriver():
    """Start webdriver manually

    Selenium does not find geckodriver even though it is in (conda) path.
    We start one here and assign it to panel's io state so holoviews
    detects it when it constructs a renderer. This is a hack to make the
    bokeh renderer work in holoviews...
    """
    import selenium
    from shutil import which
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service
    from selenium.webdriver.firefox.webdriver import WebDriver as Firefox
    from panel.io.state import state

    service = Service(executable_path=which("geckodriver"))
    options = Options()
    options.add_argument("--headless")
    options.set_preference("layout.css.devPixelsPerPx", f"{1}")

    state.webdriver = Firefox(service=service, options=options)
    return state.webdriver


def build_background(df, suffix="base", width=600, height=400):
    """Draw static background image."""
    extent = (
        df[f"x_{suffix}"].min(),
        df[f"x_{suffix}"].max(),
        df[f"y_{suffix}"].min(),
        df[f"y_{suffix}"].max(),
    )
    return hv.RGB(
        hd.shade.uint32_to_uint8_xr(
            tf.set_background(
                tf.shade(
                    tf.spread(
                        ds.Canvas(
                            plot_width=width,
                            plot_height=height,
                            x_range=(extent[0], extent[1]),
                            y_range=(extent[2], extent[3]),
                        ).points(df, f"x_{suffix}", f"y_{suffix}", agg=ds.count()),
                        px=0,
                        how="add",
                    ),
                    cmap=plt.get_cmap("gray_r"),
                ),
                "white",
            )
        ),
        kdims=[f"x_{suffix}", f"y_{suffix}"],
        vdims=hv.RGB.vdims[:],
    ).opts(
        width=width, 
        height=height, 
        padding=0.05,
        xaxis=None,
        yaxis=None
    )


def build_static_cycle(width=400, height=400):
    """Draw simple year-clock to indicate time."""
    paths = []
    for angle in np.linspace(0, 2 * np.pi, 13)[:-1]:
        paths += [
            (0.95 * np.sin(angle), 0.95 * np.cos(angle)),
            (1.05 * np.sin(angle), 1.05 * np.cos(angle)),
            (np.nan, np.nan),
        ]
    ticks = hv.Curve(paths).opts(line_width=1, color="black")
    circle = hv.Ellipse(0, 0, 2)

    return (circle * ticks).opts(
        width=400,
        height=400,
        xaxis=None,
        yaxis=None,
    )


def build_cycle_hand(loc_df, idx, width=400, height=400):
    """Draw simple year-clock to indicate time."""
    day_of_year = pd.Timestamp(loc_df.date.iloc[idx]).dayofyear
    angle = day_of_year / 366 * 2 * np.pi
    return hv.Curve([(0, 0), (np.sin(angle), np.cos(angle))]).opts(
        width=400,
        height=400,
        title=f"Date in {loc_df.year.iloc[idx]}",
        xaxis=None,
        yaxis=None,
    )


def build_tail(tail_df, points_df, suffix="base", width=600, height=400):
    """Draw tail overlay."""
    edge_df = pd.DataFrame(
        {
            "source": tail_df.index[:-1],
            "target": tail_df.index[1:],
            "tail": np.arange(len(tail_df) - 1, 0, -1),
        }
    )
    paths = directly_connect_edges(points_df, edge_df, weight="tail")
    return paths.rename(columns={"x": f"x_{suffix}", "y": f"y_{suffix}"}).hvplot.paths(
        f"x_{suffix}",
        f"y_{suffix}",
        cmap="Plasma_r",
        color="tail",
        clim=(1, 4),
        line_width=2,
        width=width,
        height=height,
        padding=0.05,
        xaxis=None,
        yaxis=None
    )


def frame_range(num_frames, parts, part_idx):
    batch_size = num_frames // parts
    start_idx = part_idx * batch_size
    if part_idx == parts - 1:
        end_idx = num_frames
    else:
        end_idx = (part_idx + 1) * batch_size
    return start_idx, end_idx


def render_tails(
    df, latitude, longitude, suffix="base", n_tails=5, parts=6, part_idx=0
):
    """Builds an animation showing how a location moves through the manifold over time."""
    # Extract data for one location
    loc_df = df.query(f"latitude=={latitude} & longitude=={longitude}")
    points_df = loc_df[[f"x_{suffix}", f"y_{suffix}"]].rename(
        columns={f"x_{suffix}": "x", f"y_{suffix}": "y"}
    )
    loc_df = loc_df[["date", "year", "days_since_start"]]
    loc_df = loc_df.sort_values("days_since_start")

    # Construct one frames
    driver = build_webdriver()
    background = build_background(df, suffix=suffix)
    clock = build_static_cycle()
    for i in tqdm(range(*frame_range(loc_df.shape[0], parts, part_idx))):
        date = loc_df.days_since_start.iloc[i]
        tail_df = loc_df.query(
            f"days_since_start > {date - n_tails} & days_since_start <= {date}"
        )
        cycle = build_cycle_hand(loc_df, i)
        if tail_df.shape[0] == 1:
            frame = background + clock * cycle
        else:
            tails = build_tail(tail_df, points_df, suffix=suffix)
            frame = background * tails + clock * cycle
        frame_name = f"./figures/frames/{i}_{suffix}_{latitude}_{longitude}.png"
        hv.save(frame, frame_name, fmt="png")
    driver.close()


def animate_frames(latitude, longitude, suffix="base", fps=10):
    frame_names = f"./figures/frames/%d_{suffix}_{latitude}_{longitude}.png"
    output_name = f"./figures/tails_{suffix}_{latitude}_{longitude}.mp4"
    
    # Generate mp4
    print("generate mp4")
    cuda_opts = "-hwaccel cuda -hwaccel_output_format cuda"
    command = f"ffmpeg -y -r {fps} {cuda_opts} -pattern_type sequence -i {frame_names} -vcodec av1_nvenc {output_name}"
    result = run(
        command.split(" "),
        stdout=PIPE,
        stderr=PIPE,
        universal_newlines=True,
        shell=True,
    )
    print("done")
