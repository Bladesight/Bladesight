import pandas as pd
import os
import pathlib
import numpy as np
import plotly.express as px
from numba import njit
from tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from numba import njit
import numpy as np

import plotly.io as pio

pio.templates.default = "ggplot2"

THIS_FILE_DIRECTORY = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))

def calculate_shaft_speed(df_opr : pd.DataFrame) -> pd.DataFrame:
    """ This function calculates the shaft speed from the zero-crossing times.

    Args:
        df_opr (pd.DataFrame): A DataFrame containing a column names
            `opr_zero_crossing_times` where each value is a zero-crossing time
            of an OPR signal.

    Returns:
        pd.DataFrame: A DataFrame with two columns `omega_n` and `n` where
            `omega_n` is the shaft speed in rad/s and `n` is the shaft speed
            in RPM.
    """
    # Calculate the shaft speed
    df_opr = df_opr.copy()
    df_opr["omega_n"] = (
        2 * np.pi / df_opr["opr_zero_crossing_times"].diff()
    )
    df_opr = df_opr.dropna()
    df_opr["n"] = np.arange(len(df_opr))
    return df_opr

def create_shaft_speed_plot(df_opr):
    df_shaft_speed = calculate_shaft_speed(df_opr)
    df_shaft_speed["Shaft speed"] = df_shaft_speed["omega_n"] / (2*np.pi) * 60
    # Plot the shaft speed
    fig = px.line(
        df_shaft_speed, 
        x="opr_zero_crossing_times", 
        y="Shaft speed"
    )
    fig.update_layout(
        title="Shaft Speed",
        xaxis_title="Time (s)",
        yaxis_title="Shaft Speed (RPM)",
    )

    fig.write_html(THIS_FILE_DIRECTORY / "shaft_speed.html")

@njit
def calculate_aoas(arr_opr_zero_crossing, arr_probe_toas):
    """
    This function calculates the angle of arrival of each ToA value relative 
    to the revolution in which it occurs.

    This function is JIT compiled, meaning that it is 
    compiled to machine code.

    Args:
        arr_opr_zero_crossing (np.array): An array of OPR zero-crossing times. 
        arr_probe_toas (np.array): An array of ToA values.
    
    Returns:
        np.array: A matrix of AoA values. Each row in the matrix corresponds to
            a ToA value. The columns are:
                0: The revolution number
                1: The zero crossing time at the start of the revolution
                2: The zero crossing time at the end of the revolution
                3: The angular velocity of the revolution
                4: The AoA of the ToA value
    """
    num_toas = len(arr_probe_toas)
    AoA_matrix = np.zeros( (num_toas, 5) )

    AoA_matrix[:, 0] = -1

    current_zero_crossing_start = arr_opr_zero_crossing[0]
    current_zero_crossing_end = arr_opr_zero_crossing[1]
    current_n = 0

    for i, toa in enumerate(arr_probe_toas):
        
        while toa > current_zero_crossing_end:
            current_n += 1
            if current_n >= (len(arr_opr_zero_crossing) - 1):
                break
            current_zero_crossing_start = arr_opr_zero_crossing[current_n]
            current_zero_crossing_end = arr_opr_zero_crossing[current_n + 1]
        
        if current_n >= (len(arr_opr_zero_crossing) - 1):
            break
        
        if toa > current_zero_crossing_start:
            AoA_matrix[i, 0] = current_n
            AoA_matrix[i, 1] = current_zero_crossing_start
            AoA_matrix[i, 2] = current_zero_crossing_end
            omega = 2 * np.pi / (current_zero_crossing_end - current_zero_crossing_start)
            AoA_matrix[i, 3] = omega
            AoA_matrix[i, 4] = omega * (toa - current_zero_crossing_start)
    return AoA_matrix

THIS_FILE_DIRECTORY = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PATH_TO_DATA = THIS_FILE_DIRECTORY.parent / "example_data"

PATH_TO_DATA = THIS_FILE_DIRECTORY.parent / "example_data"
df_opr = pd.read_csv(PATH_TO_DATA / "opr_zero_crossing_times.csv" )
df_toas = pd.read_csv(PATH_TO_DATA / "probe_1_toas.csv" )

AoA_matrix = calculate_aoas(
    df_opr["opr_zero_crossing_times"].values,
    df_toas["ToA"].values
)

df_AoAs = pd.DataFrame(
    AoA_matrix,
    columns=["n", "zero_crossing_start", "zero_crossing_end", "omega", "AoA"]
)

df_AoAs["omega_rpm"] = df_AoAs["omega"] / (2*np.pi) * 60
df_AoAs["AoA_deg"] = df_AoAs["AoA"] / (2*np.pi) * 360

## FIGURE 1
# Create a simple scatter plot of the AoA values
fig = px.scatter(
    df_AoAs,
    x="zero_crossing_start",
    y="AoA_deg",
    color="AoA_deg",
    color_continuous_scale="viridis",
    range_color=[0, 360],
    hover_data=["n", "omega_rpm"],
    render_mode="webgl"
)
fig.update_layout(
    title="AoA values",
    xaxis_title="Time (s)",
    yaxis_title="AoA (deg)",
)
fig.write_html(THIS_FILE_DIRECTORY / "aoa_scatter.html")

## Figure 2: AoA histogram with bins
bin_edges = np.linspace(0, 400, 500)
bin_counts, bin_edges = np.histogram(df_AoAs["AoA_deg"], bins=bin_edges)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

max_bin_count = np.max(bin_counts)

fig = go.Figure()

# Create a black barplot
fig = fig.add_trace(
    go.Barpolar(
        r=bin_counts,
        base=np.ones_like(bin_counts) * max_bin_count,
        theta=bin_centers,
        dtheta=360 / (len(bin_centers)*2),
        width=360 / len(bin_centers),
        marker_color="black",
        marker_line_color="black",
        showlegend=True,
        name="Blade count vs AoA",
        hovertemplate="AoA: %{theta:.2f} deg<br>Count: %{r}<extra></extra>"
    )
)

# Draw a black circle around the plot at the maximum bin count
fig = fig.add_trace(
    go.Scatterpolar(
        r=np.ones_like(bin_centers) * max_bin_count,
        theta=bin_centers,
        mode="lines",
        line_color="black",
        showlegend=False,
        hoverinfo="skip"
    )
)
if False:
    B = 5
    align_bin_edges = np.linspace(0, 360, B + 1)
    align_bin_base = np.ones_like(align_bin_edges) * max_bin_count
    align_bin_theta = align_bin_edges[:-1]
    align_bin_dtheta = 360 / (len(align_bin_theta)*2)
    align_bin_width = 360 / len(align_bin_theta) - 5

    # Make opaque red bars for the alignment bins. 
    # Ensure the colorbar has a borderwidth of 5 so that it is visible

    fig = fig.add_trace(
        go.Barpolar(
            base = align_bin_base,
            r=np.ones_like(align_bin_theta) * max_bin_count * 0.3,
            theta=align_bin_theta+(360/B)/2,
            width=align_bin_width,
            marker_color="red",
            opacity=0.5,
            showlegend=True,
            name="Alignment bins",
            hoverinfo="skip"
        )
    )

    # Add thin green bars at the bin centers
    fig = fig.add_trace(
        go.Barpolar(
            base = align_bin_base,
            r=np.ones_like(align_bin_theta) * max_bin_count * 0.5,
            theta=align_bin_theta+(360/B)/2,
            width=2,
            marker_color="green",
            showlegend=True,
            name="Bin centers",
            hovertemplate="Alignment bin center: %{theta:.2f} deg<extra></extra>"
        )
    )

# Hide the r axis and the Grid
fig.update_layout(
    title="AoA Histogram",
    polar=dict(
        radialaxis=dict(
            visible=False
            )
        ),
    showlegend=True
)

fig.write_html(THIS_FILE_DIRECTORY / "aoa_histogram_intro.html", auto_open=False)

## FIGURE 3: Problematic AoA values
df_AoAs["AoA_deg"] = df_AoAs["AoA"] / (2*np.pi) * 360 + 25.47

bin_edges = np.linspace(0, 400, 500)
bin_counts, bin_edges = np.histogram(df_AoAs["AoA_deg"], bins=bin_edges)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

max_bin_count = np.max(bin_counts)

#print(bin_counts + max_bin_count)
#print(max_bin_count)
#print(bin_centers)

fig = go.Figure()

# Create a black barplot
fig = fig.add_trace(
    go.Barpolar(
        r=bin_counts,
        base=np.ones_like(bin_counts) * max_bin_count,
        theta=bin_centers,
        dtheta=360 / (len(bin_centers)*2),
        width=360 / len(bin_centers),
        marker_color="black",
        marker_line_color="black",
        showlegend=True,
        name="Blade count vs AoA",
        hovertemplate="AoA: %{theta:.2f} deg<br>Count: %{r}<extra></extra>"
    )
)

# Draw a black circle around the plot at the maximum bin count
fig = fig.add_trace(
    go.Scatterpolar(
        r=np.ones_like(bin_centers) * max_bin_count,
        theta=bin_centers,
        mode="lines",
        line_color="black",
        showlegend=False,
        hoverinfo="skip"
    )
)
B = 5
align_bin_edges = np.linspace(0, 360, B + 1)
align_bin_base = np.ones_like(align_bin_edges) * max_bin_count
align_bin_theta = align_bin_edges[:-1]
align_bin_dtheta = 360 / (len(align_bin_theta)*2)
align_bin_width = 360 / len(align_bin_theta) - 5

# Make opaque red bars for the alignment bins. 
# Ensure the colorbar has a borderwidth of 5 so that it is visible

fig = fig.add_trace(
    go.Barpolar(
        base = align_bin_base,
        r=np.ones_like(align_bin_theta) * max_bin_count * 0.3,
        theta=align_bin_theta+(360/B)/2,
        width=align_bin_width,
        marker_color="red",
        opacity=0.5,
        showlegend=True,
        name="Alignment bins",
        hoverinfo="skip"
    )
)

# Add thin green bars at the bin centers
fig = fig.add_trace(
    go.Barpolar(
        base = align_bin_base,
        r=np.ones_like(align_bin_theta) * max_bin_count * 0.5,
        theta=align_bin_theta+(360/B)/2,
        width=2,
        marker_color="green",
        showlegend=True,
        name="Bin centers",
        hovertemplate="Alignment bin center: %{theta:.2f} deg<extra></extra>"
    )
)

# Hide the r axis and the Grid
fig.update_layout(
    title="AoA Histogram",
    polar=dict(
        radialaxis=dict(
            visible=False
            )
        ),
    showlegend=True
)


fig.write_html(THIS_FILE_DIRECTORY / "aoa_histogram_problematic.html", auto_open=False)

## FIGURE 4: BETTER ALIGNMENT
df_AoAs["AoA_deg"] = df_AoAs["AoA"] / (2*np.pi) * 360 + 25.47

bin_edges = np.linspace(0, 400, 500)
bin_counts, bin_edges = np.histogram(df_AoAs["AoA_deg"], bins=bin_edges)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

max_bin_count = np.max(bin_counts)

fig = go.Figure()

# Create a black barplot
fig = fig.add_trace(
    go.Barpolar(
        r=bin_counts,
        base=np.ones_like(bin_counts) * max_bin_count,
        theta=bin_centers,
        dtheta=360 / (len(bin_centers)*2),
        width=360 / len(bin_centers),
        marker_color="black",
        marker_line_color="black",
        showlegend=True,
        name="Blade count vs AoA",
        hovertemplate="AoA: %{theta:.2f} deg<br>Count: %{r}<extra></extra>"
    )
)

# Draw a black circle around the plot at the maximum bin count
fig = fig.add_trace(
    go.Scatterpolar(
        r=np.ones_like(bin_centers) * max_bin_count,
        theta=bin_centers,
        mode="lines",
        line_color="black",
        showlegend=False,
        hoverinfo="skip"
    )
)
B = 5
align_bin_edges = np.linspace(0+30, 360+30, B + 1)
align_bin_base = np.ones_like(align_bin_edges) * max_bin_count
align_bin_theta = align_bin_edges[:-1]
align_bin_dtheta = 360 / (len(align_bin_theta)*2)
align_bin_width = 360 / len(align_bin_theta) - 5

# Make opaque red bars for the alignment bins. 
# Ensure the colorbar has a borderwidth of 5 so that it is visible

fig = fig.add_trace(
    go.Barpolar(
        base = align_bin_base,
        r=np.ones_like(align_bin_theta) * max_bin_count * 0.3,
        theta=align_bin_theta+(360/B)/2,
        width=align_bin_width,
        marker_color="red",
        opacity=0.5,
        showlegend=True,
        name="Alignment bins",
        hoverinfo="skip"
    )
)

# Add thin green bars at the bin centers
fig = fig.add_trace(
    go.Barpolar(
        base = align_bin_base,
        r=np.ones_like(align_bin_theta) * max_bin_count * 0.5,
        theta=align_bin_theta+(360/B)/2,
        width=2,
        marker_color="green",
        showlegend=True,
        name="Bin centers",
        hovertemplate="Alignment bin center: %{theta:.2f} deg<extra></extra>"
    )
)

# Hide the r axis and the Grid
fig.update_layout(
    title="AoA Histogram",
    polar=dict(
        radialaxis=dict(
            visible=False
            )
        ),
    showlegend=True
)


fig.write_html(THIS_FILE_DIRECTORY / "aoa_histogram_better.html", auto_open=False)

df_AoAs["AoA_deg"] = df_AoAs["AoA"] / (2*np.pi) * 360


def calculate_Q(
	df_AoAs : pd.DataFrame, # (1)!
	d_theta : float,  # (2)!
	N : int  # (3)!
) -> float:	
    # Create a nwe column that is offset an entire
    # revolution from the initial values
    df_AoAs["AoA_deg_offset"] = df_AoAs["AoA_deg"] + 360

    # Calculate the alignment bin edges
    bin_edges = np.linspace(0+d_theta, 360 + d_theta, N + 1)
    
    # Initialize the quality factor
    Q = 0
    for b in range(5):
        left_edge = bin_edges[b]
        right_edge = bin_edges[b + 1]
        bin_centre = (left_edge + right_edge)/2 # noqa
        ix_blades_in_bin_ref = df_AoAs["AoA_deg"].between(left_edge, right_edge) # noqa
        ix_blades_in_bin_offset = df_AoAs["AoA_deg_offset"].between(left_edge, right_edge) # noqa
        Q_ref = np.sum(
            (
                df_AoAs.loc[ix_blades_in_bin_ref, "AoA_deg"] 
                - bin_centre
            )**2
        )
        Q_offset = np.sum(
            (
                df_AoAs.loc[ix_blades_in_bin_offset, "AoA_deg_offset"] 
                - bin_centre
            )**2
        ) # noqa
        Q += Q_ref + Q_offset
    return Q

d_thetas = np.linspace(0, 360/B, 150)
Qs = []
for d_theta in d_thetas:
    Qs.append(calculate_Q(df_AoAs, d_theta, 5))
ix_optimal = np.argmin(np.array(Qs))
d_theta_optimal = d_thetas[ix_optimal]
## FIGURE 5: Q factor

fig = go.Figure()
# Add a scatterplot of only the 10th item as a big green opaque dot called "Optimal offset"
fig = fig.add_trace(
    go.Scatter(
        x=[d_thetas[ix_optimal]],
        y=[Qs[ix_optimal]],
        mode="markers",
        line_color="green",
        showlegend=True,
        name = "Optimal offset",
        marker=dict(
            color='Green',
            size=20,
        ),
        hovertemplate="d_theta: %{x:.2f} deg<br>Q: %{y:.2f}<extra></extra>",
        opacity=0.5
    )
)

fig = fig.add_trace(
    go.Scatter(
        x=d_thetas,
        y=Qs,
        mode="lines+markers",
        line_color="black",
        showlegend=False,
        hovertemplate="d_theta: %{x:.2f} deg<br>Q: %{y:.2f}<extra></extra>"
    )
)

fig.update_layout(
    title="Quality factor vs alignment bin offset",
    xaxis_title="Alignment bin offset (deg)",
    yaxis_title="Quality factor",
    showlegend=True
)

fig.write_html(THIS_FILE_DIRECTORY / "quality_factor.html", auto_open=False)


# FIGURE 6: Optimal offset histogram
bin_edges = np.linspace(0, 400, 500)
bin_counts, bin_edges = np.histogram(df_AoAs["AoA_deg"], bins=bin_edges)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

max_bin_count = np.max(bin_counts)

fig = go.Figure()

# Create a black barplot
fig = fig.add_trace(
    go.Barpolar(
        r=bin_counts,
        base=np.ones_like(bin_counts) * max_bin_count,
        theta=bin_centers,
        dtheta=360 / (len(bin_centers)*2),
        width=360 / len(bin_centers),
        marker_color="black",
        marker_line_color="black",
        showlegend=True,
        name="Blade count vs AoA",
        hovertemplate="AoA: %{theta:.2f} deg<br>Count: %{r}<extra></extra>"
    )
)

# Draw a black circle around the plot at the maximum bin count
fig = fig.add_trace(
    go.Scatterpolar(
        r=np.ones_like(bin_centers) * max_bin_count,
        theta=bin_centers,
        mode="lines",
        line_color="black",
        showlegend=False,
        hoverinfo="skip"
    )
)
B = 5
align_bin_edges = np.linspace(0 + d_theta_optimal, 360 + d_theta_optimal, B + 1)
align_bin_base = np.ones_like(align_bin_edges) * max_bin_count
align_bin_theta = align_bin_edges[:-1]
align_bin_dtheta = 360 / (len(align_bin_theta)*2)
align_bin_width = 360 / len(align_bin_theta) - 5

# Make opaque red bars for the alignment bins. 
# Ensure the colorbar has a borderwidth of 5 so that it is visible

fig = fig.add_trace(
    go.Barpolar(
        base = align_bin_base,
        r=np.ones_like(align_bin_theta) * max_bin_count * 0.3,
        theta=align_bin_theta+(360/B)/2,
        width=align_bin_width,
        marker_color="red",
        opacity=0.5,
        showlegend=True,
        name="Alignment bins",
        hoverinfo="skip"
    )
)
bin_centres = (align_bin_edges[:-1] + align_bin_edges[1:]) / 2
# Add thin green bars at the bin centers
fig = fig.add_trace(
    go.Barpolar(
        base = align_bin_base,
        r=np.ones_like(align_bin_theta) * max_bin_count * 0.5,
        theta=bin_centres,
        width=2,
        marker_color="green",
        showlegend=True,
        name="Bin centers",
        hovertemplate="Alignment bin center: %{theta:.2f} deg<extra></extra>"
    )
)

# Hide the r axis and the Grid
fig.update_layout(
    title="AoA Histogram",
    polar=dict(
        radialaxis=dict(
            visible=False
            )
        ),
    showlegend=True
)


fig.write_html(THIS_FILE_DIRECTORY / "aoa_histogram_optimal.html", auto_open=False)

from IPython import embed; embed()