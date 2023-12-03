---
date: 2023-10-08
tags:
  - Blade Tip Timing
  - BTT
  - Non Intrusive Stress Measurement
  - NSMS
  - Time of Arrival
  - Turbine blade
  - Mechanical Vibration
hide:
  - tags
description: This chapter explains how to allocate the AoA values to specific blades.
robots: index, follow, Blade Tip Timing, BTT, Non Intrusive Stress Measurement, NSMS, Time of Arrival, Turbine blade,Mechanical Vibration
template: main_intro_to_btt.html
card_title: Intro to BTT Ch4 - Identifying the blades
card_url: "ch4/"
---
# Identifying the blades
In the previous section, we have converted the ToAs into AoAs. We also displayed the AoAs of the first blade. How did we know this was the first blade?

Our example was simple enough that selecting the AoAs of the first blade was obvious. Our blades arrive on time, every time, like a bus schedule that's never wrong. We could therefore select every 5th value from our AoA DataFrame. Easy-peasy.

In the real world this is rarely the case. Noise may cause some ToAs to be triggered twice, or never. Also, if a blade arrives at or near the same time as your OPR zero-crossing times, it may lead to the blade arriving either at the start of the current revolution, or at the end of the previous revolution, depending on the vibrational state of the blade.

A better policy for blade identification is to use an algorithm. 

## AoA histogram

We'll start by constructing a histogram of the AoAs from a single proximity probe. You can use the `np.histogram` method in numpy to do this. Below, we show how to import the data, calculate the AoAs, and calculate the histogram.

!!! tip "Bladesight functions"
	In the previous chapter, we created algorithms for transforming the ToAs into AoAs. These functions have been added to the `bladesight` package, such that it can be imported. 

	All the algorithms built up during this tutorial will be added to the `bladesight` package.

``` py linenums="1"
from bladesight import Datasets
from bladesight.btt.aoa import transform_ToAs_to_AoAs

dataset = Datasets['data/intro_to_btt/intro_to_btt_ch03']
df_opr_zero_crossings = \
    dataset[f"table/du_toit_2017_test_1_opr_zero_crossings"]
df_prox_1_toas = dataset[f"table/du_toit_2017_test_1_prox_1_toas"]

df_prox_1 = transform_ToAs_to_AoAs(
    df_opr_zero_crossings,
    df_prox_1_toas
)

blade_arrival_count, histogram_bins = np.histogram(
    df_prox_1["AoA"],
    bins=np.linspace(0, 2*np.pi, 50)
)
```

1.	The `np.linspace` function creates 50 equidistant values between 0 and $2 \pi$. These values are used as the bin edges for the histogram. The `np.histogram` function returns the number of values that fall within each bin, as well as the bin edges. There are 49 bins, but 50 bin edges. The number of bins is one less than the number of bin edges.

The histogram is shown in [Figure 1](#figure_01) below. For convenience, we have changed the x-axis units to degrees.

<script src="all_aoas_histogram.js" > </script>
<div>
	<div>
		<canvas id="ch03_all_aoas_histogram"'></canvas>
	</div>
	<script>
		async function render_chart_all_aoas_histogram() {
			const ctx = document.getElementById('ch03_all_aoas_histogram');
			// If this is a mobile device, set the canvas height to 400
			if (window.innerWidth < 500) {
				ctx.height = 400;
			}
			while (typeof Chart == "undefined") {
				await new Promise(r => setTimeout(r, 1000));
			}
			Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
			window.fig_all_aoas_histogram = new Chart(ctx, window.all_aoas_histogram);
			window.fig_all_aoas_histogram_reset = function resetZoomFig1() {
					window.fig_all_aoas_histogram.resetZoom();
				}
			}
		render_chart_all_aoas_histogram();
	</script>
	<a onclick="window.fig_all_aoas_histogram_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_01'>Figure 1</a></strong>: 
	The AoAs of all the blades from a single proximity probe. The AoAs are grouped into 49 bins. The number of AoAs that fall within each bin is shown on the y-axis. The bin edges are shown on the x-axis. Five distinct blade groups can be identified.
  </figcaption>
</figure>

In [Figure 1](#figure_01) above, we can identify the 5 blades with the naked eye. Each blade appears as a single vertical bar at a circumferential position. Blade 2's group seems anomalous. It appears to be spread over two short adjacent bars, instead of one narrow bar. The reason for this is because the AoAs of blade 2 *happens to fall* on both sides of the 88.16° histogram bin edge.

Suppose we decided, for whatever reason, that all blades arriving earlier than 88.16° belong to blade 1. The results would be disastrous. We would be assigning AoAs from blade 2 to blade 1. This would cause the AoAs for blade's 1 and 2 to be incorrect, thereby turning our exquisitely measured BTT data into garbage.

To the human eye, its tempting to scoff at such a mistake. It is, after all, trivial to tell them apart. Unfortunately, it is impractical to parse each set of AoA values by eye. We need to develop an algorithm to do this for us.

!!! question "Outcomes"

	:material-checkbox-blank-outline: Understand that we need to calculate the optimal bin edges to group the AoAs into blade bins.

    :material-checkbox-blank-outline: Understand that we can calculate the optimal bin edges by minimizing a quality factor, $Q$.

	:material-checkbox-blank-outline: Understand that the first blade may arrive close to the OPR zero-crossing time, which means it could either appear at the start or end of the revolution. We need to cater for this scenario.
	
	:material-checkbox-blank-outline: Write a function that determines the optimal blade bins for a set of AoA values, and split the proximity probe AoA DataFrames into several individual blade DataFrames.


## Determine blade bins

Let's take an initial stab at the optimal way to bin the blades. We'll proceed from the example, where we have 5 blades. A reasonable guess for our bin edges would be to split the distance from 0 to 360 degrees into 5 equidistant sections, thereby resulting in the following 6 bin edges:

```
bin_edges = [0, 72, 144, 216, 288, 360]
```

Though this might work for the example were busy with, it would fail for other examples. If, for example, the tachometer had been located 16.16° earlier - thereby causing the second blade's grouping to be around 72° - we would have exactly the problem described above.

A much better guess would be to offset the bin edges such that the second edge (currently 72°), falls right between the first two AoA values. The first two AoA values are 16° and 88.2°. The optimal edge between them would therefore be 52.1°. We can achieve this by shifting the bin edges 72 - 52.1 = 19.9° earlier. 

The new bin edges would be:

```
bin_edges = [-19.9,  52.1, 124.1, 196.1, 268.1, 340.1]
```

This bin edge choice would maximize the distance between each blade's AoAs and its respective bin edges. Put conversely, it would minimize the distance between each blade's AoAs and the centre of the bin. 

We can therefore phrase the problem as such: determine a constant bin edge offset, `d_theta`, that minimizes the mean distance between the AoA values inside each bin and the center of said bin.

## Following along
The worksheet for this chapter can be downloaded here <a href="https://github.com/Bladesight/bladesight-worksheets/blob/master/intro_to_btt/ch_04_worksheet.ipynb" target="_blank"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="Open In Github"/></a>.


You can open a Google Colab session of the worksheet by clicking here: <a href="https://colab.research.google.com/github/Bladesight/bladesight-worksheets/blob/master/intro_to_btt/ch_04_worksheet.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>.

You need to use one of the following Python versions to run the worksheet:
<img src="https://img.shields.io/badge/python-3.6-blue.svg">
<img src="https://img.shields.io/badge/python-3.7-blue.svg">
<img src="https://img.shields.io/badge/python-3.8-blue.svg">
<img src="https://img.shields.io/badge/python-3.9-blue.svg">
<img src="https://img.shields.io/badge/python-3.10-blue.svg">
<img src="https://img.shields.io/badge/python-3.11-blue.svg">


## Algorithm to determine alignment bins

Let's define a quality factor, $Q$, that represents the quality of our alignment. This factor is determined by calculating the sum of squared errors between each bin center and the AoA values falling inside said bin.

$$
Q = \sum_{b=1}^{B} \sum_{n=1}^{N_b} \left (\hat{AoA}_{b} - AoA_{b, n}\right)^2
$$

<figure markdown>
  <figcaption><strong><a name='equation_01'>Equation 1</a></strong></figcaption>
</figure>

??? info "Symbols"
	| Symbol | Description |
	| :---: | :--- |
	| $Q$ | The blade binning quality factor|
	| $b$ | The blade index|
	| $B$ | The total number of blades|
	| $n$ | The index of the AoA inside bin $b$ |
	| $N$ | The total number of AoAs inside bin $b$ |
	| $\hat{AoA}_{b}$ | The center AoA of bin $b$ |
	| $AoA_{b, n}$ | The $n$th AoA inside bin $b$ |

	Below, we show the domains of the variables for this example:
	
	$$
	\begin{align}
		Q &\in [0, \infty] \\
		b &\in [0,1,2,3,4] \\
		B &= 5 \\
		n &\in [0, ..., N] \\
		N &\rightarrow \textrm{computed for each bin} \\
		\hat{AoA}_{b} &\in [0, 2 \pi] \\
		AoA_{b, n} &\in [0, 2 \pi] \\
	\end{align}
	$$

Below we present a function that calculates Q for a given blade number and offset.

``` py linenums="1"
def calculate_Q(
    arr_aoas : np.ndarray,
    d_theta : float,
    N : int
) -> Tuple[float, np.ndarray]:
    bin_edges = np.linspace(0 + d_theta, 2*np.pi + d_theta, N + 1)
    Q = 0
    for b in range(N):
        left_edge = bin_edges[b]
        right_edge = bin_edges[b + 1]
        bin_mask = (arr_aoas > left_edge) & (arr_aoas <= right_edge)
        
		bin_centre = (left_edge + right_edge)/2
		Q += np.sum(
            (
                arr_aoas[bin_mask] 
                - bin_centre
            )**2 
        )
    if np.sum(arr_aoas < bin_edges[0]) > 0:
        return np.nan, bin_edges
    if np.sum(arr_aoas > bin_edges[-1]) > 0:
        return np.nan, bin_edges
    return Q, bin_edges
```

Let's go through the main parts of this function.

### Initialization

```py linenums="1"
def calculate_Q(
	arr_aoas : np.ndarray,
	d_theta : float,
	N : int
) -> Tuple[float, np.ndarray]:
	bin_edges = np.linspace(0 + d_theta, 2*np.pi + d_theta, N + 1)
	Q = 0
# Some code omitted below 👇
```

??? note "Full code"
	``` py linenums="1" hl_lines="1 2 3 4 5 6 7"
	def calculate_Q(
		arr_aoas : np.ndarray,
		d_theta : float,
		N : int
	) -> Tuple[float, np.ndarray]:
		bin_edges = np.linspace(0 + d_theta, 2*np.pi + d_theta, N + 1)
		Q = 0
		for b in range(N):
			left_edge = bin_edges[b]
			right_edge = bin_edges[b + 1]
			bin_mask = (arr_aoas > left_edge) & (arr_aoas <= right_edge)
			
			bin_centre = (left_edge + right_edge)/2
			Q += np.sum(
				(
					arr_aoas[bin_mask] 
					- bin_centre
				)**2 
			)
		if np.sum(arr_aoas < bin_edges[0]) > 0:
			return np.nan, bin_edges
		if np.sum(arr_aoas > bin_edges[-1]) > 0:
			return np.nan, bin_edges
		return Q, bin_edges
	```

In lines 1 - 4, the function name and input arguments are defined. `arr_aoas` is a numpy array of AoA values, in *radians*, `d_theta` is our guess for the shift to apply to the "standard" bin edges, and `N` is the number of blades. 

In Line 5, we specify that the function returns a tuple - a fancy word for an immutable list - with two values: `Q` and `bin_edges`. `Q` is the quality factor, and `bin_edges` are the bin edges resulting from this choice of `d_theta`.

!!! note "Note"
	You need to import the `Tuple` type from the `typing` library.

	``` py
	from typing import Tuple
	```

In Line 6, we calculate the `bin_edges` using `d_theta`.

In Line 7, we initialize the quality factor to zero.

### Identify the AoAs falling inside each bin

```py linenums="7"
# Some code omitted above 👆
for b in range(N):
	left_edge = bin_edges[b]
	right_edge = bin_edges[b + 1]
	bin_mask = (arr_aoas > left_edge) & (arr_aoas <= right_edge)
# Some code omitted below 👇
```

??? note "Full code"
	``` py linenums="1" hl_lines="8 9 10 11"
	def calculate_Q(
		arr_aoas : np.ndarray,
		d_theta : float,
		N : int
	) -> Tuple[float, np.ndarray]:
		bin_edges = np.linspace(0 + d_theta, 2*np.pi + d_theta, N + 1)
		Q = 0
		for b in range(N):
			left_edge = bin_edges[b]
			right_edge = bin_edges[b + 1]
			bin_mask = (arr_aoas > left_edge) & (arr_aoas <= right_edge)
			
			bin_centre = (left_edge + right_edge)/2
			Q += np.sum(
				(
					arr_aoas[bin_mask] 
					- bin_centre
				)**2 
			)
		if np.sum(arr_aoas < bin_edges[0]) > 0:
			return np.nan, bin_edges
		if np.sum(arr_aoas > bin_edges[-1]) > 0:
			return np.nan, bin_edges
		return Q, bin_edges
	```

In Line 8, we enter our main loop, whereby we will consider each bin in turn.

In lines 9 and 10, we select the left and right edges of the bin we are currently considering.

In Line 11, we determine which AoA values fall within the bin we are currently considering. `bin_mask` is a boolean array, with the same length as `arr_aoas`. If the mask has a value of `True` at a given index, then the AoA value at that index falls within the bin we are currently considering.

### Determine each bin's Q factor contribution

```py linenums="12"
# Some code omitted above 👆
bin_centre = (left_edge + right_edge)/2
Q += np.sum(
	(
		arr_aoas[bin_mask] 
		- bin_centre
	)**2 
)
# Some code omitted below 👇
```

??? note "Full code"
	``` py linenums="1" hl_lines="13 14 15 16 17 18 19"
	def calculate_Q(
		arr_aoas : np.ndarray,
		d_theta : float,
		N : int
	) -> Tuple[float, np.ndarray]:
		bin_edges = np.linspace(0 + d_theta, 2*np.pi + d_theta, N + 1)
		Q = 0
		for b in range(N):
			left_edge = bin_edges[b]
			right_edge = bin_edges[b + 1]
			bin_mask = (arr_aoas > left_edge) & (arr_aoas <= right_edge)
			
			bin_centre = (left_edge + right_edge)/2
			Q += np.sum(
				(
					arr_aoas[bin_mask] 
					- bin_centre
				)**2 
			)
		if np.sum(arr_aoas < bin_edges[0]) > 0:
			return np.nan, bin_edges
		if np.sum(arr_aoas > bin_edges[-1]) > 0:
			return np.nan, bin_edges
		return Q, bin_edges
	```
In Line 13, we calculate the centre of this bin, by taking the average of the left and right edges. This is $\hat{AoA}_{b}$ in [Equation 1](#equation_01).

In lines 14 - 19, we calculate the squared difference between each AoA value and the bin centre. We then sum these squared differences and add it to $Q$. If the bin centre is close to the AoA values in this bin, $Q$ will increase minimally. If the bin centre is far from the AoA values in this bin, $Q$ will increase significantly. 

### Sanity checks and return
```py linenums="19"
# Some code omitted above 👆
if np.sum(arr_aoas < bin_edges[0]) > 0:
	return np.nan, bin_edges
if np.sum(arr_aoas > bin_edges[-1]) > 0:
	return np.nan, bin_edges
return Q, bin_edges
```

??? note "Full code"
	``` py linenums="1" hl_lines="20 21 22 23 24" 
	def calculate_Q(
		arr_aoas : np.ndarray,
		d_theta : float,
		N : int
	) -> Tuple[float, np.ndarray]:
		bin_edges = np.linspace(0 + d_theta, 2*np.pi + d_theta, N + 1)
		Q = 0
		for b in range(N):
			left_edge = bin_edges[b]
			right_edge = bin_edges[b + 1]
			bin_mask = (arr_aoas > left_edge) & (arr_aoas <= right_edge)
			
			bin_centre = (left_edge + right_edge)/2
			Q += np.sum(
				(
					arr_aoas[bin_mask] 
					- bin_centre
				)**2 
			)
		if np.sum(arr_aoas < bin_edges[0]) > 0:
			return np.nan, bin_edges
		if np.sum(arr_aoas > bin_edges[-1]) > 0:
			return np.nan, bin_edges
		return Q, bin_edges
	```

The sanity checks added here are imperative. Our logic in lines 8 to 19 only considers AoA values that fall within one of the bins. It is possible for the offset guess `d_theta` to shift the bins so much that some AoA values do not fall within any bin. These AoA values will therefore not contribute to $Q$, and our optimal `d_theta` will be a fraud!

We therefore check, in lines 20 and 22, whether any AoA values occur before the left most bin edge, or after the right most edge. If this is the case, we return `np.nan` for `Q`, meaning this `d_theta` is invalid.

Finally, in Line 24, we return `Q` and `bin_edges`.

## Implementation example

We can see the algorithm in action by iterating over a range of `d_theta` values. We'll start by attempting offsets that are between $-\frac{\pi}{5}$ and $\frac{\pi}{5}$, essentially shifting the "standard" bins left and right by 72°. We'll then plot the resulting $Q$ values.

``` py linenums="1"
B = 5#(1)!
d_thetas = np.linspace(-np.pi/B, np.pi/B, 200) #(2)!
arr_aoas = df_prox_1["AoA"].to_numpy()#(3)!
Qs = [] #(4)!
optimal_Q, optimal_bin_edges, optimal_d_theta = np.inf, None, None#(5)!
for d_theta in d_thetas:#(6)!
    Q, bin_edges = calculate_Q(arr_aoas, d_theta, B)
    if Q < optimal_Q:#(7)!
        optimal_Q = Q*1
        optimal_bin_edges = bin_edges
        optimal_d_theta = d_theta*1
    Qs.append(Q)#(8)!
```

1.	We specify the number of blades. This is used to calculate the bin edges.
2.	We specify the range of offsets to consider. We'll consider 200 offsets between $-\frac{\pi}{5}$ and $\frac{\pi}{5}$. 
3.	We convert the AoA DataFrame into a numpy array. Our function requires a Numpy array.
4.	We initialize an empty list to store the quality factors.
5.	We initialize the optimal Q value to infinity. This is a trick to ensure that the first value we calculate is always the optimal value. It can be unseated by a better value later.
6.	We iterate over the range of offsets to consider.
7.	If the quality factor we just calculated is less than the optimal quality value, we update the optimal values.
8.	We append the quality factor to a list. This will be used to plot the quality factor as a function of the offset.

The `Q` factor for each `d_theta` is shown in [Figure 2](#figure_02) below.

<script src="q_factor_plot.js" > </script>
<div>
	<div>
		<canvas id="ch03_q_factor_plot"'></canvas>
	</div>
	<script>
		async function render_chart_q_factor_plot() {
			const ctx = document.getElementById('ch03_q_factor_plot');
			// If this is a mobile device, set the canvas height to 400
			if (window.innerWidth < 500) {
				ctx.height = 400;
			}
			while (typeof Chart == "undefined") {
				await new Promise(r => setTimeout(r, 1000));
			}
			Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
			window.fig_q_factor_plot = new Chart(ctx, window.q_factor_plot);
			window.fig_q_factor_plot_reset = function resetZoomFig2() {
					window.fig_q_factor_plot.resetZoom();
				}
			}
		render_chart_q_factor_plot();
	</script>
	<a onclick="window.fig_q_factor_plot_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_02'>Figure 2</a></strong>: 
	The quality factor, Q, as a function of the offset, d_theta. The optimal offset is -20.08°.
  </figcaption>
</figure>
And voila! We see that the optimal offset is -20.08 degrees. We also see that there are many offsets that result in an invalid quality factor.

## Grouping the blades

It is trivial, after having determined the optimal bin edges, to process the proximity probe AoA DataFrame such that we have a separate DataFrame for each blade. 

We present code that does this below.

``` py linenums="1"
blade_dfs = []
for b in range(B):
    ix_bin = (
        (df_prox_1["AoA"] > optimal_bin_edges[b])
        & (df_prox_1["AoA"] <= optimal_bin_edges[b + 1])
    )
    blade_dfs.append(
        df_prox_1.loc[ix_bin]
    )
```

``` py linenums="1"
>>> for b in range(B):
>>>     print(f"Blade {b} mean: {blade_dfs[b]['AoA'].mean()}, std: {blade_dfs[b]['AoA'].std()}")
```

``` console
Blade 0 mean: 0.280844143512115, std: 0.0014568064245125216
Blade 1 mean: 1.5390655934492143, std: 0.001784799129959647
Blade 2 mean: 2.788312464321002, std: 0.0015687549238434136
Blade 3 mean: 4.045575640802255, std: 0.0017088093157144036
Blade 4 mean: 5.305095908129366, std: 0.0014525709531342695
```

Finally, we now have 5 dataframes, each one containing only the information from a single blade. 

## Wrapping blades

Let's implement a static shift to the AoA values from the original, non-binned, DataFrame. Note that a static shift in AoA values does not change any objective information. We should be able to find the exact same $Q$ value as we did in the previous section.

``` py linenums="1" hl_lines="3"
df_prox_1_shifted = df_prox_1.copy(deep=True)
df_prox_1_shifted['AoA'] = df_prox_1_shifted['AoA'] - 0.280844143512115
df_prox_1_shifted['AoA'] = df_prox_1_shifted['AoA'] % (2*np.pi)

B = 5
d_thetas = np.linspace(-np.pi/B, np.pi/B, 200)
arr_aoas = df_prox_1_shifted["AoA"].to_numpy()
Qs = []
optimal_Q, optimal_bin_edges, optimal_d_theta = np.inf, None, None
for d_theta in d_thetas:
    Q, bin_edges = calculate_Q(arr_aoas, d_theta, B)
    if Q < optimal_Q:
        optimal_Q = Q*1
        optimal_bin_edges = bin_edges
        optimal_d_theta = d_theta*1
    Qs.append(Q)
```

In Line 1 above, we create a copy of our original DataFrame. In Line 2, we shift the AoA values by blade 0's mean AoA value as calculated in the previous section. 

We've highlighted Line 3 because it is so important. Our artificially constructed `df_prox_1_shifted` DataFrame from Line 2 will contain some `AoA` values that are negative.

In practice, our function that converts ToAs to AoAs, `transform_ToAs_to_AoAs`, will always produce AoA values that are positive. These values will therefore appear *at the end of the next revolution*. To simulate this effect, we wrap the AoA values to the range $[0, 2 \pi]$ in Line 3.

We then, from Line 5 onwards, repeat the same process as before, attempting to find the optimal bin edges. The only difference is that we use the `df_prox_1_shifted` DataFrame instead of the `df_prox_1` DataFrame.

We print the optimal $Q$ value below.

``` py linenums="1"
>>> print(optimal_Q)
>>> print(optimal_bin_edges)
inf
None
```

The optimal $Q$ is infinity and the optimal bin edges equals `None` :scream: !

This is what happens when you've got one blade that arrives at a proximity probe at approximately the same time as the tacho's zero-crossing time. Some of the AoA values will appear at the end of the previous revolution, and some will appear at the beginning of the current revolution.

The algorithm we've developed will fail for such cases, because there will be some values that do not fall within the binning.

We can cater for this scenario by adapting our `calculate_Q` algorithm to handle AoAs that fall outside the bin edges.

I'm going to leave it to you to make this change. 

{==

:material-pencil-plus-outline: Write an updated `calculate_Q` function that incorporates the AoA values that fall outside the bin edges, instead of simply returning `nan` for Q.

==}

??? example "Reveal answer (Please try it yourself before peeking)"
	``` py linenums="1" hl_lines="20 21 22 23 24 25 26 27 28 29 31 32 33 34 35 36 37 38 39 40"  
	def calculate_Q(
		arr_aoas : np.ndarray,
		d_theta : float,
		N : int
	) -> Tuple[float, np.ndarray]:
		bin_edges = np.linspace(0 + d_theta, 2*np.pi + d_theta, N + 1)
		Q = 0
		for b in range(N):
			left_edge = bin_edges[b]
			right_edge = bin_edges[b + 1]
			bin_centre = (left_edge + right_edge)/2
			bin_mask = (arr_aoas > left_edge) & (arr_aoas <= right_edge)
			Q += np.sum(
				(
					arr_aoas[bin_mask] 
					- bin_centre
				)**2 
			)
		if np.sum(arr_aoas < bin_edges[0]) > 0:
			left_edge_last = bin_edges[N-1]
			right_edge_last = bin_edges[N]
			bin_centre_last = (left_edge_last + right_edge_last)/2
			bin_mask = arr_aoas <= bin_edges[0]
			Q += np.sum(
				(
					(2*np.pi - arr_aoas[bin_mask]) 
					- bin_centre_last
				)**2 
			)
		if np.sum(arr_aoas > bin_edges[-1]) > 0:
			left_edge_first = bin_edges[0]
			right_edge_first = bin_edges[1]
			bin_centre_first = (left_edge_first + right_edge_first)/2
			bin_mask = arr_aoas > bin_edges[-1]
			Q += np.sum(
				(
					(arr_aoas[bin_mask] - 2*np.pi) 
					- bin_centre_first
				)**2 
			)
		return Q, bin_edges
	```

	In the `calculate_Q` above, we've added lines of code in the 'sanity check' sections to handle AoA values that fall outside the bin edges.

	In lines 20 - 29, we check for AoA values that occur before the left most bin edge. These values are therefore being treated as if they occur in the last bin.

	In lines 31 - 40, we check for AoA values that occur after the right most bin edge. These values are therefore being treated as if they occur in the first bin.

{==

:material-pencil-plus-outline: After you've finished the above, create a function called `transform_prox_AoAs_to_blade_AoAs` that receives the proximity probe AoA DataFrame and the number of blades on the rotor, and returns a list of DataFrames, each one containing the AoA values for a single blade. This function should therefore perform both the determination of the optimal bin edges, as well as the splitting of the AoA values into separate DataFrames.

==}

??? example "Reveal answer (Please try it yourself before peeking)"
	``` py linenums="1" 
	def transform_prox_AoAs_to_blade_AoAs(
		df_prox : pd.DataFrame,
		B : int,
	) -> List[pd.DataFrame]:
		""" This function takes a dataframe containing the AoA values of a proximity probe, 
		and returns a list of dataframes, each containing the AoA values of a single blade.

		Args:
			df_prox (pd.DataFrame): The dataframe containing the AoA values 
				of the proximity probe.
			B (int): The number of blades.

		Returns:
			List[pd.DataFrame]: A list of dataframes, each containing the 
				AoA values of a single blade.
		"""
		d_thetas = np.linspace(-2*np.pi/B, 0, 200)
		arr_aoas = df_prox["AoA"].to_numpy()
		Qs = []
		optimal_Q, optimal_bin_edges, optimal_d_theta = np.inf, None, None
		for d_theta in d_thetas:
			Q, bin_edges = calculate_Q(arr_aoas, d_theta, B)
			if Q < optimal_Q:
				optimal_Q = Q*1
				optimal_bin_edges = bin_edges
				optimal_d_theta = d_theta*1
			Qs.append(Q)

		blade_dfs = []
		for b in range(B):
			ix_bin = (
				(df_prox["AoA"] > optimal_bin_edges[b])
				& (df_prox["AoA"] <= optimal_bin_edges[b + 1])
			)
			if b == 0:
				ix_bin = ix_bin | (df_prox["AoA"] > optimal_bin_edges[-1])
				df_bin = (
					df_prox
					.loc[ix_bin]
					.copy()
					.reset_index(drop=True)
					.sort_values("ToA")
				)

				ix_wrap = df_bin["AoA"] > optimal_bin_edges[-1]
				df_bin.loc[ ix_wrap, "AoA"] = df_bin.loc[ ix_wrap, "AoA"] - 2*np.pi 
			elif b == B-1:
				ix_bin = ix_bin | (df_prox["AoA"] <= optimal_bin_edges[0])
				df_bin = (
					df_prox
					.loc[ix_bin]
					.copy()
					.reset_index(drop=True)
					.sort_values("ToA")
				)

				ix_wrap = df_bin["AoA"] > optimal_bin_edges[-1]
				df_bin.loc[ ix_wrap, "AoA"] = 2*np.pi - df_bin.loc[ ix_wrap, "AoA"] 
			else:
				df_bin = (
					df_prox
					.loc[ix_bin]
					.copy()
					.reset_index(drop=True)
					.sort_values("ToA")
				)
			blade_dfs.append(
				df_bin
			)
		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		ADD CODE HERE TO SHIFT THE BLADES TO THE CORRECT POSITIONS
		return blade_dfs
	```

	We've decided to only shift the `d_theta` value between $-\frac{2\pi}{B}$ and 0. This is simply a matter of preference, because we'd like the first blade to occur as close as possible to 0 degrees.


## Conclusion

Blade alignment is a bit like parallel parking - it seems so simple we often approach it without the respect it deserves, but it can make you look like a fool.

We started this tutorial with raw time stamps, we now have an AoA DataFrame for every blade arriving at a proximity probe. Everything we've done up to this point involves only a single proximity probe. BTT systems typically have multiple proximity probes. 

In the next chapter, we'll combine information from multiple probes together. We'll also investigate a nice visual way to check if we've done our alignment and grouping properly: *the stack plot*.

!!! question "Outcomes"

	:material-checkbox-marked:{ .checkbox-success .heart } Understand that we need to calculate the optimal bin edges to group the AoAs into blade bins.

    :material-checkbox-marked:{ .checkbox-success .heart } Understand that we can calculate the optimal bin edges by minimizing a quality factor, $Q$.

	:material-checkbox-marked:{ .checkbox-success .heart } Understand that the first blade may arrive close to the OPR zero-crossing time, which means it could either appear at the start or end of the revolution. We need to cater for this scenario.
	
	:material-checkbox-marked:{ .checkbox-success .heart } Write a function that determines the optimal blade bins for a set of AoA values, and split the proximity probe AoA DataFrame into several individual blade DataFrames.

## Acknowledgements
I thank XXX for reviewing this chapter.

\bibliography

<div style='display:flex'>
    <div>
        <a target="_blank" href="https://www.bladesight.com" class="" title="Dawie Diamond" style="border-radius:100%;"> 
            <img src="https://github.com/Bladesight.png?size=300" alt="Dawie Diamond" style="
            border-radius: 100%;
            width: 4.0rem;
        ">
        </a>
    </div>
    <div style='margin-left:2rem'>
        <p>
            <strong>Dawie Diamond</strong>
        </p>
        <p>
            2023-10-10
        </p>
    </div>
</div>