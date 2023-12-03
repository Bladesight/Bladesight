---
date: 2023-10-15
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
description: This chapter explains how to zero and filter, and scale the rotor blade AoA values.
robots: index, follow, Blade Tip Timing, BTT, Non Intrusive Stress Measurement, NSMS, Time of Arrival, Turbine blade,Mechanical Vibration
template: main_intro_to_btt.html
card_title: Intro to BTT Ch6 - Data Zeroing and Filtering
card_url: "ch6/"
---
# Data Zeroing and Filtering

In the previous chapter, we built algorithms that combine data from several proximity probes. We ended up with a single Angle of Arrival (AoA) DataFrame for each blade. We called this DataFrame the rotor blade AoA DataFrame. It contains all the AoAs for the same blade arriving at different probes. 

In this chapter, we're going to convert those AoAs into tip deflections. The process entails zeroing, filtering, and scaling the AoAs according to the rotor's radius

!!! question "Outcomes"

	:material-checkbox-blank-outline: Understand that the AoA values are offset by a constant value, and that this offset is unrelated to the deflection of the blade. 

	:material-checkbox-blank-outline: Understand that the normalized AoA values can be scaled by the rotor radius to obtain the tip deflection.

    :material-checkbox-blank-outline: Understand that the tip deflections may exhibit non-vibration related variation that is proportional to the shaft speed.

	:material-checkbox-blank-outline: Understand that we can use a low pass filter to get rid of noise. 
	
	:material-checkbox-blank-outline: Understand that we can use a peak to peak vibration to identify resonance events.

	:material-checkbox-blank-outline: Write a single function that scales, zeroes and filters the rotor blade AoA values.

## Following along
The worksheet for this chapter can be downloaded here <a href="https://github.com/Bladesight/bladesight-worksheets/blob/master/intro_to_btt/ch_06_worksheet.ipynb" target="_blank"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="Open In Github"/></a>.


You can open a Google Colab session of the worksheet by clicking here: <a href="https://colab.research.google.com/github/Bladesight/bladesight-worksheets/blob/master/intro_to_btt/ch_06_worksheet.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>.

You need to use one of the following Python versions to run the worksheet:
<img src="https://img.shields.io/badge/python-3.6-blue.svg">
<img src="https://img.shields.io/badge/python-3.7-blue.svg">
<img src="https://img.shields.io/badge/python-3.8-blue.svg">
<img src="https://img.shields.io/badge/python-3.9-blue.svg">
<img src="https://img.shields.io/badge/python-3.10-blue.svg">
<img src="https://img.shields.io/badge/python-3.11-blue.svg">


## Zeroing the AoAs
Let's load a new dataset into memory. The dataset is from the same rotor we've been using in the previous chapters. The dataset has measurements from three eddy current probes for a run-up and run-down of the rotor:

``` py linenums="1"
from bladesight import Datasets

ds = Datasets["data/intro_to_btt/intro_to_btt_ch06"]
df_opr_zero_crossings = ds['table/opr_zero_crossings']
df_prox_1 = ds['table/prox_1_toas']
df_prox_2 = ds['table/prox_2_toas']
df_prox_3 = ds['table/prox_3_toas']
```

I've included the functions that we've created in the previous chapter into the `bladesight` module. Specifically, the `get_rotor_blade_AoAs` function that we created in the coding exercises has been added to the `btt` module. It can now be used as a single entrypoint to convert the OPR zero-crossing times and the proximity probe ToAs into rotor blade AoA DataFrames.

``` py linenums="1"
from bladesight.btt import get_rotor_blade_AoAs
blade_AoAs = get_rotor_blade_AoAs(
    df_opr_zero_crossings, 
    [
        df_prox_1, 
        df_prox_2, 
        df_prox_3
    ],
    probe_spacings=np.deg2rad([10,20]),
    B=5
)
```

We indicate above that the spacing from the second and third probe to the first probe is 10 and 20 degrees respectively. We also indicate that there are 5 blades. The result is a list of DataFrames, one for each blade. The first DataFrame is shown in [Table 1](#table_01) below:

<figure markdown>
  <figcaption><a id='table_01'><strong>Table 1:</strong></a> The first 5 rows of rotor blade 1's AoA DataFrame. </figcaption>
  </figcaption>
</figure>
{{ read_csv('docs/tutorials/intro_to_btt/ch6/df_blade_1_head.csv') }}

Let's plot the AoAs for the first blade in the hope that we can make sense of it. The AoAs are shown against revolution number in [Figure 1](#figure_01) below.

<script src="blade_1_aoas_raw.js" > </script>
<div>
	<div>
		<canvas id="ch06_blade_1_aoas_raw"'></canvas>
	</div>
	<script>
		async function render_chart_blade_1_aoas_raw() {
			const ctx = document.getElementById('ch06_blade_1_aoas_raw');
			// If this is a mobile device, set the canvas height to 400
			if (window.innerWidth < 500) {
				ctx.height = 400;
			}
			while (typeof Chart == "undefined") {
				await new Promise(r => setTimeout(r, 1000));
			}
			Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
			window.fig_blade_1_aoas_raw = new Chart(ctx, window.blade_1_aoas_raw);
			window.fig_blade_1_aoas_raw_reset = function resetZoomFig1() {
					window.fig_blade_1_aoas_raw.resetZoom();
				}
			}
		render_chart_blade_1_aoas_raw();
	</script>
	<a onclick="window.fig_blade_1_aoas_raw_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_01'>Figure 1</a></strong>: The AoAs for the first blade. We see that each blade's AoA is offset by a constant value. We can say that the AoAs are <strong>not zeroed</strong>.  The shaft speed is also indicated on a second y-axis.</figcaption>
  </figcaption>
</figure>

In [Figure 1](#figure_01) above, we see three seemingly horizontal lines representing the AoAs for the blade arriving at probes one, two and three respectively. We can say that each line has a constant offset from the start of the revolution. This constant is the average distance travelled by the shaft from the start of each revolution until the blade reaches the probe. 

This constant offset is unrelated to the deflection of the blade. We therefore say that the values in [Figure 1](#figure_01) are *not zeroed*. We need to remove the static part of the signal before we can proceed with the analysis.

### Normalization and scaling

One of the simplest ways to remove the static part of the signal is to subtract the mean of the AoA column from each AoA value. We can use the code below to achieve this.

``` py linenums="1" 
df["AoA_norm_p1"] = df[f"AoA_p1"].mean() - df[f"AoA_p1"] 
df["AoA_norm_p2"] = df[f"AoA_p2"].mean() - df[f"AoA_p2"]
df["AoA_norm_p3"] = df[f"AoA_p3"].mean() - df[f"AoA_p3"]
```

The normalized AoA column can now be *scaled* using the rotor radius, resulting in the tip deflection. The code below shows how to achieve this:

``` py linenums="1"
R = 164000 #(1)!
df["x_p1"] = R * df["AoA_norm_p1"]
df["x_p2"] = R * df["AoA_norm_p2"]
df["x_p3"] = R * df["AoA_norm_p3"]
```

1.  This rotor's radius is 164 mm. We express the tip deflection in microns. This is a personal preference. 

We show the tip deflections in [Figure 2](#figure_02) below.

<script src="blade_1_x_normalized.js" > </script>
<div>
    <div>
        <canvas id="ch06_blade_1_x_normalized"'></canvas>
    </div>
    <script>
        async function render_chart_blade_1_x_normalized() {
            const ctx = document.getElementById('ch06_blade_1_x_normalized');
            // If this is a mobile device, set the canvas height to 400
            if (window.innerWidth < 500) {
                ctx.height = 400;
            }
            while (typeof Chart == "undefined") {
                await new Promise(r => setTimeout(r, 1000));
            }
            Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
            window.fig_blade_1_x_normalized = new Chart(ctx, window.blade_1_x_normalized);
            window.fig_blade_1_x_normalized_reset = function resetZoomFig2() {
                    window.fig_blade_1_x_normalized.resetZoom();
                }
            }
        render_chart_blade_1_x_normalized();
    </script>
    <a onclick="window.fig_blade_1_x_normalized_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_02'>Figure 2</a></strong>: The normalized tip deflections for the first blade. We see there are distinct areas where the blade vibration suddenly changes, as indicated by callout <strong>A</strong>. This is a typical example of a resonance event. We've also added, as indicated by callout <strong>B</strong>, dotted lines that serve as "breakpoints" for our detrending algorithm. The shaft speed is indicated on a second y-axis. 
  </figcaption>
</figure>

In [Figure 2](#figure_02) above, we can see that the tip deflections no longer have a seemingly large constant offset. All the tip deflection values are between +-1000 µm. There are multiple locations where the tip deflection seem to suddenly change. Depending on the probe signal we are looking at, the change in tip deflection may be positive or negative.

One such location is indicated by callout *A*. We refer to these events as *resonance events*. You can zoom into the resonance event by dragging across the plot. Each probe's change in tip deflections seems different. This is counter-intuitive, because the blade's vibrational state can surely not change much within a single shaft revolution. The reason we see such behavior is because of *aliasing*. 

Aliasing is one of the infamous reasons that make BTT signal processing challenging. We are going to discuss aliasing in depth in the next chapter. For now, you only need to understand that resonance events cause the tip deflections to change suddenly, and that different probes will show different tip deflection changes. 

### Piecewise linear detrending
We have done well to remove the static offset from each probe's signal. Unfortunately, there are still some static effects in the signal that have nothing to do with tip deflection. 

From [Figure 2](#figure_02) it seems as though the tip deflections change proportionally to the shaft speed. It is highly unlikely that this change is related to tip deflection. There may be many reasons for shaft speed related shifts. In this case, the proportional change in tip deflection is due to the fact that the eddy current probes we are using have a limited frequency response function. The amplitude of the pulses (as discussed in Chapter 2) therefore become smaller as the shaft speed increases, causing later triggering of the ToAs. This later triggering manifests itself as apparent tip deflection changes.

Let's remove this proportional change using a *detrending* algorithm.

A detrending algorithm is a very simple algorithm that removes a static trend from a signal. The Python library `scipy` has a built-in detrending algorithm.

``` py linenums="1"
from scipy.signal import detrend
```

This function fits and subtracts a linear curve from the signal. A linear curve, though, is not what we need. By looking at [Figure 2](#figure_02) above, we see that there are multiple regions that have different linear trends. This is especially true the moment the shaft speed reverses around revolution 1436. There are also other regions where the linear trend changes.

Fortunately, the detrending algorithm allows us to specify *breakpoints* where the linear trend changes. From [Figure 2](#figure_02) above, I've eyeballed the breakpoints that demarcate new linear sections at shaft revolutions 217, 1128, 1436, 1784, and 2670. The breakpoints are indicated by the dotted lines in [Figure 2](#figure_02) above (one breakpoint is indicated by callout B). 

The detrending algorithm can be applied for all three probes using the code below:

``` py linenums="1"
bps = np.array([217, 1128, 1436, 1784, 2670])
df[f"x_p1"] = detrend(
    df["x_p1"],
    bp=bps
)
df[f"x_p2"] = detrend(
    df["x_p2"],
    bp=bps
)
df[f"x_p3"] = detrend(
    df["x_p3"],
    bp=bps
)
```

The resulting detrended tip deflections are shown in [Figure 3](#figure_03) below.
<script src="blade_1_x_detrended.js" > </script>
<div>
    <div>
        <canvas id="ch06_blade_1_x_detrended"'></canvas>
    </div>
    <script>
        async function render_chart_blade_1_x_detrended() {
            const ctx = document.getElementById('ch06_blade_1_x_detrended');
            // If this is a mobile device, set the canvas height to 400
            if (window.innerWidth < 500) {
                ctx.height = 400;
            }
            while (typeof Chart == "undefined") {
                await new Promise(r => setTimeout(r, 1000));
            }
            Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
            window.fig_blade_1_x_detrended = new Chart(ctx, window.blade_1_x_detrended);
            window.fig_blade_1_x_detrended_reset = function resetZoomFig3() {
                    window.fig_blade_1_x_detrended.resetZoom();
                }
            }
        render_chart_blade_1_x_detrended();
    </script>
    <a onclick="window.fig_blade_1_x_detrended_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_03'>Figure 3</a></strong>: The detrended tip deflections for the first blade. We see that the tip deflections are now centered around zero for the duration of the measurement.
  </figcaption>
</figure>

Thats much better :clap:! The tip deflections from each probe are now centered around zero. We're now in a position where we can trust that a tip deflection value is related only to the dynamic vibration of the blade.

This approach gives us great control over the zeroing process, but it would be better if we could find a way to remove the static part of the signal automatically.

### Order domain polynomial detrending

We can gain some intuition about removing shaft speed related effects by creating a scatterplot of tip deflection vs shaft speed. The scatterplot is shown in [Figure 4](#figure_04) below.


<script src="blade_1_x_vs_RPM.js" > </script>
<div>
    <div>
        <canvas id="ch06_blade_1_x_vs_RPM"'></canvas>
    </div>
    <script>
        async function render_chart_blade_1_x_vs_RPM() {
            const ctx = document.getElementById('ch06_blade_1_x_vs_RPM');
            // If this is a mobile device, set the canvas height to 400
            if (window.innerWidth < 500) {
                ctx.height = 400;
            }
            while (typeof Chart == "undefined") {
                await new Promise(r => setTimeout(r, 1000));
            }
            Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
            window.fig_blade_1_x_vs_RPM = new Chart(ctx, window.blade_1_x_vs_RPM);
            window.fig_blade_1_x_vs_RPM_reset = function resetZoomFig4() {
                    window.fig_blade_1_x_vs_RPM.resetZoom();
                }
            }
        render_chart_blade_1_x_vs_RPM();
    </script>
    <a onclick="window.fig_blade_1_x_vs_RPM_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_04'>Figure 4</a></strong>: The tip deflections have been plotted vs shaft speed for each probe. The tip deflections seem to be decreasing proportionally as shaft speed increases. We can fit a high order polynomial to each signal. This polynomial can then be subtracted from the tip deflections to remove shaft speed related effects.
  </figcaption>
</figure>

From [Figure 4](#figure_04) above, we see that the tip deflections for probes 2 and 3 seem to decrease, on average, as the shaft speed increases. Probe 1's tip deflections exhibit a smaller correlation. The resonance events within these signals seem to be short lived as a function of shaft speed. 

This allows us to fit a high-order polynomial to the tip deflections vs shaft speed. The polynomial represents the shaft speed related effects in the tip deflections. By subtracting the evaluated polynomial from each signal, we end up with a signal that only contains the tip's dynamic vibration.

The polynomial can be fit using the code below:

``` py linenums="1"
df["x_p1"] = R * df["AoA_norm_p1"]
df["x_p2"] = R * df["AoA_norm_p2"]
df["x_p3"] = R * df["AoA_norm_p3"]

poly_order = 11 #(1)!
p1 = np.polyfit(df['Omega'], df['x_p1'], poly_order)
p2 = np.polyfit(df['Omega'], df['x_p2'], poly_order)
p3 = np.polyfit(df['Omega'], df['x_p3'], poly_order)

df["x_p1"] = df["x_p1"] - np.polyval(p1, df['Omega'])
df["x_p2"] = df["x_p2"] - np.polyval(p2, df['Omega'])
df["x_p3"] = df["x_p3"] - np.polyval(p3, df['Omega'])
```

1.  Here, I've selected a polynomial order of 11. This was the first value I tried. You may need to experiment with different polynomial orders to find the best one for your case. Maybe you could write an algorithm that automatically determines this polynomial order. If you do, please let me know :smile:.

The resulting detrended tip deflections are shown in [Figure 5](#figure_05) below.

<script src="blade_1_x_poly_normalised.js" ></script>
<div>
    <div>
        <canvas id="ch06_blade_1_x_poly_normalised"'></canvas>
    </div>
    <script>
        async function render_chart_blade_1_x_poly_normalised() {
            const ctx = document.getElementById('ch06_blade_1_x_poly_normalised');
            // If this is a mobile device, set the canvas height to 400
            if (window.innerWidth < 500) {
                ctx.height = 400;
            }
            while (typeof Chart == "undefined") {
                await new Promise(r => setTimeout(r, 1000));
            }
            Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
            window.fig_blade_1_x_poly_normalised = new Chart(ctx, window.blade_1_x_poly_normalised);
            window.fig_blade_1_x_poly_normalised_reset = function resetZoomFig5() {
                    window.fig_blade_1_x_poly_normalised.resetZoom();
                }
            }
        render_chart_blade_1_x_poly_normalised();
    </script>
    <a onclick="window.fig_blade_1_x_poly_normalised_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_05'>Figure 5</a></strong>: The detrended tip deflections for the first blade after zeroing using an order domain polynomial. We see that the tip deflections are now centered around zero for the duration of the measurement.
  </figcaption>
</figure>

Great :smile:! We've managed to remove the static, non-vibration related part of the signal. We can now move on to the next step of the analysis, filtering.

!!! tip
    Zeroing can be achieved in many different ways. I've presented you with one simple method to get you going. It is, however, a discipline on its own. Maybe you can come up with different ways of zeroing the signal. If you do, please let me know :smile:.

## Filtering
We've now removed the static part of the signal. We have, essentially, implemented a high-pass filter on our signals. We are now left with high frequency noise in our signals. We can remove this noise using a low-pass filter. 

Once again, the `scipy` package in Python comes to the rescue. There are multiple filters in the `scipy.signal` module. We're going to demonstrate two methods here:

1.  A Butterworth filter.
2.  A Gaussian filter.

### Butterworth filter
A Butterworth filter is a low-pass filter. It is often implemented in analogue systems, but we can implement it digitally using the `scipy.signal.butter` function. The code below shows how to implement a Butterworth filter.

``` py linenums="1"
from scipy.signal import butter, filtfilt

butterworth_filter = butter(N=2, Wn=0.3) #(1)!

df["x_p1_filt_butter"] = filtfilt(*butterworth_filter, df["x_p1"]) #(2)!
df["x_p2_filt_butter"] = filtfilt(*butterworth_filter, df["x_p2"])
df["x_p3_filt_butter"] = filtfilt(*butterworth_filter, df["x_p3"])
```

1.  The `butter` function is responsible for *designing* the filter. By tuning the order parameter, `N`, and the cutoff frequency `Wn`, you end up with different frequency response functions. I am not going to go into filter design in this tutorial, the primary reason being I am not an expert at it. The important parameter here is the cutoff frequency `Wn`. The larger this value, the more high frequency components will be permitted to stay. In our case, the faster our resonances occur, the higher we want this cutoff. I think 0.3 is a good starting value to use.
2.  The `filtfilt` function is the part that *applies* the filter. The first argument is the filter coefficients, and the second argument is the signal that requires filtering. The `filtfilt` function is a forward-backward filter. This means that it applies the filter twice, once forward, and once backward. This is done to remove phase shifts that occur when applying a filter. The result is a signal that is filtered, but has the same phase as the original signal.

The butterworth filter is simple to implement. It works well for most cases.

### Gaussian filter

A Gaussian filter is a low-pass filter that is often used in image processing. It is a simple filter to implement. The code below shows how to implement a Gaussian filter.

``` py linenums="1"
from scipy.ndimage import gaussian_filter1d

df["x_p1_filt_gaussian"] = gaussian_filter1d(df["x_p1"], 1) #(1)!
df["x_p2_filt_gaussian"] = gaussian_filter1d(df["x_p2"], 1)
df["x_p3_filt_gaussian"] = gaussian_filter1d(df["x_p3"], 1)
```

1.  The `gaussian_filter1d` takes as its first argument the signal to filter, and as its second argument the standard deviation of the Gaussian filter. The larger the standard deviation, the more smoothing will be applied to the signal. By selecting a standard deviation of 1, we're saying the Gaussian kernel has a standard deviation of *1 shaft revolution*. You need to experiment with different values to see what works best for your case.

### Comparing the filters

We show the effect of applying both filters in [Figure 6](#figure_06) below.


<script src="blade_1_x_filtered.js" ></script>
<div>
    <div>
        <canvas id="ch06_blade_1_x_filtered"'></canvas>
    </div>
    <script>
        async function render_chart_blade_1_x_filtered() {
            const ctx = document.getElementById('ch06_blade_1_x_filtered');
            // If this is a mobile device, set the canvas height to 400
            if (window.innerWidth < 500) {
                ctx.height = 400;
            }
            while (typeof Chart == "undefined") {
                await new Promise(r => setTimeout(r, 1000));
            }
            Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
            window.fig_blade_1_x_filtered = new Chart(ctx, window.blade_1_x_filtered);
            window.fig_blade_1_x_filtered_reset = function resetZoomFig6() {
                    window.fig_blade_1_x_filtered.resetZoom();
                }
            }
        render_chart_blade_1_x_filtered();
    </script>
    <a onclick="window.fig_blade_1_x_filtered_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_06'>Figure 6</a></strong>: The filtered tip deflections for the first blade arriving at the first probe. If you zoom into the signal, you'll see that both the Butterworth and Gaussian filters have a similar effect. The Gaussian filter leaves slightly more noise in the signal, but it is not significant. Both leave the signal significantly smoother than the raw signal.
  </figcaption>
</figure>

From [Figure 6](#figure_06) above, we see that the Gaussian filter and the Butterworth filter performs well. The Gaussian filter leaves a bit more high frequency noise, but that should not bother us.

!!! tip
    It is not mandatory to remove the noise in the signal for frequency analysis. Some algorithms, such as the ones we'll cover in subsequent chapters, work better with the noise removed. There are, however, algorithms that can take the noise into account. It is therefore a good idea to keep the raw signal around for later analysis if need be.

## Peak to Peak vibration

A well established concept in vibration analysis is the peak to peak vibration. It is simply the difference between the maximum and minimum values of a signal within a certain timeframe. The peak to peak vibration gives us a nice tool to combine the tip deflections from each probe into a single value. 

We can calculate the peak to peak vibration for each blade by subtracting the minimum value from the maximum value inside each revolution. 

We can achieve this using the code below:

``` py linenums="1"
x_matrix = (
    df[["x_p1_filt_butter", "x_p2_filt_butter", "x_p3_filt_butter"]]
    .to_numpy()
)
df["pk-pk"] = x_matrix.max(axis=1) - x_matrix.min(axis=1)
```

The peak to peak vibration for the first blade is shown in [Figure 7](#figure_07) below.

<script src="blade_1_x_pk_pk.js" ></script>
<div>
    <div>
        <canvas id="ch06_blade_1_x_pk_pk"'></canvas>
    </div>
    <script>
        async function render_chart_blade_1_x_pk_pk() {
            const ctx = document.getElementById('ch06_blade_1_x_pk_pk');
            // If this is a mobile device, set the canvas height to 400
            if (window.innerWidth < 500) {
                ctx.height = 400;
            }
            while (typeof Chart == "undefined") {
                await new Promise(r => setTimeout(r, 1000));
            }
            Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
            window.fig_blade_1_x_pk_pk = new Chart(ctx, window.blade_1_x_pk_pk);
            window.fig_blade_1_x_pk_pk_reset = function resetZoomFig7() {
                    window.fig_blade_1_x_pk_pk.resetZoom();
                }
            }
        render_chart_blade_1_x_pk_pk();
    </script>
    <a onclick="window.fig_blade_1_x_pk_pk_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_07'>Figure 7</a></strong>: The peak to peak vibration for the first blade. We now have a single indicator representing the vibration from all the proximity probes for a blade. It is much easier to identify resonance zones using this indicator than using multiple probe signals.
  </figcaption>
</figure>

In [Figure 7](#figure_07) above, we see that the peak to peak vibration is an indicator signal representing the maximum difference between the tip deflections occurring within every revolution. It is simpler to identify resonance events using this signal than using multiple probe signals.

In most of our plots, we've also included the shaft speed on a secondary y-axis. You may have noticed there is a symmetry to the resonances in [Figure 7](#figure_07) above. This symmetry occurs because we measured on a run-up and run-down, and the vibration is synchronous. We're going to delve into synchronous vibration in the next chapter. For now, you only have to understand that a blade will experience resonance at a particular shaft speed. Whether we traverse this shaft speed on the run-up or run-down is irrelevant. 

We can take this concept further by creating a scatterplot of the peak to peak values vs the shaft speed. In doing this, we lose information about whether the shaft is running up or down, but it becomes easier to identify resonances. The scatterplot is shown in [Figure 8](#figure_08) below.

<script src="blade_1_x_pk_pk_vs_shaft_speed.js" ></script>
<div>
    <div>
        <canvas id="ch06_blade_1_x_pk_pk_vs_shaft_speed"'></canvas>
    </div>
    <script>
        async function render_chart_blade_1_x_pk_pk_vs_shaft_speed() {
            const ctx = document.getElementById('ch06_blade_1_x_pk_pk_vs_shaft_speed');
            // If this is a mobile device, set the canvas height to 400
            if (window.innerWidth < 500) {
                ctx.height = 400;
            }
            while (typeof Chart == "undefined") {
                await new Promise(r => setTimeout(r, 1000));
            }
            Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
            window.fig_blade_1_x_pk_pk_vs_shaft_speed = new Chart(ctx, window.blade_1_x_pk_pk_vs_shaft_speed);
            window.fig_blade_1_x_pk_pk_vs_shaft_speed_reset = function resetZoomFig8() {
                    window.fig_blade_1_x_pk_pk_vs_shaft_speed.resetZoom();
                }
            }
        render_chart_blade_1_x_pk_pk_vs_shaft_speed();
    </script>
    <a onclick="window.fig_blade_1_x_pk_pk_vs_shaft_speed_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_08'>Figure 8</a></strong>: The peak to peak vibration for the first blade vs shaft speed. This makes it easier to identify resonances. We've identified resonances using arrows (as indicated using callout A) at 572, 622, 672, 934, 1078, and 1248 RPM.
  </figcaption>
</figure>

The order domain scatterplot, shown in [Figure 8](#figure_08) above, beautifully reveal resonance events that occur on both the run-up and the run-down. We've indicated the resonance events using arrows in [Figure 8](#figure_08) above. 

!!! tip
    Please note that the peak to peak vibration is *not the same* as the vibration amplitude. It is a quick and dirty way to visually spot resonances. It is not a replacement for a proper frequency domain analysis.

## Conclusion

In this chapter, we went from having raw AoA values for each blade to having tip deflection values. The tip deflection values represent the dynamic vibration of the blade. All the static components have been removed. We've also seen how to use the peak to peak vibration to identify resonance events.

In the next chapter, we're going to delve into the concept of synchronous vibration and sampling.

!!! question "Outcomes"

	:material-checkbox-marked:{ .checkbox-success .heart } Understand that the AoA values are offset by a constant value, and that this offset is unrelated to the deflection of the blade. 

	:material-checkbox-marked:{ .checkbox-success .heart } Understand that the normalized AoA values can be scaled by the rotor radius to obtain the tip deflection.

    :material-checkbox-marked:{ .checkbox-success .heart } Understand what the tip deflections may vary proportionally to the shaft speed, and that this is not due to tip deflection.

	:material-checkbox-marked:{ .checkbox-success .heart } Understand that we can use a low pass filter to get rid of noise. 
	
	:material-checkbox-marked:{ .checkbox-success .heart } Understand that we can use a peak to peak vibration to identify resonance events.

	:material-checkbox-blank-outline: Write a single function that scales, zeroes and filters the rotor blade AoA values 👇.

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
            2023-10-26
        </p>
    </div>
</div>

## Coding exercises

### 1. AoA to Tip Deflection algorithm

We've performed several steps in this chapter to zero and filter a signal. We want to combine these steps into a single function that scales, zeroes and filters a signal. 


{==

:material-pencil-plus-outline: Write a function that's called `get_blade_tip_deflections_from_AoAs` that receives the following arguments:

1.  A DataFrame containing the raw AoA values of a rotor blade. This is a DataFrame from the output of the `get_rotor_blade_AoAs` function.
2.  A polynomial order to use for detrending.
3.  A cutoff frequency to use for the butterworth filter.

and returns a new DataFrame with the zeroed and filtered tip deflections. Also include the peak to peak vibration levels.

==}

??? example "Reveal answer (Please try it yourself before peeking)"
    ``` py linenums="1"
    def get_blade_tip_deflections_from_AoAs(
        df_rotor_blade_AoAs : pd.DataFrame,
        blade_radius : float,
        poly_order : int = 11,
        filter_order : int = 2,
        filter_cutoff : float = 0.3
    ) -> pd.DataFrame:
        """This function performs the following operations:
            1. Normalizes the AoAs of each probe.
            2. Scales the AoAs to tip deflections.
            3. Detrends the tip deflections using a polynomial fit.
            4. Filters the tip deflections using a Butterworth filter.
            5. Calculates the peak to peak tip deflection.

        Args:
            df_rotor_blade_AoAs (pd.DataFrame): The DataFrame containing the 
                AoAs of each probe. This is an item from the list 
                returned by the function `get_rotor_blade_AoAs`.
            blade_radius (float): The radius of the blade in microns.
            poly_order (int, optional): The polynomial order to use for the detrending
                algorithm . Defaults to 11.
            filter_order (int, optional): The order of the butterworth filter. Defaults to 2.
            filter_cutoff (float, optional): The butterworth filter cutoff 
                frequency. Defaults to 0.3.

        Returns:
            pd.DataFrame: The DataFrame containing the detrended and filtered tip deflections.
        """
        df = df_rotor_blade_AoAs.copy(deep=True)
        all_aoa_columns = [
            col_name 
            for col_name 
            in df.columns 
            if col_name.startswith("AoA_p")
        ]
        butterworth_filter = butter(N=filter_order, Wn=filter_cutoff)
        for col in all_aoa_columns:
            df[col + "_norm"] = df[col].mean() - df[col]
            deflection_col_name = col.replace("AoA", "x")
            df[deflection_col_name] = blade_radius * df[col + "_norm"]
            poly = np.polyfit(df['Omega'], df[deflection_col_name], poly_order)
            df[deflection_col_name] = (
                df[deflection_col_name] 
                - np.polyval(poly, df['Omega'])
            )
            df[deflection_col_name + '_filt'] = filtfilt(
                *butterworth_filter,
                df[deflection_col_name]
            )
        x_matrix = df[[col for col in df.columns if col.endswith("_filt")]].to_numpy()
        df["pk-pk"] = x_matrix.max(axis=1) - x_matrix.min(axis=1)
        return df
    ```

    Usage example:

    ``` py linenums="1"
    >>> df_filtered = get_blade_tip_deflections_from_AoAs(
        df,
        R=164000,
        poly_order=10
    )
    ```

