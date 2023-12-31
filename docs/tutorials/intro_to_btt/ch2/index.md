---
date: 2023-09-18
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
description: This chapter explains how to extract a Time of Arrival (ToA) from proximity probe signals.
robots: index, follow, Blade Tip Timing, BTT, Non Intrusive Stress Measurement, NSMS, Time of Arrival, Turbine blade,Mechanical Vibration
template: main_intro_to_btt.html
card_title: Intro to BTT Ch2 - Time of Arrival triggering criteria
card_url: "ch2/"
---

# Time of Arrival (ToA)

In the previous section, we stated that BTT is based on the measurement of time. Specifically, we want to measure the *exact instant* a blade passed a probe. This instant is called the __Time of Arrival__ (ToA). The ToA is determined by applying a *triggering criterion* to the pulse. Several different triggering criteria have been proposed [@diamond2021constant]. The most common ones are:

1. The threshold triggering method
2. The constant-fraction crossing method
3. The maximum voltage method
4. The maximum slope method

We'll focus on the *threshold triggering method* because of its simplicity and widespread adoption.

## Threshold Triggering Method
This triggering method works by comparing a probe's signal to a predefined voltage threshold. The instant the threshold has been crossed, the blade is said to have "arrived". Each arrival event is stored for subsequent processing. 

An example of a pulse generated by a blade traveling past a probe is shown in [Figure 1](#figure_01). A triggering threshold of 0.4 V has been set. The moment the voltage signal crosses this level, at approximately 40 $\mu s$, the ToA is registered. 
<script src="proximity_probe_data.js" > </script>
<div>
	<div>
		<canvas id="ch02_prox_probe_signal"'></canvas>
	</div>
	<script>
		async function render_chart_fig1() {

			const ctx = document.getElementById('ch02_prox_probe_signal');
			// If this is a mobile device, set the canvas height to 400
			if (window.innerWidth < 500) {
				ctx.height = 400;
			}
			while (typeof Chart == "undefined") {
				await new Promise(r => setTimeout(r, 1000));
				console.log("CHECKED FOR CHART")
			}
			Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
			window.fig1 = new Chart(ctx, window.proximity_probe_data);
			}
		
		render_chart_fig1();
	</script>
</div>

<figure markdown>
  <figcaption><strong><a name='figure_01'>Figure 1</a></strong>: A pulse generated by a blade traveling past a proximity probe. When the pulse rises above the constant threshold, a ToA is stored.</figcaption>
</figure>

Before we get into the code, let's get you up and running with Python and this tutorial's supplementary material.

## Following along using the worksheet
This BTT tutorial is code-centric. I try to explain everything using reproducible code examples. These examples are made available in several worksheets, which can be found at this Github repo:

<a href=https://github.com/Bladesight/bladesight-worksheets target="_blank">
	https://github.com/Bladesight/bladesight-worksheets
</a>

There are two ways in which you can follow along with these worksheets:

1.  Usinge Google Colab, and 
2.  Using a local installation

### Google Colab
Google Colab is an excellent platform that you can use to follow along with these tutorials in the cloud. You can open the Google Colab notebook for this chapter by clicking on the button below: 

<a href="https://colab.research.google.com/github/Bladesight/bladesight-worksheets/blob/master/intro_to_btt/ch_02_worksheet.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

You need a Google account to use Colab. If you don't have one, you can create one for free <a href="https://accounts.google.com/signup/v2/createaccount?theme=glif&flowName=GlifWebSignIn&flowEntry=SignUp" target="_blank">here</a>.

### Local Python Installation
At some stage you'll want to set up your local environment to run the notebooks and do your own development. Here is an excellent (and entertaining) <a href="https://youtu.be/YKSpANU8jPE?feature=shared" target="_blank">:material-open-in-new: video</a> on how to install Python for Windows. 

#### Virtual environments
Virtual environments are an excellent way to isolate different Python projects from one another. I highly recommend setting one up. Here is an excellent <a href="https://youtu.be/kz4gbWNO1cw?si=uFBotRmhP1oTp8K8" target="_blank">:material-open-in-new: video</a> on how to set up virtual environments. 

!!! tip
	If you are having difficulty with installing a virtual environment or the bladesight package, I'd be more than willing to meet with you to help you get set up. You can email me at <a href="mailto:dawie.diamond@bladesight.com">dawie.diamond@bladesight.com</a> to set up a meeting.

The video covers different kinds of operating systems, I've created links for each operating system below:

* <a href="https://youtu.be/kz4gbWNO1cw?feature=shared&t=371" target="_blank">:material-microsoft-windows: Windows</a>
* <a href="https://youtu.be/kz4gbWNO1cw?feature=shared&t=45" target="_blank"> :material-apple: Mac</a>
* <a href="https://youtu.be/kz4gbWNO1cw?feature=shared&t=243" target="_blank">:fontawesome-brands-linux: Linux</a>

### :simple-pypi: Bladesight package
I have released a  `pip` installable package, called `bladesight`, that you can use to access the datasets and functions developed during this tutorial. You can install bladesight using the following console command:

``` console
pip install bladesight
```

Let's get going.

## Vectorized implementation
We consider a generated signal containing three pulses to develop our first implementation. The signal is shown in [Figure 2](#figure_02) below.

<script src="three_pulses.js" > </script>
<div>
	<div>
		<canvas id="ch02_three_pulses_canvas"'></canvas>
	</div>
	<script>
		async function render_chart() {

			const ctx = document.getElementById('ch02_three_pulses_canvas');
			// If this is a mobile device, set the canvas height to 400
			if (window.innerWidth < 500) {
				ctx.height = 400;
			}
			while (typeof Chart == "undefined") {
				await new Promise(r => setTimeout(r, 1000));
				console.log("CHECKED FOR CHART")
			}
			Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
			new Chart(ctx, window.three_pulses);
		}
		render_chart();
	</script>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_02'>Figure 2</a></strong>: A signal containing three pulses. </figcaption>
</figure>

!!! note "How to go through the code"
	These are the first code examples. In it, we __*repeatedly display*__ a complete implementation of the example. Each time we display the code, we highlight different lines and explain them in detail.


### Step 1: Load the probe signal
``` py linenums="1" hl_lines="1 3 5"
from bladesight import Datasets

ds_ch2 = Datasets["data/intro_to_btt/intro_to_btt_ch02"]

df_proximity_probe = ds_ch2["table/three_generated_pulses"] # (1)!

TRIGGER_ON_RISING_EDGE = True
THRESHOLD_LEVEL = 0.4 # Volts

if TRIGGER_ON_RISING_EDGE:
    sr_threshold_over = (
		df_proximity_probe['data'] >= THRESHOLD_LEVEL
	).astype(int)
else:
    sr_threshold_over = (
		df_proximity_probe['data'] <= THRESHOLD_LEVEL
	).astype(int)

diff_sr_threshold = sr_threshold_over.diff()

diff_sr_threshold = diff_sr_threshold.bfill()

sr_threshold_change = diff_sr_threshold > 0

sr_toas = df_proximity_probe['time'][sr_threshold_change]
```

1.	This line loads the table into memory. It returns a `Pandas DataFrame`. We will be making extensive use of Pandas DataFrames throughout this tutorial. The documentation for using it can be found here: <a target="_blank" href="https://pandas.pydata.org/docs/" >https://pandas.pydata.org/docs/</a>


First, on Line 1, we import the `bladesight` package. This package allows you to download the data used for this tutorial. It also contains some functions that we develop in this tutorial.

On lines 3 and 5, we download the dataset corresponding to this chapter and load the "three_generated_pulses" table. The first ten rows of the dataset is shown below:

{{ read_csv('docs/tutorials/intro_to_btt/ch2/three_pulses_head.csv') }}

The DataFrame has 2 columns: `time` and `data`. The `time` column contains the timestamp corresponding to each data value. 

### Step 2: Set the threshold direction and value

``` py linenums="1" hl_lines="7 8"
from bladesight import Datasets

ds_ch2 = Datasets["data/intro_to_btt/intro_to_btt_ch02"]

df_proximity_probe = ds_ch2["table/three_generated_pulses"]

TRIGGER_ON_RISING_EDGE = True
THRESHOLD_LEVEL = 0.4 # Volts

if TRIGGER_ON_RISING_EDGE:
    sr_threshold_over = (
		df_proximity_probe['data'] >= THRESHOLD_LEVEL
	).astype(int)
else:
    sr_threshold_over = (
		df_proximity_probe['data'] <= THRESHOLD_LEVEL
	).astype(int)

diff_sr_threshold = sr_threshold_over.diff()

diff_sr_threshold = diff_sr_threshold.bfill()

sr_threshold_change = diff_sr_threshold > 0

sr_toas = df_proximity_probe['time'][sr_threshold_change]
```

In Line 7, we specify the direction of the trigger. If `TRIGGER_ON_RISING_EDGE` is `True`, then we trigger when the signal crosses the threshold on the rising edge. If `TRIGGER_ON_RISING_EDGE` is `False`, then we trigger when the signal crosses the threshold on the falling edge.

In Line 8 we set the threshold level. In this example, we set the threshold level to 0.4 Volts.

### Step 3: Determine when the signal has crossed the threshold

``` py linenums="1" hl_lines="10 11 12 13 14 15 16 17"
from bladesight import Datasets

ds_ch2 = Datasets["data/intro_to_btt/intro_to_btt_ch02"]

df_proximity_probe = ds_ch2["table/three_generated_pulses"]

TRIGGER_ON_RISING_EDGE = True
THRESHOLD_LEVEL = 0.4 # Volts

if TRIGGER_ON_RISING_EDGE:
    sr_threshold_over = (
		df_proximity_probe['data'] >= THRESHOLD_LEVEL
	).astype(int) # (1)!
else:
    sr_threshold_over = (
		df_proximity_probe['data'] <= THRESHOLD_LEVEL
	).astype(int)

diff_sr_threshold = sr_threshold_over.diff()

diff_sr_threshold = diff_sr_threshold.bfill()

sr_threshold_change = diff_sr_threshold > 0

sr_toas = df_proximity_probe['time'][sr_threshold_change]
```

1.	We use the method `.astype(int)` at the end of this line because, by default, comparison operators such as `>=` and `<=` result in boolean values. We, however, need an equivalent integer representation for subsequent steps.

In lines 10 - 17, we determine when the signal is "over" the threshold level. The definition of "over" depends on the direction of the trigger. If we are triggering on a rising edge, then the signal is "over" the threshold level when the signal is greater than or equal to the threshold level. If we are triggering on a falling edge, then the signal is "over" the threshold level when the signal is less than or equal to the threshold level.

The variable `sr_threshold_over` contains an array of ones and zeros indicating whether the signal is above or below the threshold. This variable is shown on top of the original signal in [Figure 3](#figure_03) below:

<script src="over_under_indicator.js" > </script>
<div>
	<div>
		<canvas id="ch03_three_pulses_canvas"'></canvas>
	</div>
	<script>
		async function render_chart() {

			const ctx = document.getElementById('ch03_three_pulses_canvas');
			// If this is a mobile device, set the canvas height to 400
			if (window.innerWidth < 500) {
				ctx.height = 400;
			}
			while (typeof Chart == "undefined") {
				await new Promise(r => setTimeout(r, 1000));
				console.log("CHECKED FOR CHART")
			}
			Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
			new Chart(ctx, window.over_under_indicator);
		}
		render_chart();
	</script>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_03'>Figure 3</a></strong>: The original signal with the over/under indicator plotted over the signal. We can see that the indicator goes to 1 (boolean True) when the signal is larger than the threshold and stays 0, when the signal is smaller than the signal. </figcaption>
</figure>


### Step 4: Determine when the threshold has changed

``` py linenums="1" hl_lines="19 21 23"
from bladesight import Datasets

ds_ch2 = Datasets["data/intro_to_btt/intro_to_btt_ch02"]

df_proximity_probe = ds_ch2["table/three_generated_pulses"]

TRIGGER_ON_RISING_EDGE = True
THRESHOLD_LEVEL = 0.4 # Volts

if TRIGGER_ON_RISING_EDGE:
    sr_threshold_over = (
		df_proximity_probe['data'] >= THRESHOLD_LEVEL
	).astype(int)
else:
    sr_threshold_over = (
		df_proximity_probe['data'] <= THRESHOLD_LEVEL
	).astype(int)

diff_sr_threshold = sr_threshold_over.diff()

diff_sr_threshold = diff_sr_threshold.bfill() # (1)!

sr_threshold_change = diff_sr_threshold > 0

sr_toas = df_proximity_probe['time'][sr_threshold_change]
```

1.	The `bfill` method is used to backwards fill any missing values in a pandas Series. Because we are calculating the difference between consecutive measurements, the first value in the `sr_threshold_change` will be empty, or  `NaN` (because there is no prior value before the first value). We use `bfill` here to set this value to the second value. 

Since we are interested in when the signal has crossed the threshold, we need to calculate the *change* in `sr_threshold_over`. In Line 19, the `diff` method is used to calculate the consecutive differences between adjacent measurements. In Line 23, the `>` operator is used to get the indices indicating when the thresholds have changed. The result of this operation is shown in [Figure 4](#figure_04) below:


<script src="change_in_over_under_indicator.js" > </script>
<div>
	<div>
		<canvas id="ch04_change_in_over_under_indicator"'></canvas>
	</div>
	<script>
		async function render_chart() {

			const ctx = document.getElementById('ch04_change_in_over_under_indicator');
			// If this is a mobile device, set the canvas height to 400
			if (window.innerWidth < 500) {
				ctx.height = 400;
			}
			while (typeof Chart == "undefined") {
				await new Promise(r => setTimeout(r, 1000));
				console.log("CHECKED FOR CHART")
			}
			Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
			new Chart(ctx, window.change_in_over_under_indicator);
		}
		render_chart();
	</script>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_04'>Figure 4</a></strong>: The original signal, along with the change in over/under indicator shown. We can see the change indicator is exactly 1 when the instant the threshold is crossed, and -1 when the signal falls below the threshold again 
  </figcaption>
</figure>

### Step 5: Select the ToAs
``` py linenums="1" hl_lines="25"
from bladesight import Datasets

ds_ch2 = Datasets["data/intro_to_btt/intro_to_btt_ch02"]

df_proximity_probe = ds_ch2["table/three_generated_pulses"]

TRIGGER_ON_RISING_EDGE = True
THRESHOLD_LEVEL = 0.4 # Volts

if TRIGGER_ON_RISING_EDGE:
    sr_threshold_over = (
		df_proximity_probe['data'] >= THRESHOLD_LEVEL
	).astype(int)
else:
    sr_threshold_over = (
		df_proximity_probe['data'] <= THRESHOLD_LEVEL
	).astype(int)

diff_sr_threshold = sr_threshold_over.diff()

diff_sr_threshold = diff_sr_threshold.bfill() # (1)!

sr_threshold_change = diff_sr_threshold > 0

sr_toas = df_proximity_probe['time'][sr_threshold_change]
```

In Line 25, we select the `time` values corresponding to the time instants a ToA shifted from under the threshold to over the threshold. The `sr_toas` has the following values:

``` py
>>> print(sr_toas)
[27.44248083, 52.45081694, 77.45915305]
```

## Sequential Implementation
We've just wrapped up an implementation that performs operations on the entire signal as a whole. Sometimes, however, it's easier to reason about sequentially processing the signal. Here, we'll create a ToA extraction algorithm. This approach allows us to work with the signal in a sample-to-sample fashion.

### Installing Numba

Python is typically regarded as a slow language due to being interpreted. There are, however, many methods available to enhance its speed, often achieving performance levels comparable to compiled languages. 

One effective approach is to use Numba. Numba is a powerful Just-In-Time (JIT) compiler for Python that compiles portions of your code into machine code at runtime. This often leads to blistering fast functions. To get started with Numba, you can install it using the following console command:

!!! tip "Installing numba"
	Enter the following code into your console:
	``` console
	pip install numba
	```

To use Numba, we import the `njit` decorator from the package, and wrap our function in it.

``` py 
from numba import njit
```

### Simple example

The simplest implementation of a sequential algorithm involves using a `for` loop to monitor when the signal passes a constant threshold.

``` py linenums="1"
@njit
def seq_simple_threshold_crossing(
    arr_t : np.ndarray, #(1)!
    arr_s : np.ndarray, #(2)!
    threshold : float, #(3)!
    n_est : Optional[float] = None,#(4)!
    trigger_on_rising_edge : bool = True#(5)!
) -> np.ndarray: #(6)!   
    if n_est is None:
        arr_toa = -1 * np.ones(arr_t.shape)
    else:
        arr_toa = -1 * np.ones(n_est) #(7)!

    i_toa = 0 #(8)!

    prev_sample = arr_s[0] #(9)!

    for i_sample in range(1, arr_s.shape[0]): #(10)!
 
        curr_sample = arr_s[i_sample]

        if trigger_on_rising_edge:
            if (prev_sample < threshold) and (curr_sample >= threshold): #(11)!
                arr_toa[i_toa] = arr_t[i_sample] #(12)!
                i_toa += 1#(13)!
        else:
            if (prev_sample > threshold) and (curr_sample <= threshold): #(14)!
                arr_toa[i_toa] = arr_t[i_sample]
                i_toa += 1

        prev_sample = curr_sample #(15)!

    return arr_toa[:i_toa] #(16)!
```

1.	The array containing the time values. Must be a Numpy array.
2.	The array containing the voltage values. Must be a Numpy array.
3.	The threshold value. Must be a float.
4.	The estimated number of ToAs in this signal. Defaults to None. This number is used to pre-allocate the array containing the ToAs. If this number is not provided, the array will be pre-allocated as the same dimension as arr_t and arr_s. You should specify this value for large signals.
5.	Whether to trigger ToAs on the rising or falling edge. Defaults to True. If True, the ToA is triggered on the rising edge.
6.	This type annotation indicates the expected return type of the function. In this case, its a Numpy array containing the identified ToAs.
7.	We pre-allocate the array containing the ToAs. This is a performance optimization. If we don't pre-allocate the array, the function will have to resize the array each time a ToA is found. Its easy to estimate the expected number of ToAs in the signal, and should be done for any real signal.
8.	Initialize the number of ToAs found in this signal. `i_toa` will increase by one each time a ToA has been found.
9.	The sequential approach works by comparing each sample to the previous sample. Here, we initialize the `prev_sample` value to the first value in the array.
10.	We loop through all remaining samples and perform a comparison on each sample.
11.	Here, we check if the threshold has been crossed. If the threshold is crossed, we store the ToA in the `arr_toa` array. This is the check for a rising edge trigger.
12.	If the threshold has been crossed, we store the ToA in the `arr_toa` array.
13. Increment the `i_toa` value by one. This indicates that we've found a ToA.
14.	This is the check for a falling edge.
15. We're done with this sample. Now we're preparing for the next loop pass.
16.	Only return the ToAs that have been found. The rest of the array is filled with -1 values.


!!! Note "Type annotations in Python"
	Throughout this tutorial, you'll notice that in many of the functions, the function arguments are accompanied by *type annotations*. In the first argument above, `arr_t : np.ndarray`, `arr_t` is the variable name and `: np.ndarray` is the type annotation . These type annotations specify the expected data types of the arguments. A type annotation is essentially a way to indicate what kind of input a function is designed to work with. 
    
    It's important to note that Python itself does not enforce these type annotations; instead, they are primarily utilized by type checkers and serve as a helpful guide for users of the function.

	We often need to import annotations from the `typing` library. In the example above, we need to import the `Optional` type using `from typing import Optional`.

If we pass our three pulses signal through this function, we get the exact same result:
``` py
>>> toas = seq_simple_threshold_crossing(
	df_proximity_probe['time'].values, #(1)!
	df_proximity_probe['data'].values, 
	0.4
)
>>> print(toas)
[27.44248083 52.45081694 77.45915305]
```

1.	Note how we use the `.values` attribute of the Pandas object. This returns a Numpy array, which is required by the function.

### Interpolate on voltage
When we have the advantage of an entire analog signal stored on disk, it's wise to capitalize on the continuous nature of the signal. It's unlikely that the threshold will align precisely with the exact moment a sample is acquired by the data acquisition system. In most cases, the acquisition system will record a sample *after* the threshold has already been crossed. 

To improve the accuracy of ToA determination, we can employ interpolation techniques between the two nearest samples when the threshold is crossed.

Below is a function that performs this interpolation.

``` py linenums="1"
@njit
def seq_threshold_crossing_interp(
    arr_t : np.ndarray,
    arr_s : np.ndarray,
    threshold : float,
    n_est : Optional[float] = None,
    trigger_on_rising_edge : bool = True
) -> np.ndarray:
    if n_est is None:
        arr_toa = -1 * np.ones(arr_t.shape)
    else:
        arr_toa = -1 * np.ones(n_est)

    i_toa = 0

    prev_sample = arr_s[0]

    for i_sample in range(1, arr_s.shape[0]):

        curr_sample = arr_s[i_sample]

        if trigger_on_rising_edge:
            if (prev_sample < threshold) and (curr_sample >= threshold):
                # Interpolate the ToA
                arr_toa[i_toa] = (
                    arr_t[i_sample - 1] 
                    + (arr_t[i_sample] - arr_t[i_sample - 1]) 
                    * (threshold - prev_sample) 
                    / (curr_sample - prev_sample)
                ) # (1)!
                i_toa += 1
        else:
            if (prev_sample > threshold) and (curr_sample <= threshold):
                arr_toa[i_toa] = (
                    arr_t[i_sample - 1] 
                    + (arr_t[i_sample] - arr_t[i_sample - 1]) 
                    * (threshold - prev_sample) 
                    / (curr_sample - prev_sample)
                )
                i_toa += 1

        prev_sample = curr_sample

    return arr_toa[:i_toa]
```

1.	This part performs linear interpolation between the two samples either side of the threshold. It is the only difference between this function and the previous one.

Let's see how this function performs on the same signal:

``` py
>>> toas = seq_threshold_crossing_interp(
    df_proximity_probe['time'].values, 
    df_proximity_probe['data'].values, 
    0.4
)
>>> print(toas)
[27.42940329 52.44126697 77.45548277]
```

We see these values are slightly different than the previous ones ( the previous ones are `[27.44248083 52.45081694 77.45915305]` ). The new values are more accurate than the old ones.

### Hysteresis
Up to this point, the signals we've examined have exhibited minimal noise, making ToA determination straightforward. However, in real-world scenarios, signals are seldom devoid of noise. 

Sometimes, excessive noise is consistently present in the signal, while at other times, it appears in short, sporadic bursts. Noise complicates ToA extraction.

To illustrate this point vividly, we have generated a noisy signal consisting of three pulses, shown in [Figure 5](#figure_05) below: 

!!! Note "Zoom"
    If there is a "Reset zoom" button on the bottom of the figure, you can zoom and pan. __zoom__ by dragging across the screen, __pan__ by holding `ctrl` and dragging across the screen.
<script src="three_pulses_noisy.js" > </script>
<div>
	<div>
		<canvas id="ch02_three_pulses_noisy"'></canvas>
	</div>
	<script>
		async function render_chart_three_pulses_noisy() {
			const ctx = document.getElementById('ch02_three_pulses_noisy');
			// If this is a mobile device, set the canvas height to 400
			if (window.innerWidth < 500) {
				ctx.height = 400;
			}
			while (typeof Chart == "undefined") {
				await new Promise(r => setTimeout(r, 1000));
				console.log("CHECKED FOR CHART")
			}
			Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
			window.fig_three_pulses_noisy = new Chart(ctx, window.three_pulses_noisy);
			window.fig_three_pulses_noisy_reset = function resetZoomFig1() {
					window.fig_three_pulses_noisy.resetZoom();
				}
			}
		render_chart_three_pulses_noisy();
	</script>
	<a onclick="window.fig_three_pulses_noisy_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_05'>Figure 5</a></strong>: A noisy signal containing three pulses. Zoom into the areas around 0.4 V to see the signal crosses the threshold multiple times.
  </figcaption>
</figure>
If you zoom into each pulse around 0.4 V, you'll see the signal crosses the threshold many times. If we were to apply our algorithm to this noisy signal, the resulting ToAs are as shown below:. 

``` console
[27.20280579, 27.26724046, 27.45521975, 27.54516822, 32.4110573,  32.48523812,
 32.59630977, 32.73320484, 52.10882236, 52.46204252, 52.59380688, 57.42079388,
 57.64309888, 57.69638523, 57.78086048, 57.87833845, 77.46723661, 77.66015932,
 78.09842428, 82.3623517,  82.50989437, 82.74746473]
```

We see that our algorithm returns many ToAs around the 0.4 V level for each pulse. This outcome is clearly incorrect.

We can build in *hysteresis* to solve the issue. Hysteresis introduces a *second threshold* and a *state variable* to our algorithm. The algorithm therefore uses a *lower* and an *upper* threshold. The state can only change based on certain rules. The rules are:

1.	If the state is *low* and the current sample is *above* the upper threshold, the state becomes high.
2.	If the state is *high* and the current sample falls *below* the lower threshold, the state becomes low.
3.	Otherwise, maintain the state.

Here is an example of a ToA extraction algorithm that uses hysteresis to trigger on the rising edge.

``` py linenums="1"
@njit
def seq_threshold_crossing_hysteresis_pos(
    arr_t : np.ndarray,
    arr_s : np.ndarray,
    threshold : float,
    hysteresis_height : float, #(1)!
    n_est : Optional[float] = None,
) -> np.ndarray:
    threshold_lower = threshold - hysteresis_height
    trigger_state = ( # (2)!
		True 
		if arr_s[0] > threshold_lower 
		else False
	)

    if n_est is None:
        arr_toa = -1 * np.ones(arr_t.shape)
    else:
        arr_toa = -1 * np.ones(n_est)

    i_toa = 0

    prev_sample = arr_s[0]

    for i_sample in range(1, arr_s.shape[0]):
        curr_sample = arr_s[i_sample]

        if trigger_state is True:
            if curr_sample <= threshold_lower: #(3)!
                trigger_state = False
        else:
            if curr_sample >= threshold:#(4)!
                trigger_state = True
                arr_toa[i_toa] = (
                    arr_t[i_sample - 1] 
                    + (arr_t[i_sample] - arr_t[i_sample - 1]) 
                    * (threshold - prev_sample) 
                    / (curr_sample - prev_sample)
                )
                i_toa += 1

        prev_sample = curr_sample

    return arr_toa[:i_toa]
```

1.  The height of the hysteresis. It has the same units as the signal. This value is used to calculate the lower threshold.
2.  The trigger state is a boolean value indicating whether the trigger is currently *high* or *low*. We initialize it to True if the first sample is above the lower threshold, and False otherwise.
3.  If the trigger state is True, we check if the current sample is below the lower threshold. If it is, we set the trigger state to False.
4.  If the trigger state is False, we check if the current sample is above the upper threshold. If it is, we set the trigger state to True and calculate the ToA using interpolation.

We can use this function to extract the ToAs from the noisy signal:

``` py
>>> toas = seq_threshold_crossing_hysteresis_pos(
	df_proximity_probe_noisy['time'].values, 
	df_proximity_probe_noisy['data'].values, 
	0.4, 
	0.2
)
>>> print(taos)
[27.20280579 52.10882236 77.46723661]
```
 Building in hysteresis stops multiple triggering, but it doesn't guarantee accurate ToAs for noisy signals. We'll always end up with the *correct number* of ToAs, but each ToA will be the first one where the signal crosses the threshold. This may not be the correct ToA. You may therefore need to include filters into the triggering criterion.

## Performance of Sequential vs Vectorized Implementations
We've now seen that you can use *vectorized* or *sequential* implementations to determine the ToA of a signal. The question is, which one is faster? Let's use both approaches on a real signal.

The dataset we've been using contains a signal acquired from a proximity probe, `table/aluminium_blisk_1200_rpm`. The signal was acquired for a five blade rotor running at 1200 RPM. The acquisition was performed at a sampling rate of 2 MHz. The signal is shown in [Figure 6](#figure_06) below:

<script src="eddy_current_probe_raw.js" > </script>
<div>
	<div>
		<canvas id="ch02_eddy_current_probe_raw"'></canvas>
	</div>
	<script>
		async function render_chart_fig_eddy_current_raw() {
			const ctx = document.getElementById('ch02_eddy_current_probe_raw');
			// If this is a mobile device, set the canvas height to 400
			if (window.innerWidth < 500) {
				ctx.height = 400;
			}
			while (typeof Chart == "undefined") {
				await new Promise(r => setTimeout(r, 1000));
				console.log("CHECKED FOR CHART")
			}
			Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
			window.fig_eddy_current_raw = new Chart(ctx, window.eddy_current_probe_raw);
			window.fig_eddy_current_raw_reset = function resetZoomFig2() {
					window.fig_eddy_current_raw.resetZoom();
				}
			}
		render_chart_fig_eddy_current_raw();
	</script>
	<a onclick="window.fig_eddy_current_raw_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_06'>Figure 6</a></strong>: An eddy current probe signal acquired for a five blade aluminium rotor rotating at a rate of 1200 RPM. The sampling rate of the DAQ was 2 MHz. For this figure, the data points are heavily decimated.</figcaption>
</figure>

We can use the Jupyter `%%timeit` to repeat a piece of code several times and determine how fast the code runs. If we do this for both approaches, we get the following results:


| Vectorized | Sequential |
|------------|------------|
| 301 ms     | 18 ms |

The sequential method is, therefore, almost 20 times faster than the vectorized method. The sequential method also contains hysteresis and interpolation, which the vectorized method does not.

I believe the sequential method should always be preferred for ToA extraction.


## Conclusion
We have discussed a method for extracting the ToAs from an analogue signal. This method can be used when you measure your entire analogue signal can be stored to disk. If you're turbine is small and your rotor turns slowly, then this method is suitable.

When your rotor blades spin faster, you need to determine the ToAs in real-time. This typically requires custom hardware. Many different providers of custom hardware exist:

* <a target='_blank' href="https://www.bladesight.com"> :material-open-in-new: Bladesight</a>
* <a target='_blank' href="https://emtd-measurement.com/"> :material-open-in-new: EMTD Measurement</a>
* <a target='_blank' href="https://hoodtech.com/bvm/"> :material-open-in-new: Hood Technologies</a>
* <a target='_blank' href="https://agilis.com/"> :material-open-in-new: Agilis</a>

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
            2023-09-20
        </p>
    </div>
</div>
