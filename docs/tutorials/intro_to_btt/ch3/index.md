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
description: This chapter explains how to convert the Time of Arrival (ToA) values into Angle of Arrival (AoA) values.
robots: index, follow, Blade Tip Timing, BTT, Non Intrusive Stress Measurement, NSMS, Time of Arrival, Turbine blade,Mechanical Vibration
template: main_intro_to_btt.html
card_title: Intro to BTT Ch3 - Angle of Arrival
card_url: "ch3/"
---
# Angle of Arrival (AoA)
We have said that tip displacements must be *inferred* from *time*. We have already learned how to measure *time* in the previous chapter. We achieved this by extracting the ToAs from the proximity probe signals. Next, we need to convert these ToA values into *angular positions*. Each ToA has a corresponding Angle of Arrival (AoA). The AoA is the exact circumferential position of the shaft at the ToA.

If the blades are not vibrating, we expect the AoAs to be constant for each blade at each probe. It is when the AoAs are not constant that we infer the presence of some unknown physical phenomena, waiting to be discovered.

!!! question "Outcomes"

	:material-checkbox-blank-outline: Understand that we use a shaft encoder to calculate the shaft speed, $\Omega$, and the start and end of each revolution. 

    :material-checkbox-blank-outline: Understand that we need to find the shaft revolution in which each ToA occurs.

	:material-checkbox-blank-outline: Understand that each ToA is used to calculate the precise shaft circumferential displacement in said revolution, leading to the AoA.
	
	:material-checkbox-blank-outline: Write a function that calculates a matrix of AoA values from the shaft encoder zero-crossing times and the ToA values.

## Shaft encoder
Most BTT systems use a shaft encoder installed somewhere on the shaft. The shaft encoder produces a pulse train, similar to the proximity probes. Most shaft encoders produce one pulse per revolution, but some produce multiple pulses per revolution. These shaft encoders are referred to as OPR and MPR encoders respectively. This tutorial covers the OPR case.

We extract the ToAs from the OPR signal just like we extract the ToAs from the proximity probe signals. The OPR ToAs are normally referred to as *zero-crossing* times, creating the impression that they are registered when the signal crosses 0 V. Though this is often the case, the zero-crossing times can be extracted using any triggering criteria. Each zero-crossing time therefore corresponds to the start of a new shaft revolution.

Once we have a vector of zero-crossing times. The shaft speed in between zero-crossing times can be calculated using [Equation 1](#equation_01) below.

$$
\Omega_n = \frac{2 \pi}{t_{n} - t_{n-1}}
$$
??? info "Symbols"
	| Symbol | Description |
	| :---: | :--- |
	| $\Omega_n$ | Shaft speed during the $n$th revolution |
	| $n$ | The revolution number |
	| $t_{n}$ | The $n$th zero-crossing time |

<figure markdown>
  <figcaption><strong><a name='equation_01'>Equation 1</a></strong></figcaption>
</figure>

An example of the shaft speed derived from zero-crossing times is shown in [Figure 1](#figure_01) below.
<script src="shaft_run_up_and_down.js" > </script>
<div>
	<div>
		<canvas id="ch03_shaft_run_up_and_down"'></canvas>
	</div>
	<script>
		async function render_chart_shaft_run_up_and_down() {
			const ctx = document.getElementById('ch03_shaft_run_up_and_down');
			// If this is a mobile device, set the canvas height to 400
			if (window.innerWidth < 500) {
				ctx.height = 400;
			}
			while (typeof Chart == "undefined") {
				await new Promise(r => setTimeout(r, 1000));
				console.log("CHECKED FOR CHART")
			}
			Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
			window.fig_shaft_run_up_and_down = new Chart(ctx, window.shaft_run_up_and_down);
			window.fig_shaft_run_up_and_down_reset = function resetZoomFig1() {
					window.fig_shaft_run_up_and_down.resetZoom();
				}
			}
		render_chart_shaft_run_up_and_down();
	</script>
	<a onclick="window.fig_shaft_run_up_and_down_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_01'>Figure 1</a></strong>: A shaft running up from 950 RPM to 1325 RPM and back again over a time of 41 seconds. </figcaption>
</figure>

The figure above shows the run-up and down of a shaft between 950 RPM and 1325 RPM over approximately 41 seconds.

## Angle of Arrival (AoA)
The AoA of each proximity probe ToA is the shaft's circumferential position at the time of the ToA. It is calculated relative to the revolution in which it occurs. In other words, it is always a quantity between 0 and $2 \pi$ within each revolution. We can use [Equation 2](#equation_02) below to calculate the AoA.

$$
\theta_{n} (\textrm{ToA}) =  \Omega_n \times (\textrm{ToA} - t_{n})
$$
??? info "Symbols"
	| Symbol | Description |
	| :---: | :--- |
	| $\theta_{n} (\textrm{ToA})$ | The AoA of the ToA within the $n$th revolution |
	| $n$ | The revolution number |
	| $\textrm{ToA}$ | The ToA of a blade passing underneath a probe |
	| $\Omega_n$ | The shaft speed during the $n$th revolution |
<figure markdown>
  <figcaption><strong><a name='equation_02'>Equation 2</a></strong></figcaption>
</figure>

The task of converting the ToAs into AoAs boils down to:

1. Associating a revolution number to each ToA
2. Calculating the AoA of each ToA within its corresponding revolution

This two-step process can be performed using a single function. We're going to write a function that takes in a vector of ToAs and a vector of zero-crossing times and returns a matrix of values relevant to the AoAs. We will, once again, be using Numba to speed things up.

## Following along
The worksheet for this chapter can be downloaded here <a href="https://github.com/Bladesight/bladesight-worksheets/blob/master/intro_to_btt/ch_03_worksheet.ipynb" target="_blank"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="Open In Github"/></a>.


You can open a Google Colab session of the worksheet by clicking here: <a href="https://colab.research.google.com/github/Bladesight/bladesight-worksheets/blob/master/intro_to_btt/ch_03_worksheet.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>.

You need to use one of the following Python versions to run the worksheet:
<img src="https://img.shields.io/badge/python-3.6-blue.svg">
<img src="https://img.shields.io/badge/python-3.7-blue.svg">
<img src="https://img.shields.io/badge/python-3.8-blue.svg">
<img src="https://img.shields.io/badge/python-3.9-blue.svg">
<img src="https://img.shields.io/badge/python-3.10-blue.svg">
<img src="https://img.shields.io/badge/python-3.11-blue.svg">

## Algorithm

``` py linenums="1"
from numba import njit
import numpy as np

@njit
def calculate_aoa(
    arr_opr_zero_crossing : np.ndarray, 
    arr_probe_toas : np.ndarray
):
    """
    This function calculates the angle of arrival of 
    each ToA value relative to the revolution in 
    which it occurs.

    Args:
        arr_opr_zero_crossing (np.array): An array of 
            OPR zero-crossing times. 
        arr_probe_toas (np.array): An array of 
            ToA values.

    Returns:
        np.array: A matrix of AoA values. Each row in the 
            matrix corresponds to a ToA value. The columns 
            are:
            0: The revolution number
            1: The zero crossing time at the start of the revolution
            2: The zero crossing time at the end of the revolution
            3: The angular velocity of the revolution
			4: The ToA
            5: The AoA of the ToA value
    """
    num_toas = len(arr_probe_toas)
    AoA_matrix = np.zeros( (num_toas, 6))

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
            Omega = 2 * np.pi / (
                current_zero_crossing_end 
                - current_zero_crossing_start
            )
            AoA_matrix[i, 3] = Omega
            AoA_matrix[i, 4] = toa
            AoA_matrix[i, 5] = Omega * (
                toa 
                - current_zero_crossing_start
            )

    return AoA_matrix
```

This function appears intimidating, but we're going to go through it slowly, step by step.

### Imports

``` py linenums="1"
from numba import njit
import numpy as np
# Some code omitted below 👇
```

??? note "Full code"
	``` py hl_lines="1 2" linenums="1" 
	from numba import njit
	import numpy as np

	@njit
	def calculate_aoa(
		arr_opr_zero_crossing : np.ndarray, 
		arr_probe_toas : np.ndarray
	):
		"""
		This function calculates the angle of arrival of 
		each ToA value relative to the revolution in 
		which it occurs.

		Args:
			arr_opr_zero_crossing (np.array): An array of 
				OPR zero-crossing times. 
			arr_probe_toas (np.array): An array of 
				ToA values.

		Returns:
			np.array: A matrix of AoA values. Each row in the 
				matrix corresponds to a ToA value. The columns 
				are:
				0: The revolution number
				1: The zero crossing time at the start of the revolution
				2: The zero crossing time at the end of the revolution
				3: The angular velocity of the revolution
				4: The ToA
				5: The AoA of the ToA value
		"""
		num_toas = len(arr_probe_toas)
		AoA_matrix = np.zeros( (num_toas, 6))

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
				Omega = 2 * np.pi / (
					current_zero_crossing_end 
					- current_zero_crossing_start
				)
				AoA_matrix[i, 3] = Omega
				AoA_matrix[i, 4] = toa
				AoA_matrix[i, 5] = Omega * (
					toa 
					- current_zero_crossing_start
				)

		return AoA_matrix
	```
We have three imports:

1. `njit` from the `numba` library. `njit` is a *decorator* that tells the `numba` compiler to compile the function to machine code. This step speeds `calculate_aoa` up to near C speed. 
2. `numpy`, which we rename using an *alias* to `np`. This means we can access the Numpy library through writing `np`. This is a common convention. Numpy is a numerical computation library. 

### Function definition

``` py linenums="3"
# Some code omitted above 👆
@njit
def calculate_aoa(
	arr_opr_zero_crossing : np.ndarray, 
	arr_probe_toas : np.ndarray
):
# Some code omitted below 👇
```

??? note "Full code"
	``` py hl_lines="4 5 6 7 8" linenums="1" 
	from numba import njit
	import numpy as np

	@njit
	def calculate_aoa(
		arr_opr_zero_crossing : np.ndarray, 
		arr_probe_toas : np.ndarray
	):
		"""
		This function calculates the angle of arrival of 
		each ToA value relative to the revolution in 
		which it occurs.

		Args:
			arr_opr_zero_crossing (np.array): An array of 
				OPR zero-crossing times. 
			arr_probe_toas (np.array): An array of 
				ToA values.

		Returns:
			np.array: A matrix of AoA values. Each row in the 
				matrix corresponds to a ToA value. The columns 
				are:
				0: The revolution number
				1: The zero crossing time at the start of the revolution
				2: The zero crossing time at the end of the revolution
				3: The angular velocity of the revolution
				4: The ToA
				5: The AoA of the ToA value
		"""
		num_toas = len(arr_probe_toas)
		AoA_matrix = np.zeros( (num_toas, 6))

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
				Omega = 2 * np.pi / (
					current_zero_crossing_end 
					- current_zero_crossing_start
				)
				AoA_matrix[i, 3] = Omega
				AoA_matrix[i, 4] = toa
				AoA_matrix[i, 5] = Omega * (
					toa 
					- current_zero_crossing_start
				)

		return AoA_matrix
	```
In Python, you define a function using the `def` keyword. We define a function called `calculate_aoa` on Line 6. This function has two arguments:

1. `arr_opr_zero_crossing`, which is the array containing the OPR zero-crossing times.
2. `arr_probe_toas`, which is the array containing the ToA values from a single probe.

In Line 4 we wrap our function with the `njit` *decorator*. A *decorator* in Python allows you to alter a function before it is run. In this case, we are telling the `numba` compiler to compile the function to machine code, significantly speeding up the function.

### Initialising the AoA matrix and other variables

``` py linenums="30"
# Some code omitted above 👆
num_toas = len(arr_probe_toas)
AoA_matrix = np.zeros( (num_toas, 6) )

AoA_matrix[:, 0] = -1

current_zero_crossing_start = arr_opr_zero_crossing[0]
current_zero_crossing_end = arr_opr_zero_crossing[1]
current_n = 0
# Some code omitted below 👇
```
??? note "Full code"
	``` py hl_lines="31 32 34 36 37 38" linenums="1"
	from numba import njit
	import numpy as np

	@njit
	def calculate_aoa(
		arr_opr_zero_crossing : np.ndarray, 
		arr_probe_toas : np.ndarray
	):
		"""
		This function calculates the angle of arrival of 
		each ToA value relative to the revolution in 
		which it occurs.

		Args:
			arr_opr_zero_crossing (np.array): An array of 
				OPR zero-crossing times. 
			arr_probe_toas (np.array): An array of 
				ToA values.

		Returns:
			np.array: A matrix of AoA values. Each row in the 
				matrix corresponds to a ToA value. The columns 
				are:
				0: The revolution number
				1: The zero crossing time at the start of the revolution
				2: The zero crossing time at the end of the revolution
				3: The angular velocity of the revolution
				4: The ToA
				5: The AoA of the ToA value
		"""
		num_toas = len(arr_probe_toas)
		AoA_matrix = np.zeros( (num_toas, 6))

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
				Omega = 2 * np.pi / (
					current_zero_crossing_end 
					- current_zero_crossing_start
				)
				AoA_matrix[i, 3] = Omega
				AoA_matrix[i, 4] = toa
				AoA_matrix[i, 5] = Omega * (
					toa 
					- current_zero_crossing_start
				)

		return AoA_matrix
	```

In this piece of code we initialize constants we are going to use and variables we are going to populate: 

* In Line 31, we get the number of ToA values in the `arr_probe_toas` array. We are going use this variable to instantiate our output matrix. 
* In Line 32 we instantiate our output matrix, i.e. the variable that will contain the algorithm's results. This matrix is called `AoA_matrix`. The AoA matrix is a (`num_toas` $\times$ 6) matrix. Each column will hold the following kinds of values:

	* __column 1__: The revolution number within which each ToA falls
	* __column 2__: The zero crossing time at the start of the revolution
	* __column 3__: The zero crossing time at the end of the revolution
	* __column 4__: The angular velocity of the shaft within the revolution
	* __column 5__: The ToA value
	* __column 6__: The AoA of the ToA value

* In Line 34 we initialize the revolution number of all the rows in `AoA_matrix` to -1. We are going to use the -1 revolution number to flag the ToAs that could not be converted to AoAs.
* In Line 36-38 we introduce the concept of the *current revolution*. Since this is a sequential function we will handle each shaft revolution in turn. `current_n` keeps track of the revolution the loop is in, `current_zero_crossing_start` holds the start of the current revolution, and  `current_zero_crossing_end` holds the end of the current revolution. These "current" values will be updated as we iterate through the ToA values.


### Master loop
``` python linenums="39"
# Some code omitted above 👆
for i, toa in enumerate(arr_probe_toas):
# Some code omitted below 👇
```
??? note "Full code"
	``` py hl_lines="40" linenums="1"
	from numba import njit
	import numpy as np

	@njit
	def calculate_aoa(
		arr_opr_zero_crossing : np.ndarray, 
		arr_probe_toas : np.ndarray
	):
		"""
		This function calculates the angle of arrival of 
		each ToA value relative to the revolution in 
		which it occurs.

		Args:
			arr_opr_zero_crossing (np.array): An array of 
				OPR zero-crossing times. 
			arr_probe_toas (np.array): An array of 
				ToA values.

		Returns:
			np.array: A matrix of AoA values. Each row in the 
				matrix corresponds to a ToA value. The columns 
				are:
				0: The revolution number
				1: The zero crossing time at the start of the revolution
				2: The zero crossing time at the end of the revolution
				3: The angular velocity of the revolution
				4: The ToA
				5: The AoA of the ToA value
		"""
		num_toas = len(arr_probe_toas)
		AoA_matrix = np.zeros( (num_toas, 6))

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
				Omega = 2 * np.pi / (
					current_zero_crossing_end 
					- current_zero_crossing_start
				)
				AoA_matrix[i, 3] = Omega
				AoA_matrix[i, 4] = toa
				AoA_matrix[i, 5] = Omega * (
					toa 
					- current_zero_crossing_start
				)

		return AoA_matrix
	```


This `for` loop is the master loop of the algorithm. It iterates through each ToA value in the `arr_probe_toas` array. The counter, `i`, starts at `0`- corresponding to the first ToA value - and increments after each iteration. The variable `toa` is the current ToA value.

### Search for the correct shaft revolution
``` python linenums="41"
# Some code omitted above 👆
while toa > current_zero_crossing_end:
	current_n += 1
	if current_n >= (len(arr_opr_zero_crossing) - 1):
		break
	current_zero_crossing_start = arr_opr_zero_crossing[current_n]
	current_zero_crossing_end = arr_opr_zero_crossing[current_n + 1]

if current_n >= (len(arr_opr_zero_crossing) - 1):
	break
# Some code omitted below 👇
```
??? note "Full code"
	``` py hl_lines="42 43 44 45 46 47 49 50" linenums="1"
	from numba import njit
	import numpy as np

	@njit
	def calculate_aoa(
		arr_opr_zero_crossing : np.ndarray, 
		arr_probe_toas : np.ndarray
	):
		"""
		This function calculates the angle of arrival of 
		each ToA value relative to the revolution in 
		which it occurs.

		Args:
			arr_opr_zero_crossing (np.array): An array of 
				OPR zero-crossing times. 
			arr_probe_toas (np.array): An array of 
				ToA values.

		Returns:
			np.array: A matrix of AoA values. Each row in the 
				matrix corresponds to a ToA value. The columns 
				are:
				0: The revolution number
				1: The zero crossing time at the start of the revolution
				2: The zero crossing time at the end of the revolution
				3: The angular velocity of the revolution
				4: The ToA
				5: The AoA of the ToA value
		"""
		num_toas = len(arr_probe_toas)
		AoA_matrix = np.zeros( (num_toas, 6))

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
				Omega = 2 * np.pi / (
					current_zero_crossing_end 
					- current_zero_crossing_start
				)
				AoA_matrix[i, 3] = Omega
				AoA_matrix[i, 4] = toa
				AoA_matrix[i, 5] = Omega * (
					toa 
					- current_zero_crossing_start
				)

		return AoA_matrix
	```

	
This code section has the responsibility of finding the shaft revolution within which the current ToA occurs. This is achieved by repeatedly checking, in Line 42, if the current ToA is larger than the current shaft revolution's end time. If it is, then we know that the shaft has completed a revolution since the previous ToA. We then increment the current shaft revolution (` current_n`) variable. We also update the current shaft revolution's start and end times in lines 46 and 47.

We add checks in lines 44 and 49 that break out of the main loop if there are no more zero-crossing times to compare the ToAs to. If this happens, our algorithm has finished its job.

### Calculate the AoA matrix values for each ToA
``` py linenums="51"
# Some code omitted above 👆
if toa > current_zero_crossing_start:
	AoA_matrix[i, 0] = current_n
	AoA_matrix[i, 1] = current_zero_crossing_start
	AoA_matrix[i, 2] = current_zero_crossing_end
	Omega = 2 * np.pi / (
		current_zero_crossing_end 
		- current_zero_crossing_start
	)
	AoA_matrix[i, 3] = Omega
	AoA_matrix[i, 4] = toa
	AoA_matrix[i, 5] = Omega * (
		toa 
		- current_zero_crossing_start
	)

return AoA_matrix
```

??? note "Full code"
	``` py hl_lines="52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68" linenums="1"
	from numba import njit
	import numpy as np

	@njit
	def calculate_aoa(
		arr_opr_zero_crossing : np.ndarray, 
		arr_probe_toas : np.ndarray
	):
		"""
		This function calculates the angle of arrival of 
		each ToA value relative to the revolution in 
		which it occurs.

		Args:
			arr_opr_zero_crossing (np.array): An array of 
				OPR zero-crossing times. 
			arr_probe_toas (np.array): An array of 
				ToA values.

		Returns:
			np.array: A matrix of AoA values. Each row in the 
				matrix corresponds to a ToA value. The columns 
				are:
				0: The revolution number
				1: The zero crossing time at the start of the revolution
				2: The zero crossing time at the end of the revolution
				3: The angular velocity of the revolution
				4: The ToA
				5: The AoA of the ToA value
		"""
		num_toas = len(arr_probe_toas)
		AoA_matrix = np.zeros( (num_toas, 6))

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
				Omega = 2 * np.pi / (
					current_zero_crossing_end 
					- current_zero_crossing_start
				)
				AoA_matrix[i, 3] = Omega
				AoA_matrix[i, 4] = toa
				AoA_matrix[i, 5] = Omega * (
					toa 
					- current_zero_crossing_start
				)

		return AoA_matrix
	```

This section is responsible for calculating the AoA values. In Line 52 we test whether the current ToA value is larger than the current revolution's start time. If we pass this test, we know that this ToA  falls within the `current_n` revolution. We then assign the current revolution, starting zero-crossing time, and ending zero-crossing time to the AoA matrix in lines 53 - 55.

In lines 56 - 65 we calculate the shaft speed and the corresponding AoA value for this ToA as defined in [Equation 1](#equation_01) and [Equation 2](#equation_02) respectively.

If we've reached Line 67, it means we have either iterated over all ToAs, or we don't have OPR zero-crossing times left. We therefore return the AoA matrix.

## Example usage

We'll use some real experimental data to test this function. We're going to use a shaft run-up and down from Du Toit et al. [@du2019stochastic]. In the experiment, a five blade rotor was run up and down, as already indicated in [Figure 1](#figure_01). Four eddy current probes were used to measure the ToAs. 

ToA extraction has already been done. We're only going to use a single probe's data for this example.  

``` py linenums="1"
dataset = Datasets["data/intro_to_btt/intro_to_btt_ch03"] #(1)!
df_opr_zero_crossings = \ #(2)!
	dataset['table/du_toit_2017_test_1_opr_zero_crossings'] #(3)!
df_probe_toas = dataset['table/du_toit_2017_test_1_prox_1_toas'] #(4)!
AoA_matrix = calculate_aoa(#(5)!
    df_opr_zero_crossings["time"].to_numpy(),
    df_probe_toas["time"].to_numpy()
)
df_AoA = pd.DataFrame(#(6)!
    AoA_matrix, 
    columns=[
        "n",
        "n_start_time",
        "n_end_time",
        "Omega",
        "ToA",
        "AoA"
    ]
)
```

1.  This downloads the data from Bladesight's data repository. The dataset comprises 6 tests. Each test has one set of OPR zero-crossing times, four sets of ToAs, and one set of MPR zero-crossing times. We do not use the MPR zero-crossing times in this tutorial.
2.  The `\` token allows us to break a line. It is simply to improve readability.
3.  This line loads the OPR zero-crossing times from Test 1 as a Pandas DataFrame.
4.  This line loads the first proximity probe's ToAs from Test 1 as a Pandas DataFrame.
5.  In this line, we run the `calculate_aoa` function using the OPR zero-crossing times and the ToAs as inputs. Note how we convert the Pandas columns to Numpy arrays using `.to_numpy()`
6.  The algorithm returns a numpy matrix, which does not have column names. Here we convert the matrix into a DataFrame. DataFrames are much simpler to work with.

We show the first 10 rows of the AoA DataFrame in [Table 1](#table_01) below.

<figure markdown>
  <figcaption><strong><a name='table_01'>Table 1</a></strong>: The first 10 rows of the AoA DataFrame. </figcaption>
</figure>
{{ read_csv('docs/tutorials/intro_to_btt/ch3/AoA_top_10.csv') }}

We note two things:

*  The first two rows have `n` values of -1. This means the first two ToAs were recorded *before* the first zero-crossing time. We can therefore not calculate the AoA values for these ToAs. You'll find a couple of these values at the end of the DataFrame as well.
*  We see that the `AoA` values seem to repeat themselves every 5 rows. This is because the shaft has 5 blades.

We can simply drop the ToA values that could not be converted to AoAs. We do this by filtering the DataFrame to only include rows where `n` is not equal to -1.

``` py linenums="1"
df_AoA = df_AoA[df_AoA["n"] != -1]
df_AoA.reset_index(inplace=True, drop=True) #(1)!
```

1.	Because we dropped the first two rows of data, the DataFrame's index is going to start at 2. It's always best to reset the index after dropping rows, unless the index contains important information.


In [Figure 2](#figure_02) below, we show a plot of the AoA values for the first blade over all revolutions. We also plot the shaft speed on a secondary y-axis.

<script src="blade_1_aoas.js" > </script>
<div>
	<div>
		<canvas id="ch03_blade_1_aoas"'></canvas>
	</div>
	<script>
		async function render_chart_blade_1_aoas() {
			const ctx = document.getElementById('ch03_blade_1_aoas');
			// If this is a mobile device, set the canvas height to 400
			if (window.innerWidth < 500) {
				ctx.height = 400;
			}
			while (typeof Chart == "undefined") {
				await new Promise(r => setTimeout(r, 1000));
			}
			Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
			window.fig_blade_1_aoas = new Chart(ctx, window.blade_1_aoas);
			window.fig_blade_1_aoas_reset = function resetZoomFig2() {
					window.fig_blade_1_aoas.resetZoom();
				}
			}
		render_chart_blade_1_aoas();
	</script>
	<a onclick="window.fig_blade_1_aoas_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_01'>Figure 2</a></strong>: The AoAs of the first blade arriving at the first proximity probe. </figcaption>
</figure>

In [Figure 2](#figure_02) above, we see that the AoA values are not constant. We now discuss three forms of noise found in the signal.

### Sensor limited bandwidth noise
Firstly, we see that the AoA values exhibit a drift that seems to be correlated to the shaft speed. In other words, as the shaft speeds up, the AoA values increase proportionally to the shaft speed. The moment the shaft starts to run down, the AoA values decrease proportionally to the shaft speed. 

This is *not* related to blade vibration. This is related to the bandwidth of the sensor you are using. Any sensor has a limited bandwidth. As a blade's tip moves faster, its presence inside your sensor's field of view becomes shorter. This causes the input function experienced by the sensor to contain more energy at higher frequencies. 

If your sensor's bandwidth is limited, the response cuts out these high frequency components, leading to lower amplitude pulses. A lower amplitude pulse will exhibit later ToA triggering, and hence a larger AoA value. 

This is not a problem, however, because our resonance events occur at higher frequencies than this phenomenon. In later chapters, we will remove this form of noise using a simple detrending algorithm.

### Blade vibration
There are four clear resonance events in this signal. They occur at 7.5, 17.83, 24.3, and 34.5 seconds respectively. We will delve into the theory of resonance in a later chapter. For now, we highlight that there are actually just two unique resonances. The two resonances occur at 1087 RPM and 1275 RPM on both the run-up and the run down.

### High frequency noise
We also observe high-frequency noise that seems random. This could be because of many things, such as electrical noise, shaft torsional vibration, casing vibration, or even higher frequency or random vibration of the blades. You may remove this noise, or construct inference algorithms that take it into account. 

## Conclusion
In this chapter, we have converted the ToA values into AoA values. Using only this step, we have managed to reveal some resonance events to the naked eye.

We are, however, only at the start of our journey. In the next chapter, we'll learn how to identify individual blades from this AoA DataFrame.

!!! success "Outcomes"

	:material-checkbox-marked:{ .checkbox-success .heart } Understand that we use a shaft encoder to calculate the shaft speed, $\Omega$, and the start and end of each revolution. 

    :material-checkbox-marked:{ .checkbox-success .heart } Understand that we need to find the shaft revolution in which each ToA occurs.

	:material-checkbox-marked:{ .checkbox-success .heart } Understand that each ToA is used to calculate the precise shaft circumferential displacement in said revolution, leading to the AoA.
	
	:material-checkbox-marked:{ .checkbox-success .heart } Write a function that calculates a matrix of AoA values from the shaft encoder zero-crossing times and the ToA values.

{==

Consider doing the coding exercises below 👇 to solidify your understanding of the concepts we've covered in this chapter. 

==}

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
            2023-10-07
        </p>
    </div>
</div>

## :material-weight-lifter:{ .checkbox-success } Coding Exercises

Here are some coding exercises to solidify the concepts we've covered in this chapter.

### Problem 1 :yawning_face:
In this chapter, we've written the `calculate_aoa` function. This function receives and returns Numpy arrays, even though we work with Pandas DataFrames natively. We therefore needed to use the `.to_numpy()` method to convert the Pandas objects to numpy arrays. We also needed to cast the resulting matrix into a Pandas DataFrame, and drop the ToA values that could not be converted to AoA values. 

In future, we do not want to do this every time we call the function.

{==

:material-pencil-plus-outline: Write a new function, called `transform_ToAs_to_AoAs`, that receives Pandas DataFrames as inputs and returns a Pandas DataFrame as output. The function should call the `calculate_aoa` function to do the actual work.

==}

??? example "Reveal answer (Please try it yourself before peeking)"
	``` py linenums="1"
	def transform_ToAs_to_AoAs(
		df_opr_zero_crossings : pd.DataFrame,
		df_probe_toas : pd.DataFrame,
	) -> pd.DataFrame:
		""" This function transforms the ToA values to AoA values for a 
		single probe, given the OPR zero-crossing times and the proximity
		probe's ToA values.

		The timestamps are assumed to reside in the first column of
		each DataFrame.

		Args:
			df_opr_zero_crossings (pd.DataFrame): A DataFrame with the 
				OPR zero-crossing times.
			df_probe_toas (pd.DataFrame): A DataFrame with the probe's 
				ToA values.

		Returns:
			pd.DataFrame: A DataFrame with the AoA values.
		"""
		AoA_matrix = calculate_aoa(
			df_opr_zero_crossings.iloc[:, 0].to_numpy(), #(1)!
			df_probe_toas.iloc[:, 0].to_numpy()
		)
		df_AoA = pd.DataFrame(
			AoA_matrix, 
			columns=[
				"n",
				"n_start_time",
				"n_end_time",
				"Omega",
				"ToA",
				"AoA"
			]
		)
		df_AoA = df_AoA[df_AoA["n"] != -1]
		df_AoA.reset_index(inplace=True, drop=True)
		return df_AoA
	```

	1.	We may want to pass in DataFrames with different columns names than the ones we used in the example. We therefore use the `.iloc` method to get the first column of each DataFrame, regardless of what it is called.


	Example usage:
	``` py linenums="1"
	>>> dataset = Datasets["data/intro_to_btt/intro_to_btt_ch03"]
	>>> df_opr_zero_crossings = dataset['table/du_toit_2017_test_1_opr_zero_crossings']
	>>> df_probe_toas = dataset['table/du_toit_2017_test_1_prox_1_toas']

	>>> df_AoA = transform_ToAs_to_AoAs(
		df_opr_zero_crossings,
		df_probe_toas
	)
	```

### Problem 2 :neutral_face:
Because our Python functions are being converted to C, it is tempting to treat our code inefficiencies with a wink and a wry smile, like one does with a child that does something naughty you are secretly proud of.

We must, however, resist this temptation. We must always strive to write efficient code. 

{==

:material-pencil-plus-outline: We are doing unnecessary calculations in the `calculate_aoa` function. Can you spot where? Rewrite the function to remove this inefficiency.

==}

??? example "Reveal answer (Please try it yourself before peeking)"
	``` py linenums="1" hl_lines="35 36 37 38 49 50 51 52"	
	@njit
	def calculate_aoa(
		arr_opr_zero_crossing : np.ndarray, 
		arr_probe_toas : np.ndarray
	):
		"""
		This function calculates the angle of arrival of 
		each ToA value relative to the revolution in 
		which it occurs.

		Args:
			arr_opr_zero_crossing (np.array): An array of 
				OPR zero-crossing times. 
			arr_probe_toas (np.array): An array of 
				ToA values.

		Returns:
			np.array: A matrix of AoA values. Each row in the 
				matrix corresponds to a ToA value. The columns 
				are:
				0: The revolution number
				1: The zero crossing time at the start of the revolution
				2: The zero crossing time at the end of the revolution
				3: The angular velocity of the revolution
				4: The ToA
				5: The AoA of the ToA value
		"""
		num_toas = len(arr_probe_toas)
		AoA_matrix = np.zeros( (num_toas, 6))

		AoA_matrix[:, 0] = -1

		current_zero_crossing_start = arr_opr_zero_crossing[0]
		current_zero_crossing_end = arr_opr_zero_crossing[1]
		Omega = 2 * np.pi / (
			current_zero_crossing_end 
			- current_zero_crossing_start
		)
		current_n = 0

		for i, toa in enumerate(arr_probe_toas):

			while toa > current_zero_crossing_end:
				current_n += 1
				if current_n >= (len(arr_opr_zero_crossing) - 1):
					break
				current_zero_crossing_start = arr_opr_zero_crossing[current_n]
				current_zero_crossing_end = arr_opr_zero_crossing[current_n + 1]
				Omega = 2 * np.pi / (
					current_zero_crossing_end 
					- current_zero_crossing_start
				)
			if current_n >= (len(arr_opr_zero_crossing) - 1):
				break

			if toa > current_zero_crossing_start:
				AoA_matrix[i, 0] = current_n
				AoA_matrix[i, 1] = current_zero_crossing_start
				AoA_matrix[i, 2] = current_zero_crossing_end
				AoA_matrix[i, 3] = Omega
				AoA_matrix[i, 4] = toa
				AoA_matrix[i, 5] = Omega * (
					toa 
					- current_zero_crossing_start
				)

		return AoA_matrix
	```
	We have moved the calculation of the shaft speed to the while loop from the if statement at the bottom. Now, every time we update the zero-crossing times, we calculate the shaft speed only once. The previous method calculated the shaft speed once for every blade on the rotor, since that is the number of ToAs occurring inside the revolution.

### Problem 3 :thinking:
The dataset we used in this chapter also has MPR zero-crossing times which we did not use. The MPR zero crossing times can be loaded with the following command:

``` py
df_mpr_zero_crossings = dataset['table/du_toit_2017_test_1_mpr_zero_crossings']
```

The MPR encoder used to measure these zero-crossing times has 79 sections. You can therefore assume that each zero-crossing time corresponds to $\frac{1}{79}$ of a shaft rotation.

{==

:material-pencil-plus-outline: Write a new function, `calculate_aoa_from_mpr`, that receives the MPR timestamps, the number of sections in the MPR, and the ToA values as inputs. The function should return a DataFrame with the AoA values. The resulting AoA matrix should also include the MPR section number within which each ToA occurs.

==}

<script src="blade_5_aoas_using_mpr.js" > </script>
<div>
	<script>
		async function render_chart_blade_5_aoas_using_mpr() {
			const ctx = document.getElementById('ch03_blade_5_aoas_using_mpr');
			// If this is a mobile device, set the canvas height to 400
			if (window.innerWidth <500) {
				ctx.height = 400;
			}
			while (typeof Chart == "undefined") {
				await new Promise(r => setTimeout(r, 1000));
			}
			Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
			window.fig_blade_5_aoas_using_mpr = new Chart(ctx, window.blade_5_aoas_using_mpr);
			window.fig_blade_5_aoas_using_mpr_reset = function resetZoomFig3() {
					window.fig_blade_5_aoas_using_mpr.resetZoom();
				}
			}
	</script>
</div>

??? example "Reveal answer (Please try it yourself before peeking)"
	``` py linenums="1"
	def calculate_aoa_from_mpr(
		arr_mpr_zero_crossing : np.ndarray,
		arr_probe_toas : np.ndarray,
		mpr_sections : int = 1,
	) -> np.ndarray:
		""" This function calculates the angle of arrival of
		each ToA value relative to the section and revolution in
		which it occurs when using an MPR encoder.

		Args:
			arr_mpr_zero_crossing (np.ndarray): An array of MPR
				zero-crossing times.
			arr_probe_toas (np.ndarray): An array of ToA values.
			mpr_sections (int, optional): The number of sections
				in the MPR encoder. Defaults to 1, in this case,
				this function will be treated as an OPR encoder.

		Returns:
			np.ndarray: A matrix of AoA values. Each row in the
				matrix corresponds to a ToA value. The columns
				are:
				0: The revolution number
				1: The section number
				2: The zero crossing time at the start of the revolution
				3: The zero crossing time at the end of the revolution
				4: The angular velocity of the revolution
				5: The ToA
				6: The AoA of the ToA value
		"""
		num_toas = len(arr_probe_toas)
		AoA_matrix = np.zeros((num_toas, 7))
		rad_per_section = 2 * np.pi / mpr_sections
		AoA_matrix[:, 0] = -1

		current_zero_crossing_start = arr_mpr_zero_crossing[0]
		current_zero_crossing_end = arr_mpr_zero_crossing[1]
		Omega = rad_per_section / (
			current_zero_crossing_end 
			- current_zero_crossing_start
		)
		current_n = 0
		current_revo = 0
		current_section = 0

		for i, toa in enumerate(arr_probe_toas):

			while toa > current_zero_crossing_end:
				current_n += 1
				if current_n >= (len(arr_mpr_zero_crossing) - 1):
					break
				current_zero_crossing_start = arr_mpr_zero_crossing[current_n]
				current_zero_crossing_end = arr_mpr_zero_crossing[current_n + 1]
				Omega = rad_per_section / (
					current_zero_crossing_end 
					- current_zero_crossing_start
				)
				current_section += 1
				if current_section == mpr_sections:
					current_section = 0
					current_revo += 1

					
			if current_n >= (len(arr_mpr_zero_crossing) - 1):
				break

			if toa > current_zero_crossing_start:
				AoA_matrix[i, 0] = current_revo
				AoA_matrix[i, 1] = current_section
				AoA_matrix[i, 2] = current_zero_crossing_start
				AoA_matrix[i, 3] = current_zero_crossing_end
				AoA_matrix[i, 4] = Omega
				AoA_matrix[i, 5] = toa
				AoA_matrix[i, 6] = Omega * (
					toa
					- current_zero_crossing_start
				) + current_section * rad_per_section

		return AoA_matrix
		
	def transform_ToAs_to_AoAs_mpr(
		df_mpr_zero_crossings : pd.DataFrame,
		df_probe_toas : pd.DataFrame,
		mpr_sections : int = 1,
	) -> pd.DataFrame:
		""" This function transforms the ToA values to AoA values for a 
		single probe, given the MPR zero-crossing times and the proximity
		probe's ToA values.

		The timestamps are assumed to reside in the first column of
		each DataFrame.

		Args:
			df_opr_zero_crossings (pd.DataFrame): A DataFrame with the 
				OPR zero-crossing times.
			df_probe_toas (pd.DataFrame): A DataFrame with the probe's 
				ToA values.
			mpr_sections (int, optional): The number of sections
				in the MPR encoder. Defaults to 1, in this case,
				this function will be treated as an OPR encoder.

		Returns:
			pd.DataFrame: A DataFrame with the AoA values.
		"""
		AoA_matrix = calculate_aoa_from_mpr(
			df_mpr_zero_crossings.iloc[:, 0].to_numpy(),
			df_probe_toas.iloc[:, 0].to_numpy(),
			mpr_sections
		)
		df_AoA = pd.DataFrame(
			AoA_matrix, 
			columns=[
				"n",
				"mpr_section",
				"section_start_time",
				"section_end_time",
				"Omega",
				"ToA",
				"AoA"
			]
		)
		df_AoA = df_AoA[df_AoA["n"] != -1]
		df_AoA.reset_index(inplace=True, drop=True)
		return df_AoA
	```

	The function `calculate_aoa_from_mpr` increments each section, instead of each revolution. The function is, however, capable of transforming the ToAs using an OPR encoder if you set `mpr_sections` to 1.

	In [Figure 3](#figure_03) below, we show the AoA values as calculated using the MPR algorithm vs the exact same values calculated using the OPR algorithm. We see that the AoA values from the MPR algorithm appear less noisy than the AoA values from the OPR algorithm. 
	<div>
		<div>
			<canvas id="ch03_blade_5_aoas_using_mpr"'></canvas>
		</div>
		<a onclick="window.fig_blade_5_aoas_using_mpr_reset()" class='md-button'>Reset Zoom</a>
	</div>
	<script> render_chart_blade_5_aoas_using_mpr(); </script>
	<figure markdown>
	  <figcaption><strong><a name='figure_03'>Figure 3</a></strong>: The AoAs of the fifth blade arriving at the first proximity probe. The AoAs were calculated using the MPR algorithm and the OPR algorithm. </figcaption>
	</figure>

	MPR encoders will always produce more accurate BTT results than OPR encoders. One reason for this is because MPR encoders capture high frequency shaft torsional vibration, allowing us to remove it from the AoAs. An OPR encoder cannot capture this torsional vibration. High frequency shaft torsional vibration will therefore *appear* as high-frequency content in our AoAs when using OPR encoders.

	MPR encoder signal processing is, however, nontrivial. For instance, we have assumed here that all encoder sections have the same circumferential width. This is almost never the case. We will not venture into the dark art of MPR encoders here, we'll leave that for a future tutorial!

