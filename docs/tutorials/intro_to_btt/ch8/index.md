---
date: 2024-03-19
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
description: This chapter explains how we can use the SDoF fit method to infer the vibration parameters of a blade.
robots: index, follow, Blade Tip Timing, BTT, Non Intrusive Stress Measurement, NSMS, Time of Arrival, Turbine blade,Mechanical Vibration
template: main_intro_to_btt_ch8.html
card_title: Intro to BTT Ch8 - The SDoF fit method
card_url: "ch8/"
---
??? abstract "You are here"
	<figure markdown>
	![BTT Tutorial Roadmap](BTT_Tutorial_Outline_Ch7.svg){ width="500" }
	</figure>
# Single Degree of Freedom (SDoF) Fit Method
In the previous chapter, we used a fully defined SDoF vibration model to *generate* the samples we typically observe in BTT data. 

In this chapter, we do the opposite. 

The *SDoF fit method* comprises solving the *inverse problem*. In other words, we *already have*  the sampled tip deflections for each probe. Now we need to find the SDoF model's parameters. I've put together a GIF that captures the essence of the task. Our model (represented by the red line) changes until it lies on top of the observed datapoints (blue).

<figure markdown>
  ![SDoF Fit intro gif](intro_gif.gif)
  <figcaption><a id='figure_01'><strong>Figure 1:</strong></a>
    An animation of how the SDoF fit method's predicted tip deflections change as the natural frequency, damping ratio, and static deflection converge to the correct solution. I added some animations to the dots and a glitch effect. These effects do not have any meaning... I just couldn't help myself 😁.
  </figcaption>
</figure>

We restate the SDoF equations below for convenience.
<a name='eq_1'></a>

\begin{equation}
x(t) = X(\omega) \cos \left( \theta_s \cdot EO - \phi(\omega) \right)
\end{equation}

<a name='eq_2'></a>

\begin{equation}
X(\omega) = \frac{\delta_{\text{st}}}{ \sqrt{ (1 - r^2)^2 + (2 \zeta r)^2 } }\\
\end{equation}

<a name='eq_3'></a>

\begin{equation}
\phi(\omega) = \arctan \left( \frac{2 \zeta r}{1 - r^2} \right)\\
\end{equation}

<a name='eq_4'></a>

\begin{equation}
r = \frac{\omega}{\omega_n}
\end{equation}

??? info "Symbols"

    | Symbol | Meaning | SI Unit | Domain|
    | --- | --- | --- | --- |
    | $x(t)$ | Tip deflection | $\mu m$ | $\mathbb{R}$ |
    | $X(\omega)$ | Vibration amplitude | $\mu m$  | $\mathbb{R}+$  |
    | $\delta_{\text{st}}$ | Static deflection | $\mu m$ | $\mathbb{R}+$ |
    | $\zeta$ | Damping ratio | - | $[0,1)$ for underdamped systems |
    | $\omega_n$ | Natural frequency | $rad/s$ or Hz | $\mathbb{R}+$ |
    | $\theta_s$ | Sensor position | $rad$ | $[0, 2\pi]$ |
    | $\phi(\omega)$ | Phase angle | $rad$ | $\mathbb{R}$ |
    | $\omega$ | Excitation frequency | $rad/s$ or Hz | $\mathbb{R}+$ |
    | $EO$ | Engine Order | - | $\mathbb{Z}+$ |


Our task is to find values for $\delta_{\text{st}}$, $\zeta$, and $\omega_n$ that best fit the data. This approach was set forth as early as 1978. It is, to my best knowledge, the first BTT vibration inference method ever proposed [@zablotskiy1978measurement]. 

In this chapter, we first produce a simple solution to the above equations. Layers of complexity are added incrementally. I use the incremental approach to emphasize the nonlinear nature of this problem. If you invest time in understanding these concepts, you can apply the lessons wherever you need to determine parameters inside sinusoidal terms in the future. 

This chapter culminates in the fulfillment of the promise I made at the start of the tutorial. You'll be able to infer blade frequency, amplitude, and phase from raw time stamps.

!!! question "Outcomes"

	:material-checkbox-blank-outline: Understand why we require an optimization function to determine the model parameters of the SDoF fit method.

	:material-checkbox-blank-outline: Understand why we need to specify upper and lower bounds for the model parameters.

    :material-checkbox-blank-outline: Understand why the SDoF model, in its raw form, is not adequate. We need to add phase and amplitude offsets to the objective function.

	:material-checkbox-blank-outline: Understand how we can loop over each probe's data to fit the SDoF model to multiple probes.
	
	:material-checkbox-blank-outline: Understand why the EO is treated differently than the other model parameters. Because the EO can only be a positive integer, we iterate over the EOs to identify the optimal EO.


## Follow along

The worksheet for this chapter can be downloaded here <a href="https://github.com/Bladesight/bladesight-worksheets/blob/master/intro_to_btt/ch_08_worksheet.ipynb" target="_blank"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="Open In Github"/></a>.


You can open a Google Colab session of the worksheet here: <a href="https://colab.research.google.com/github/Bladesight/bladesight-worksheets/blob/master/intro_to_btt/ch_08_worksheet.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>.

You need to use one of these Python versions to run the worksheet:
<img src="https://img.shields.io/badge/python-3.9-blue.svg">
<img src="https://img.shields.io/badge/python-3.10-blue.svg">
<img src="https://img.shields.io/badge/python-3.11-blue.svg">

## Getting the dataset
Let's use the dataset from Chapter 5 again. The functions developed in chapters 1 - 6 have been included in the `bladesight` package. 

We use these functions to generate the tip deflections for the dataset:

``` py linenums="1"
ds = Datasets["data/intro_to_btt/intro_to_btt_ch05"]
df_opr_zero_crossings = ds['table/opr_zero_crossings']
df_prox_1 = ds['table/prox_1_toas']
df_prox_2 = ds['table/prox_2_toas']
df_prox_3 = ds['table/prox_3_toas']
df_prox_4 = ds['table/prox_4_toas']

BLADE_COUNT = 5
RADIUS = 164000

rotor_blade_AoA_dfs = get_rotor_blade_AoAs(
    df_opr_zero_crossings,
    [df_prox_1, df_prox_2, df_prox_3, df_prox_4],
    np.cumsum(np.deg2rad(np.array([19.34, 19.34, 19.34]))), # (1)!
    BLADE_COUNT
)
tip_deflection_dfs = []
for df_AoAs in rotor_blade_AoA_dfs:
    df_tip_deflections = get_blade_tip_deflections_from_AoAs(
        df_AoAs,
        RADIUS,
        11,#(2)!
        2,#(3)!
        0.5#(4)!
    )
    tip_deflection_dfs.append(df_tip_deflections)
```

1.  These are the angles between probes 2 to 4 relative to probe 1.
2.  The order of the zeroing polynomial.
3.  The order of the Butterworth filter.
4.  The cutoff frequency of the Butterworth filter.

We plot the peak-to-peak vibration for blade 1 in [Figure 2](#figure_02) below.

<script src="c08_blade_1_pk_pk.js" > </script>
<div>
	<div>
		<canvas id="c08_blade_1_pk_pk"'></canvas>
	</div>
	<script>
		async function render_chart_c08_blade_1_pk_pk() {
			const ctx = document.getElementById('c08_blade_1_pk_pk');
			// If this is a mobile device, set the canvas height to 400
			if (window.innerWidth < 500) {
				ctx.height = 400;
			}
			while (typeof Chart == "undefined") {
				await new Promise(r => setTimeout(r, 1000));
			}
			Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
			window.fig_c08_blade_1_pk_pk = new Chart(ctx, window.c08_blade_1_pk_pk);
			window.fig_c08_blade_1_pk_pk_reset = function resetZoomFig1() {
					window.fig_c08_blade_1_pk_pk.resetZoom();
				}
			}
		render_chart_c08_blade_1_pk_pk();
	</script>
	<a onclick="window.fig_c08_blade_1_pk_pk_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption>
        <strong><a name='figure_02'>Figure 2</a></strong>: The peak-to-peak vibration levels for a five blade blisk with four probes. The left and right edges of the peak-to-peak chart curls up but never returns back to a low value, as is usually the case with transient resonances. This is a consequence of the zeroing we performed. Since those regions don't contain resonances, we can live with these end effects.
  </figcaption>
</figure>

The peak-to-peak vibration for blade 1 is presented in [Figure 2](figure_02) above. There seems to be symmetric resonance events on the up and downward ramps. 

For illustrative purposes, we focus on the resonance event between shaft revolutions 500 and 600.

!!! note "End effects"

    You may have noticed the ends of the peak-to-peak vibration increase without a subsequent decrease. Resonance events are usually identified by the peak-to-peak vibration increasing as the event starts, and decreasing as the event finishes.
    
    The end effects observed in [Figure 2](#figure_02) is not a resonance event. It is an artifact of the order domain polynomial detrending we performed <a href=https://docs.bladesight.com/tutorials/intro_to_btt/ch6/#order-domain-polynomial-detrending target="_blank">in Chapter 6</a>. The order domain zeroing often skews the peak-to-peak vibration values near the edges.
    
    If there were resonances in those regions, we would have taken the time to perform better zeroing. From prior experience, I can confirm there are no resonances in those regions, so we can simply proceed.

The original method [@zablotskiy1978measurement] used only one probe's data to fit the model. We'll also start with just one probe. 

## The nontrivial nature of the problem
It is conventional for engineers to invert the equation that contains the variable we want to solve. Let's start with solving for the EO. We invert [Equation 1](#eq_1) below:

<a name='eq_5'></a>

\begin{align}
x(t) &= X(\omega) \cos \left( \theta_s \cdot EO - \phi(\omega) \right ) \\
\frac{x(t)}{X(\omega)} &= \cos \left( \theta_s \cdot EO - \phi(\omega) \right ) \\
\arccos \left( \frac{x(t)}{X(\omega)} \right) &= \theta_s \cdot EO - \phi(\omega) \\
\theta_s \cdot EO &= \arccos \left( \frac{x(t)}{X(\omega)} \right) + \phi(\omega) \\
EO &= \frac{\arccos \left( \frac{x(t)}{X(\omega)} \right) + \phi(\omega)}{\theta_s}
\end{align}

the below domain constraint is required:

\begin{equation}
-1 \leq \frac{x(t)}{X(\omega)} \leq 1
\end{equation}

How easy is it to solve for $EO$ using this inversion? I have no idea :man_shrugging: . It seems absurdly difficult to me. 

Firstly, the model parameters, $\delta_{\text{st}}$, $\zeta$, and $\omega_n$, are all inside the $X(\omega)$ and $\phi(\omega)$ terms. They are also unknown! We could rearrange the equations to solve for these parameters, but we would have the same problem. Our model parameters cannot be solved for directly.

Secondly, we have a domain constraint on the argument of the arccos function. This means we have to solve for $EO$ using a nonlinear constraint. I don't like the sound of that. If you do manage to find a closed form analytic solution for the model parameters, don't send it to me, send it to the Nobel committee.

We therefore resort to computational techniques. We'll use an optimization algorithm to find the solution.

The first step we take toward a solution is to *guess* the EO of vibration. The decision to arbitrarily *guess* the EO may make you uneasy. In most undergraduate courses, we are used to well-posed problems. Give me the mass of a body and the force acting on it, and I can tell you the acceleration with 100% certainty. Solving the SDoF fit method is not like that.

Solving the SDoF fit problem is a bit like dating. The only way to find the right person is to first commit to the relationship, and assess how it goes.

How can we make an intelligent guess for the EO? We can use the formula below:

\begin{equation}
EO = \frac{\text{Natural frequency}}{\text{Shaft speed}}
\end{equation}

We've seen this equation before. It is the definition of the Engine Order. The EO is the integer multiple of the shaft speed that determines the excitation frequency.

For the blades used in this tutorial, I happen to know (because I designed these blades) that the blades' first natural frequency is approximately 125 Hz. To get a shaft speed, we take the median shaft speed inside the resonance region that we've identified between revolutions 500 and 600, which is 950 RPM:

$$
\text{EO} = \frac{125}{950 / 60} \approx 7.9
$$

We'll therefore start with a guess of 8 for our EO. Is it a good guess? That will be revealed at the end of this chapter.

!!! note "What if the natural frequencies are not known?"

    Don't worry. We just need to pick an EO to start with for our optimization algorithm. Later in the chapter, we loop over a range of EOs and compare the objective function values to identify the optimal EO. 

!!! note "Why are we assuming there is only one frequency that dominates the response?"

    This entire chapter is based on the assumption that there is one frequency that dominates the response. That is the definition of the SDoF model. It is a simplification that is often good enough. Here's an appropriate quote often attributed to George Box: "All models are wrong, but some are useful" [@box1979robustness]. 
    
    In reality, the blade will respond at many different frequencies simultaneously. However, if the blades are in resonance, the response will likely be dominated by the natural frequency.

## Resonance window selection

To fit our SDoF model, we need to identify data in a resonance region. In other words , we must create a new `DataFrame` that contains only the tip deflections *within* a resonance region. 

We can use the shaft revolution number to construct this new `DataFrame` . The `eval` method of a Pandas DataFrame can be used to this end:

``` py 
df_resonance_window = tip_deflection_dfs[0].query("n > 500 and n < 600")
```
In the code snippet above, we first select the tip deflections that belong to the first blade (`tip_deflection_dfs[0]`), and then we select only the tip deflections occurring between shaft revolutions 500 and 600 (`query("n > 500 and n < 600")`).

!!! note "Can't window selection be automated?"

    Yes. Techniques exist to automatically identify resonance regions. Some of them have been patented. They are not included here because they are not required for solving the SDoF fit equations.

## A simple objective function

Many optimization algorithms are available in Python. These algorithms require an *objective function*. The objective function evaluates how *good* a proposed set of model parameters are by returning an error between the predicted tip deflections and the observed tip deflections. It should return a small error for a good fit and a large error for a bad fit. The optimization algorithm searches for parameters that minimize the objective function.

Let's start with a simple objective function. We'll use the sum of squared error between the measured tip deflections, and the predicted tip deflections defined in [Equation 1](#eq_1) to [Equation 4](#eq_4). 

The code for this objective function is shown below:

``` py linenums="1"
def get_X(
        omega : np.ndarray,
        omega_n : float, 
        zeta: float,
        delta_st: float
    ) -> np.ndarray:#(1)!
    r = omega / omega_n
    return (
        delta_st 
        / np.sqrt(
            (1 - r**2)**2 
            + (2*zeta*r)**2
        )
    )

def get_phi(
    omega : np.ndarray, 
    omega_n : float, 
    zeta: float
) -> np.ndarray:#(2)!
    r = omega / omega_n
    return np.arctan2(2 * zeta * r,1 - r**2)

def predict_sdof_samples_simple(
    omega_n : float,
    zeta : float,
    delta_st : float,
    EO : int,
    theta_sensor : float,
    arr_omega : np.ndarray
) -> np.ndarray:#(3)!
    X = get_X(arr_omega*EO, omega_n, zeta, delta_st)#(4)!
    phi = get_phi(arr_omega*EO, omega_n, zeta)
    predicted_tip_deflections = X * np.cos(theta_sensor * EO - phi)
    return predicted_tip_deflections

def SDoF_simple_loss(
        model_params : np.ndarray,
        arr_tip_deflections : np.ndarray,
        arr_omega : np.ndarray, 
        EO : int, 
        theta_sensor : float
    ) -> np.ndarray:#(5)!
    omega_n, ln_zeta, delta_st = model_params
    zeta = np.exp(ln_zeta)#(6)!
    predicted_tip_deflections = predict_sdof_samples_simple(
        omega_n, zeta, delta_st, EO, theta_sensor, arr_omega
    )
    return np.sum(
        (
            arr_tip_deflections
            - predicted_tip_deflections
        )**2
    )
```

1.  This function returns the vibration amplitude of 
    the blade vibration.
    
    `x(ω) = 	δ_st / sqrt( (1 - r**2)**2 + (2*ζ*r)**2)`

    where:

    r = ω/ω_0

    Args:

        omega (np.ndarray): The excitation frequencies in rad/s.
        
        omega_n (float): The natural frequency of the blade in rad/s.
        
        zeta (float): The damping ratio of the blade vibration.
        
        delta_st (float, optional): The static deflection of the blade. 
            This value is usually given in units of µm.

    Returns:

        np.ndarray: The amplitude of the blade vibration in the
            same units as delta_st.

2.  Get the phase between the tip deflection and 
        the forcing function. 

    φ(ω) = arctan(2*ζ*r /  (1 - r**2))
    
    where:
    r = ω/ω_n

    Args:
        omega (np.ndarray): The excitation frequencies in rad/s.

        omega_0 (float): The natural frequency of the blade in rad/s.
        
        delta (float): The damping ratio of the blade vibration.

    Returns:

        np.ndarray: The phase of the blade vibration in rad.

3.  This function determined the predicted SDoF fit
    samples at a proximity probe given the SDoF parameters.

    Args:

        omega_n (float): The natural frequency of the SDoF system.
        
        zeta (float): The damping ratio of the SDoF system.
        
        delta_st (float): The static deflection of the SDoF system.
        
        EO (int): The EO of vibration you want to fit.
        
        theta_sensor (float): The sensor's angular position on the rotor.
        
        arr_omega (np.ndarray): The angular velocity of the rotor corresponding
            to each revolution for which we want to predict the SDoF samples.

    Returns:

        np.ndarray: The predicted SDoF samples.

4.  Note how we multiply the shaft speed with EO to get the excitation frequency.

5.  This function returns the error between the SDoF model given the 
    parameters and the measured tip deflection data.

    Args:
        
        model_params (np.ndarray): The SDoF fit method's model parameters. It
            includes a list of the following parameters:
            omega_n (float): The natural frequency of the SDoF system.
            ln_zeta (float): The damping ratio of the SDoF system.
            delta_st (float): The static deflection of the SDoF system.
        
        arr_tip_deflections (np.ndarray): The tip deflection data of the probe.
        
        arr_omega (np.ndarray): The angular velocity of the rotor corresponding
            to the tip deflection data.
        
        EO (int): The EO of vibration you want to fit.
        
        theta_sensor (float): The sensor's angular position on the rotor.

    Returns:

        np.ndarray: The sum of squared error between the tip deflection data
            and the predicted tip deflections.

6.  We use the logarithm of the damping ratio to make the optimization
    surface more linear. This helps the optimization algorithm converge faster.

We've broken up our loss function to make it easier to understand. It is dependant on three other functions:

1.  `get_X`
2.  `get_phi`
3.  `predict_sdof_samples_simple`

The `get_X` and `get_phi` functions are used to calculate the vibration amplitude and phase. The `predict_sdof_samples_simple` function uses these values to predict the tip deflections for a given set of SDoF parameters. The suffix `_simple` has been added to the function names. It will later become clear why I did this.

## Optimization bounds
Which optimization function should we use?

One of the challenges of solving optimization problems where the optimization parameters occur inside sinusoidal terms is they often struggle to converge. This is because the solution space may have many local minima and maxima, and the gradient may not point in the direction of the global optimum. 

Our problem contains both cosine and arctan functions, which means the solution space is likely complex and nonlinear. We therefore choose to use a global optimization method because it does not rely on gradients, but rather explores the solution space with a population of candidate solutions. 

I experimented with several methods, and found the `scipy.optimize.differential_evolution` method works well for this problem. I did not have a more profound reason than this. 

This method requires us to specify bounds for the optimization parameters. We can specify bounds based on our physical constraints and prior knowledge.

### Bounds for the natural frequency
To set the bounds for the natural frequency, we can use a simple formula based on the engine order and the shaft speed. We have already identified the resonance window. We therefore have all shaft speed values occurring inside the resonance window.

We can therefore calculate the minimum and maximum possible natural frequency as follows:

\begin{equation}
\omega_{n, \textrm{min}} = EO \cdot \Omega_{\textrm{min}}
\end{equation}

\begin{equation}
\omega_{n, \textrm{max}} = EO \cdot \Omega_{\textrm{max}}
\end{equation}

The snippet below shows how to implement this in code:
``` py linenums="1"
omega_n_min = df_resonance_window["Omega"].min() * EO
omega_n_max = df_resonance_window["Omega"].max() * EO
```

### Bounds for damping
Our system is underdamped, which means the damping ratio, $\zeta$, is positive and less than 1 [@rao2003vibrations]. From experience, a range between 0.0001 and 0.3 should cover all cases encountered in practice.

In the previous chapter, you had an opportunity to change the $\zeta$ parameter with a slider and observe the effect this has on the vibration response. You may recall that, when $\zeta$ is small, a tiny change in $\zeta$ causes a huge change in the vibration amplitude. The optimization algorithm is not privy to this information. The more linear the optimization surface, the easier it is for the optimization algorithm to find the global minimum.

We therefore transform the damping ratio by calculating its natural logarithm. This way, the optimization algorithm can work with a more straightforward relationship between the parameter it is optimizing, $\ln(\zeta)$, and the objective function. 

We convert the logarithm of $\zeta$ back to the original value inside the objective function, as indicated in Line 45 of the code snippet above.

The bounds for the damping ratio is therefore:

``` py linenums="1"
ln_zeta_min = np.log(0.0001)#(1)!
ln_zeta_max = np.log(0.3)
```

1.  I expected the natural logarithm function to be `np.ln`. For some reason, it is actually `np.log`.

### Bounds for the static deflection

The static deflection bounds are the most application specific. It depends on how large your resonance amplitudes generally are. The higher the vibration amplitude, the wider these bounds should be.

A good initial guess for the current rotor is a value between 0 and 10 $\mu m$ for $\delta_{\textrm{st}}$. You can adjust this value later based on your results.

``` py linenums="1"
delta_st_min = 0
delta_st_max = 10
```

## Solving the simple problem

Now we simply provide these values, along with the optimization function, to the `scipy.optimize.differential_evolution` function. The code snippet below shows how to do this:

``` py linenums="1"
from scipy.optimize import differential_evolution #(1)!
simple_solution = differential_evolution(
    func = SDoF_simple_loss,
    bounds=[#(2)!
        (omega_n_min, omega_n_max),
        (ln_zeta_min, ln_zeta_max),
        (delta_st_min, delta_st_max)
    ],
    args=(#(3)!
        df_resonance_window[f'x_p1_filt'].values,
        df_resonance_window['Omega'].values,
        EO,
        df_resonance_window["AoA_p1"].median()#(4)!
    ),
    seed=42#(5)!
)
```

1.  We import the `differential_evolution` function from the `scipy.optimize` package. In the worksheet, we do this at the top of the worksheet, not where we solve the problem.
2.  We provide the bounds for the optimization parameters. The optimization algorithm calculates there are 3 parameters, and therefore creates an array of three values being passed into the `model_params` argument of the `SDoF_simple_loss` function.
3.  Each one of these arguments correspond to the positional arguments in the `SDoF_simple_loss` function after the `model_params` argument. The `differential_evolution` function will pass these arguments to the `SDoF_simple_loss` function in the same order.
4.  We use the median value of the AoA as the sensor position. Remember, the `AoA_p1` column has not been zeroed or normalized yet. 
5.  The seed is not strictly necessary, but it ensures the results are reproducible on the same computer. You may even get the exact same answers as I did if you use the same seed.

The `differential_evolution` function returns a `OptimizeResult` object. We can print this object:
``` py linenums="1"
>>> print(simple_solution)
```
``` console
 message: Optimization terminated successfully.
 success: True
     fun: 1881551.798646529
       x: [ 7.887e+02 -6.175e+00  1.469e+00]
     nit: 15
    nfev: 760
     jac: [-9.313e-02  3.423e+00 -4.005e+00]
```

Here, the optimization algorithm converged successfully. We can print out the results of the optimization algorithm as follows:

``` py linenums="1"
print("ω_n = ", simple_solution.x[0] / (2*np.pi), " Hz")
print("ζ   = ", np.exp(simple_solution.x[1]))
print("δ_st= ", simple_solution.x[2], " µm")
```

``` console
ω_n =  125.53304440293931  Hz
ζ   =  0.0020816334883631683
δ_st=  1.4691101377911184  µm
```

We have solved the SDoF fit method with a global optimization algorithm 👏. 

Can we confirm this solution is correct?

It is always a good idea to plot the results. The predicted vs true tip deflections are presented in [Figure 3](#figure_03) below.
<script src="c08_resonance_1_simple_fit.js" > </script>
<div>
    <div>
        <canvas id="c08_resonance_1_simple_fit"'></canvas>
    </div>
    <script>
        async function render_chart_c08_resonance_1_simple_fit() {
            const ctx = document.getElementById('c08_resonance_1_simple_fit');
            // If this is a mobile device, set the canvas height to 400
            if (window.innerWidth < 500) {
                ctx.height = 400;
            }
            while (typeof Chart == "undefined") {
                await new Promise(r => setTimeout(r, 1000));
            }
            Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
            window.fig_c08_resonance_1_simple_fit = new Chart(ctx, window.c08_resonance_1_simple_fit);
            window.fig_c08_resonance_1_simple_fit_reset = function resetZoomFig2() {
                    window.fig_c08_resonance_1_simple_fit.resetZoom();
                }
            }
        render_chart_c08_resonance_1_simple_fit();
    </script>
    <a onclick="window.fig_c08_resonance_1_simple_fit_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption>
        <strong><a name='figure_03'>Figure 3</a></strong>: The SDoF fit method's predicted tip deflections vs the measured tip deflections.
  </figcaption>
</figure>

[Figure 3](#figure_03) reveals our model is not a good fit for the measured tip deflections. The two lines share some resemblance, but our fit is far from satisfactory.

One thing our model does capture is the increase in amplitude around $n=560$. This suggests we are on the right track, but we need to refine our model further.

There are two shortcomings we need to fix. First, our model predicts zero tip deflection near the start and end of the resonance. Contrast this to the measured tip deflections, which seems to settle at -80 and -110 μm respectively. 

The most likely reason for this is because our data zeroing step pushed the "measured" tip deflections past its true zero point. The seemingly negative values may be an artifact of the zeroing process. We need to account for this in our model.

Second, our model *only* produces positive tip deflections. The measured data, however, shows a sudden negative dip around $n=568$. This means our model misses an important phenomenon. 

Why is our model flat when it should predict tip deflections along this sudden dip? 

Because we have made a terribly simplistic assumption concerning our *phase angle*. [Equation 3](#eq_3) is the phase angle between the forcing function and the response. This implicitly assumes the forcing function is zero at the start of each revolution. In practice, the phase angle of the forcing function is almost certainly unknown. We therefore need to solve the forcing function offset as well.

## Amplitude and phase offsets

We have diagnosed what's wrong with our model: it does not account for amplitude and phase offsets. How do we fix it? Let's start with the phase offset. This is a simple adjustment: we just add a constant term to the argument of the cosine function from [Equation 1](#eq_1). 

The new equation is given below:

<a name='eq_5'></a>

\begin{equation}
x(t) = X(\omega) \cos \left( \theta_s \cdot EO - \phi(\omega) + \phi_0 \right)
\end{equation}

where $\phi_0$ is our new phase offset. What are the possible values for $\phi_0$? Since the start of the forcing function is unknown, we can assume any value between 0 and $2 \pi$. That's our first improvement.

Next, we need to add an amplitude offset. This is more complicated. A reasonable attempt would be to just add a constant term to each probe's measurements. That would, however, imply our zeroing artifact at the start and the end of the resonance has the same value. No justification for this assumption exists.

Instead, we introduce a linearly-varying amplitude correction. The correction function is defined by two terms, $z_{\Omega_{\textrm{median}}}$ and $z_{\Omega_{\textrm{max}}}$. These terms are defined as follows:

1.  $z_{\Omega_{\textrm{median}}}$ is the amplitude correction at the median shaft speed in the resonance window.
2.  $z_{\Omega_{\textrm{max}}}$ is the amplitude correction at the maximum shaft speed in the resonance window.

From these two terms, we can calculate a correction function that varies linearly with shaft speed. 

The correction function is defined as follows:

\begin{equation}
z(\Omega) = m_z \cdot \Omega + b_z 
\end{equation}

\begin{equation}
m_z = \frac{z_{\Omega_{\textrm{max}}} - z_{\Omega_{\textrm{median}}}}{\Omega_{\textrm{max}} - \Omega_{\textrm{median}}}
\end{equation}

\begin{equation}
b_z = z_{\Omega_{\textrm{median}}} - m_z \cdot \Omega_{\textrm{median}}
\end{equation}

The terms $m_z$ and $b_z$ are the slope and intercept of the correction function. They depend on the correction factors, $z_{\Omega_{\textrm{median}}}$ and $z_{\Omega_{\textrm{max}}}$, and the median and maximum shaft speeds in the resonance window, $\Omega_{\textrm{median}}$ and $\Omega_{\textrm{max}}$.

!!! note "Correction function vs phase offset"
    
    We've added two new concepts: the phase offset term, $\phi_0$, and the correction factors, $z_{\Omega_{\textrm{median}}}$ and $z_{\Omega_{\textrm{max}}}$. There is a subtle difference between *where these two terms are applied*. 
    
    The phase offset term, $\phi_0$, is part of our SDoF vibration model. It shifts the entire vibration waveform such that $\phi$ represents the phase difference between the force and the tip deflection. The correction function, $z(\Omega)$, is *not part of the SDoF vibration model*. This correction factor is applied to the *measured tip deflections*. 

How do we choose the bounds for the correction factors? Hopefully, our optimization algorithm is smart enough to find the optimal solution without us constraining the correction factors too much. We therefore set the positive and negative bounds' size equal to the maximum absolute tip deflection in the resonance window.

The bounds for our new parameters are therefore:

``` py linenums="1"
phi_0_min = 0
phi_0_max = 2*np.pi
z_max = df_resonance_window["x_p1_filt"].abs().max()
z_min = -z_max
```

We rewrite our SDoF model as follows:

``` py linenums="1"
def predict_sdof_samples(
    omega_n : float,
    zeta : float,
    delta_st : float,
    EO : int,
    theta_sensor : float,
    phi_0 : float,
    arr_omega : np.ndarray
) -> np.ndarray:#(1)!
    X = get_X(arr_omega*EO, omega_n, zeta, delta_st)  
    phi = get_phi(arr_omega*EO, omega_n, zeta)
    predicted_tip_deflections = X * np.cos(theta_sensor * EO - phi + phi_0)
    return predicted_tip_deflections
```

1.  This function determined the predicted SDoF fit
    samples at a proximity probe given the SDoF parameters.

    Args:

        omega_n (float): The natural frequency of the SDoF system.
        
        zeta (float): The damping ratio of the SDoF system.
        
        delta_st (float): The static deflection of the SDoF system.
        
        phi_0 (float): The phase offset of the SDoF system.
        
        EO (int): The EO of vibration you want to fit.
        
        theta_sensor (float): The sensor's angular position on the rotor.
        
        phi_0 (float): The phase offset of the SDoF system.
        
        arr_omega (np.ndarray): The angular velocity of the rotor corresponding
            to each revolution for which we want to predict the SDoF samples.

    Returns:
        
        np.ndarray: The predicted SDoF samples.

Our new loss function is given below:

``` py linenums="1"
def get_correction_values(
    arr_omegas : float,
    z_median : float,
    z_max : float, 
) -> np.ndarray:#(1)!
    omega_median = np.median(arr_omegas)
    omega_max = np.min(arr_omegas)
    m = (
        z_max
        - z_median
    ) / (
        omega_max 
        - omega_median
    )
    b = z_median - m * omega_median
    correction_values = m * arr_omegas  + b
    return correction_values

def SDoF_loss(
        model_params : np.ndarray,
        arr_tip_deflections : np.ndarray,
        arr_omega : np.ndarray, 
        EO : int, 
        theta_sensor : float
) -> np.ndarray:#(2)!
    omega_n, ln_zeta, delta_st, phi_0, z_median, z_max = model_params
    zeta = np.exp(ln_zeta)
    predicted_tip_deflections = predict_sdof_samples(
        omega_n, zeta, delta_st, EO, theta_sensor, phi_0, arr_omega
    )
    arr_tip_deflection_corrections = get_correction_values(
        arr_omega, z_median, z_max
    )
    arr_tip_deflections_corrected = (
        arr_tip_deflections
        + arr_tip_deflection_corrections
    )
    return np.sum(
        (
            arr_tip_deflections_corrected
            - predicted_tip_deflections
        )**2
    )
```

1.  This function calculates the correction values for each sample
    based on the correction factors.

    Args:
        arr_omegas (float): The omega values for each sample.
        
        z_median (float): The correction value at the median shaft speed.
        
        z_max (float): The correction value at the max shaft speed.

    Returns:
        np.ndarray: The sample offsets for each sample.

2.  This function fits the SDoF parameters to a single 
    probe's tip deflection data.

    Args:
        model_params (np.ndarray): The SDoF fit method's model parameters. It
            includes a list of the following parameters:
            omega_n (float): The natural frequency of the SDoF system.
            ln_zeta (float): The damping ratio of the SDoF system.
            delta_st (float): The static deflection of the SDoF system.
            phi_0 (float): The phase offset of the SDoF system.
            z_median (float): The amplitude offset at the median shaft speed.
            z_max (float): The maximum amplitude offset.

        arr_tip_deflections (np.ndarray): The tip deflection data of the probe.
        
        arr_omega (np.ndarray): The angular velocity of the rotor corresponding
            to the tip deflection data.
        
        EO (int): The EO of vibration you want to fit.
        
        theta_sensor (float): The sensor's angular position on the rotor.

    Returns:
        np.ndarray: The sum of squared error between the tip deflection data

We've created a new function, `get_correction_values`. This function calculates the correction values for each sample based on the correction factors. The correction factors are the values that need to be determined during optimization.

## Solving the improved problem

We solve the improved problem with the code below:

``` py linenums="1"
improved_solution = differential_evolution(
    func = SDoF_loss,
    bounds=[
        (omega_n_min, omega_n_max),
        (ln_zeta_min, ln_zeta_max),
        (delta_st_min, delta_st_max),
        (phi_0_min, phi_0_max),
        (z_min, z_max),
        (z_min, z_max),
    ],
    args=(
        df_resonance_window[f'x_p1_filt'].values,
        df_resonance_window['Omega'].values,
        EO,
        df_resonance_window["AoA_p1"].median()
    ),
    seed=42
)
```

Once again, we can print the results:

``` py linenums="1"
print("ω_n      = ", improved_solution.x[0] / (2*np.pi), " Hz")
print("ζ        = ", np.exp(improved_solution.x[1]))
print("δ_st     = ", improved_solution.x[2], " µm")
print("φ_0      = ", improved_solution.x[3], " rad")
print("z_median = ", improved_solution.x[4], " µm")
print("z_max    = ", improved_solution.x[5], " µm")
```

``` console
ω_n      =  126.27747716213543  Hz
ζ        =  0.0026626951829294677
δ_st     =  3.6772997184546794  µm
φ_0      =  4.683877456932529  rad
z_median =  55.39278985745745  µm
z_max    =  134.43742560770318  µm
```

The values are in the same region as the previous solution, except for the static deflection and newly added terms, which were zero by implication in the previous solution.

The predicted vs true tip deflections are presented in [Figure 4](#figure_04) below.

<script src="c08_resonance_1_improved_fit.js" > </script>
<div>
    <div>
        <canvas id="c08_resonance_1_improved_fit"'></canvas>
    </div>
    <script>
        async function render_chart_c08_resonance_1_improved_fit() {
            const ctx = document.getElementById('c08_resonance_1_improved_fit');
            // If this is a mobile device, set the canvas height to 400
            if (window.innerWidth < 500) {
                ctx.height = 400;
            }
            while (typeof Chart == "undefined") {
                await new Promise(r => setTimeout(r, 1000));
            }
            Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
            window.fig_c08_resonance_1_improved_fit = new Chart(ctx, window.c08_resonance_1_improved_fit);
            window.fig_c08_resonance_1_improved_fit_reset = function resetZoomFig3() {
                    window.fig_c08_resonance_1_improved_fit.resetZoom();
                }
            }
        render_chart_c08_resonance_1_improved_fit();
    </script>
    <a onclick="window.fig_c08_resonance_1_improved_fit_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption>
        <strong><a name='figure_04'>Figure 4</a></strong>: The improved SDoF fit method's predicted tip deflections vs the corrected measured tip deflections.
  </figcaption>
</figure>

That's much better 👏. The model now captures the negative dip around $n=568$. The model also captures the tip deflection at the start and end of the resonance. The only critique is that the model seems to have a lower amplitude than the measured tip deflections where the resonance occurs. 

We could delve into a solution. Unfortunately, it makes for a beautiful coding exercise later on 😁. 

## Multiple probe solution

We have seen how to apply the SDoF fit method to a single probe's tip deflections. Now we want to extend this method to multiple probes. How do we do that?

The idea is similar to the single probe case. In the single probe case, we had one array of observed tip deflections. Now, we have a set of observed tip deflection arrays, where each array in the set corresponds to a different probe. We also have different $\theta_s$ values for each probe, depending on the circumferential spacing between the probes. 

We have the same number of *model parameters*, because the model parameters are independent of the number of probes. We must, however, introduce a set of *correction values* for each probe. Because of this, our optimization search space scales linearly with the number of probes.

Our new loss function is given in the code below.

``` py linenums="1"
def SDoF_loss_multiple_probes(
        model_params : np.ndarray,
        tip_deflections_set : List[np.ndarray],
        arr_omega : np.ndarray, 
        EO : int, 
        theta_sensor_set : List[float]
) -> np.ndarray:#(1)!
    omega_n, ln_zeta, delta_st, phi_0, *correction_factors = model_params#(2)!
    zeta = np.exp(ln_zeta)
    error = 0#(3)!
    for i_probe, arr_tip_deflections in enumerate(tip_deflections_set):#(4)!
        theta_sensor = theta_sensor_set[i_probe]
        predicted_tip_deflections = predict_sdof_samples(
            omega_n, zeta, delta_st, EO, theta_sensor, phi_0, arr_omega
        )
        z_median = correction_factors[i_probe*2]
        z_max = correction_factors[i_probe*2+1]#(5)!
        arr_tip_deflection_corrections = get_correction_values(
            arr_omega, z_median, z_max
        )
        arr_tip_deflections_corrected = (
            arr_tip_deflections
            + arr_tip_deflection_corrections
        )
        error += np.sum(
            (
                arr_tip_deflections_corrected
                - predicted_tip_deflections
            )**2
        )
    return error
```

1.  This function fits the SDoF parameters to 
        multiple probes' data.

    Args:
        model_params (np.ndarray): The SDoF fit method's model parameters. It
            includes a list of the following parameters:
            
            omega_n (float): The natural frequency of the SDoF system.
            
            ln_zeta (float): The damping ratio of the SDoF system.
            
            delta_st (float): The static deflection of the SDoF system.
            
            phi_0 (float): The phase offset of the SDoF system.
            
            And the z_median and z_max for each probe.
            
                z_median (float): The amplitude offset at the 
                    median shaft speed.
            
                z_max (float): The maximum amplitude offset.

        tip_deflections_set (List[np.ndarray]): The tip deflection data for each probe.
        
        arr_omega (np.ndarray): The angular velocity of the rotor corresponding
            to the tip deflection data.
        
        EO (int): The EO of vibration you want to fit.
        
        theta_sensor_set (List[float]): Each sensor's angular position 
            relative to the start of the revolution.

    Returns:
        
        np.ndarray: The sum of squared error between 
            the tip deflection data of each probe and
            the predicted tip deflections.

2.  This code unpacks the optimization algorithm model arguments, `model_params`, into the SDoF parameters and the correction factors. The last variable, `correction_factors`, has an asterisk, `*`, in front of it. This is Python's way of allocating *the rest* of `model_params` into `correction_factors`. With this notation, we can use the same loss function, regardless of how many probes we have.
3.  `error` are now initialized and will be incremented for each probe.
4.  We loop over every probe. Essentially, we do exactly what we did in the previous loss function for each probe.
5.  We grab the correction factors pertaining to this probe. The correction factors are located in `correction_factors` in the following order:

    ``` py
    correction_factors = [
        z_median_probe_1,
        z_max_probe_1,
        z_median_probe_2,
        z_max_probe_2,
        .
        .
        .,
        z_median_probe_n,
        z_max_probe_n,
    ]
    ```


The code below shows how to use the new loss function to solve the multiple probe problem:

``` py linenums="1"
PROBE_COUNT = 4
bounds = [
    (omega_n_min, omega_n_max),
    (ln_zeta_min, ln_zeta_max),
    (delta_st_min, delta_st_max),
    (phi_0_min, phi_0_max),
]
tip_deflections_set = []
theta_sensor_set = []
for i_probe in range(PROBE_COUNT):#(1)!
    z_max = df_resonance_window[f"x_p{i_probe+1}_filt"].abs().max()
    z_min = -z_max
    bounds.extend(
        [
            (z_min, z_max),
            (z_min, z_max)
        ]
    )
    tip_deflections_set.append(
        df_resonance_window[f"x_p{i_probe+1}_filt"].values
    )
    theta_sensor_set.append(
        df_resonance_window[f"AoA_p{i_probe+1}"].median()
    )

multiple_probes_solution = differential_evolution(
    func = SDoF_loss_multiple_probes,
    bounds=bounds,
    args=(
        tip_deflections_set,
        df_resonance_window['Omega'].values,
        EO,
        theta_sensor_set
    ),
    seed=42
)
```

1.  We loop over each probe and add the bounds, tip deflection array and the probe static position to the optimization problem.

There's a bit more effort involved with the setup of our bounds and arguments, but it is conceptually the same as the single probe case.

The results are shown below:

``` py linenums="1"
print("ω_n      = ", multiple_probes_solution.x[0] / (2*np.pi), " Hz")
print("ζ        = ", np.exp(multiple_probes_solution.x[1]))
print("δ_st     = ", multiple_probes_solution.x[2], " µm")
print("φ_0      = ", multiple_probes_solution.x[3], " rad")
for i_probe in range(PROBE_COUNT):
    print(f"z_median_{i_probe+1} = ", multiple_probes_solution.x[4+i_probe*2], " µm")
    print(f"z_max_{i_probe+1}    = ", multiple_probes_solution.x[5+i_probe*2], " µm")
```

``` console
ω_n      =  126.25099097038297  Hz
ζ        =  0.0026064673895625176
δ_st     =  3.687735366342961  µm
φ_0      =  4.833934260209016  rad
z_median_1 =  64.63941828855289  µm
z_max_1    =  132.82804180903605  µm
z_median_2 =  96.0353967585784  µm
z_max_2    =  37.229680208617246  µm
z_median_3 =  33.09137450249458  µm
z_max_3    =  93.7017772103181  µm
z_median_4 =  163.029855309319  µm
z_max_4    =  160.06562042963864  µm
```

Again, our model parameters are in the same region as the previous solution. The predicted vs true tip deflections are shown in [Figure 5](#figure_05) below. Each probe's fit is drawn on a different tab.

=== "Probe 1 fit"
    
    <div>
        <div>
            <canvas id="c08_resonance_multi_probe_1"></canvas>
        </div>
        <a onclick="window.fig_c08_resonance_multi_probe_1_reset()" class='md-button'>Reset Zoom</a>
    </div>

=== "Probe 2 fit"

    <div>
        <div>
            <canvas id="c08_resonance_multi_probe_2"></canvas>
        </div>
        <a onclick="window.fig_c08_resonance_multi_probe_2_reset()" class='md-button'>Reset Zoom</a>
    </div>

=== "Probe 3 fit"

    <div>
        <div>
            <canvas id="c08_resonance_multi_probe_3"></canvas>
        </div>
        <a onclick="window.fig_c08_resonance_multi_probe_3_reset()" class='md-button'>Reset Zoom</a>
    </div>

=== "Probe 4 fit"

    <div>
        <div>
            <canvas id="c08_resonance_multi_probe_4"></canvas>
        </div>
        <a onclick="window.fig_c08_resonance_multi_probe_4_reset()" class='md-button'>Reset Zoom</a>
    </div>

<script src="c08_resonance_multi_probe_1.js" > </script>
<script src="c08_resonance_multi_probe_2.js" > </script>
<script src="c08_resonance_multi_probe_3.js" > </script>
<script src="c08_resonance_multi_probe_4.js" > </script>
<script>
    async function render_chart_c08_multi_resonances() {
        const ctx_1 = document.getElementById('c08_resonance_multi_probe_1');
        const ctx_2 = document.getElementById('c08_resonance_multi_probe_2');
        const ctx_3 = document.getElementById('c08_resonance_multi_probe_3');
        const ctx_4 = document.getElementById('c08_resonance_multi_probe_4');

        // If this is a mobile device, set the canvas height to 400
        if (window.innerWidth < 500) {
            ctx_1.height = 400;
            ctx_2.height = 400;
            ctx_3.height = 400;
            ctx_4.height = 400;
        }
        while (typeof Chart == "undefined") {
            await new Promise(r => setTimeout(r, 1000));
        }
        Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
        window.fig_c08_resonance_multi_probe_1 = new Chart(ctx_1, window.c08_resonance_multi_probe_1);
        window.fig_c08_resonance_multi_probe_1_reset = function resetZoomMultiFig1() {
                window.fig_c08_resonance_multi_probe_1.resetZoom();
        }
        window.fig_c08_resonance_multi_probe_2 = new Chart(ctx_2, window.c08_resonance_multi_probe_2);
        window.fig_c08_resonance_multi_probe_2_reset = function resetZoomMultiFig2() {
                window.fig_c08_resonance_multi_probe_2.resetZoom();
        }
        window.fig_c08_resonance_multi_probe_3 = new Chart(ctx_3, window.c08_resonance_multi_probe_3);
        window.fig_c08_resonance_multi_probe_3_reset = function resetZoomMultiFig3() {
                window.fig_c08_resonance_multi_probe_3.resetZoom();
        }
        window.fig_c08_resonance_multi_probe_4 = new Chart(ctx_4, window.c08_resonance_multi_probe_4);
        window.fig_c08_resonance_multi_probe_4_reset = function resetZoomMultiFig4() {
                window.fig_c08_resonance_multi_probe_4.resetZoom();
        }
    }
    render_chart_c08_multi_resonances();
</script>

<figure markdown>
  <figcaption>
        <strong><a name='figure_05'>Figure 5</a></strong>: The improved SDoF fit method's predicted tip deflections vs the corrected measured tip deflections for each probe.
  </figcaption>
</figure>

We are happy with the fit for each probe. Our model parameters did not change much from the single probe case. This is exactly what we want to see. A model should be able to fit multiple probes with the same parameters.

## Estimating the EO

Until this point, we have assumed the EO=8. The optimization algorithm will, however, converge to a solution regardless of our choice for EO. Which EO is the correct one?

We can simply repeat the above exercise for multiple EOs and compare the objective function errors. The EO that results in the lowest error value is most likely the correct one.

The code below shows us how to do this:

``` py linenums="1"
PROBE_COUNT = 4
EOs = np.arange(1,17)
EO_solutions = []

for EO in EOs:
    print("NOW SOLVING FOR EO = ", EO)
    omega_n_min = df_resonance_window["Omega"].min() * EO
    omega_n_max = df_resonance_window["Omega"].max() * EO
    ln_zeta_min = np.log(0.0001)
    ln_zeta_max = np.log(0.3)
    delta_st_min = 0
    delta_st_max = 10
    phi_0_min = 0
    phi_0_max = 2*np.pi
    bounds = [
        (omega_n_min, omega_n_max),
        (ln_zeta_min, ln_zeta_max),
        (delta_st_min, delta_st_max),
        (phi_0_min, phi_0_max),
    ]
    tip_deflections_set = []
    theta_sensor_set = []
    for i_probe in range(PROBE_COUNT):
        z_max = df_resonance_window[f"x_p{i_probe+1}_filt"].abs().max()
        z_min = -z_max
        bounds.extend(
            [
                (z_min, z_max),
                (z_min, z_max)
            ]
        )
        tip_deflections_set.append(
            df_resonance_window[f"x_p{i_probe+1}_filt"].values
        )
        theta_sensor_set.append(
            df_resonance_window[f"AoA_p{i_probe+1}"].median()
        )

    multiple_probes_solution = differential_evolution(
        func = SDoF_loss_multiple_probes,
        bounds=bounds,
        args=(
            tip_deflections_set,
            df_resonance_window['Omega'].values,
            EO,
            theta_sensor_set
        ),
        seed=42
    )
    EO_solutions.append(multiple_probes_solution)
```

This loop takes approximately 1 minute and 30 seconds to run on my laptop. The optimal solution error function value for each EO can be accessed in the `.fun` attribute of each solution. 

<script src="c08_resonance_1_EO_selection.js" > </script>
<div>
    <div>
        <canvas id="c08_resonance_1_EO_selection"></canvas>
    </div>
    <script>
        async function render_chart_c08_resonance_1_EO_selection() {
            const ctx = document.getElementById('c08_resonance_1_EO_selection');
            // If this is a mobile device, set the canvas height to 400
            if (window.innerWidth < 500) {
                ctx.height = 400;
            }
            while (typeof Chart == "undefined") {
                await new Promise(r => setTimeout(r, 1000));
            }
            Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
            window.fig_c08_resonance_1_EO_selection = new Chart(ctx, window.c08_resonance_1_EO_selection);
            window.fig_c08_resonance_1_EO_selection_reset = function resetZoomFig4() {
                    window.fig_c08_resonance_1_EO_selection.resetZoom();
                }
            }
        render_chart_c08_resonance_1_EO_selection();
    </script>
    <a onclick="window.fig_c08_resonance_1_EO_selection_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption>
        <strong><a name='figure_06'>Figure 6</a></strong>: The objective function value for each EO.
  </figcaption>
</figure>

From [Figure 6](#figure_06) above, our initial guess was correct 💪! The optimal solution errors for each EO decreases gradually as we approach EO=8. EO=8 exhibits the smallest value. It is therefore reasonable to assume the true EO of vibration is 8.

## Conclusion
We've come a long way in this chapter. We've learned how to fit the SDoF fit parameters to BTT probe data with a global optimization function. We've also learned how to combine the data from multiple probes to get a more robust estimate of the model parameters.

But what's even more remarkable is how well our simple model works. With just four parameters, we can capture the vibration of the blade tip over multiple revolutions with high accuracy.

In the next and final chapter, we'll explore another way to infer the vibration parameters. This method, however, can be solved using linear algebra. It makes the solution faster and therefore more suitable for real-time analysis.

!!! question "Outcomes"

	:material-checkbox-marked:{ .checkbox-success .heart } Understand why we require an optimization function to determine the model parameters of the SDoF fit method.

	:material-checkbox-marked:{ .checkbox-success .heart } Understand why we need to specify upper and lower bounds for the model parameters.

    :material-checkbox-marked:{ .checkbox-success .heart } Understand why the SDoF model, in its raw form, is not adequate. We need to add phase and amplitude offsets to the objective function.

	:material-checkbox-marked:{ .checkbox-success .heart } Understand how we can loop over each probe's data to fit the SDoF model to multiple probes.
	
	:material-checkbox-marked:{ .checkbox-success .heart } Understand why the EO is treated differently than the other model parameters. Because the EO can only be a positive integer, we iterate over the EOs to identify the optimal EO.

## Acknowledgements
Thanks to <a href="https://www.linkedin.com/in/justin-s-507338116/" target="_blank">Justin Smith</a> and <a href="https://www.linkedin.com/in/alex-brocco-70218b25b/" target="_blank">Alex Brocco</a> for reviewing this chapter and providing feedback.

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
            2024-03-19
        </p>
    </div>
</div>

## Coding exercises

### 1. Getting the amplitude larger

We've seen throughout this chapter that our model's amplitude is not quite right. It's too low.

{==

:material-pencil-plus-outline: Why do you think the amplitude is constantly too low?

Can you write a new objective function that produces a larger $\delta_{st}$?

==}

??? example "Reveal answer (Please try it yourself before revealing the solution)"
    ``` py linenums="1"
    def SDoF_loss_multiple_probes(
            model_params : np.ndarray,
            tip_deflections_set : List[np.ndarray],
            arr_omega : np.ndarray, 
            EO : int, 
            theta_sensor_set : List[float],
            amplitude_scaling_factor : float = 1 #(1)!
        ) -> np.ndarray:
        """ This function fits the SDoF parameters to 
            multiple probes' data.

        Args:
            model_params (np.ndarray): The SDoF fit method's model parameters. It
                includes a list of the following parameters:
                
                omega_n (float): The natural frequency of the SDoF system.
                ln_zeta (float): The damping ratio of the SDoF system.
                delta_st (float): The static deflection of the SDoF system.
                phi_0 (float): The phase offset of the SDoF system.
                And then the z_median and z_max for each probe.
                    z_median (float): The amplitude offset at the 
                        median shaft speed.
                    z_max (float): The maximum amplitude offset.

            tip_deflections_set (List[np.ndarray]): The tip deflection data for each probe.
            arr_omega (np.ndarray): The angular velocity of the rotor corresponding
                to the tip deflection data.
            EO (int): The EO of vibration you want to fit.
            theta_sensor_set (List[float]): Each sensor's angular position 
                relative to the start of the revolution.
            amplitude_scaling_factor (float, optional): A scaling factor to
                weight the measured tip deflections. Defaults to 1. Use this value
                to reward solutions that better capture the full amplitude of the
                tip deflections.

        Returns:
            np.ndarray: The sum of squared error between 
                the tip deflection data of each probe and
                the predicted tip deflections.
        """
        omega_n, ln_zeta, delta_st, phi_0, *correction_factors = model_params
        zeta = np.exp(ln_zeta)
        probe_count = len(tip_deflections_set)
        error = 0
        for i_probe, arr_tip_deflections in enumerate(tip_deflections_set):    
            theta_sensor = theta_sensor_set[i_probe]
            predicted_tip_deflections = predict_sdof_samples(
                omega_n, zeta, delta_st, EO, theta_sensor, phi_0, arr_omega
            )
            z_median = correction_factors[i_probe*2]
            z_max = correction_factors[i_probe*2+1]
            arr_tip_deflection_corrections = get_correction_values(
                arr_omega, z_median, z_max
            )
            arr_tip_deflections_corrected = (
                arr_tip_deflections
                + arr_tip_deflection_corrections
            )
            error += np.sum(
                np.abs(arr_tip_deflections_corrected)**amplitude_scaling_factor #(2)!
                *(
                    arr_tip_deflections_corrected
                    - predicted_tip_deflections
                )**2
            )
        return error
    ```

    1.  This is a new parameter.
    2.  We use the new parameter to scale the error values according to the absolute size of the measured tip deflections. This rewards solutions that better capture the full amplitude of the tip deflections.

    Usage example:

    ``` py linenums="1"
    PROBE_COUNT = 4
    EO = 8

    omega_n_min = df_resonance_window["Omega"].min() * EO
    omega_n_max = df_resonance_window["Omega"].max() * EO
    ln_zeta_min = np.log(0.0001)
    ln_zeta_max = np.log(0.3)
    delta_st_min = 0
    delta_st_max = 10
    phi_0_min = 0
    phi_0_max = 2*np.pi

    bounds = [
        (omega_n_min, omega_n_max),
        (ln_zeta_min, ln_zeta_max),
        (delta_st_min, delta_st_max),
        (phi_0_min, phi_0_max),
    ]
    tip_deflections_set = []
    theta_sensor_set = []
    for i_probe in range(PROBE_COUNT):
        z_max = df_resonance_window[f"x_p{i_probe+1}_filt"].abs().max()
        z_min = -z_max
        bounds.extend(
            [
                (z_min, z_max),
                (z_min, z_max)
            ]
        )
        tip_deflections_set.append(
            df_resonance_window[f"x_p{i_probe+1}_filt"].values
        )
        theta_sensor_set.append(
            df_resonance_window[f"AoA_p{i_probe+1}"].median()
        )

    multiple_probes_solution = differential_evolution(
        func = SDoF_loss_multiple_probes,
        bounds=bounds,
        args=(
            tip_deflections_set,
            df_resonance_window['Omega'].values,
            EO,
            theta_sensor_set,
            2 #(1)!
        ),
        seed=42
    )
    ```

    1.  We set the `amplitude_scaling_factor` to 2. This means the error will be scaled by the square of the absolute value of the measured tip deflections. This rewards solutions that better capture the full amplitude of the tip deflections.

### 2. Writing a function we can use

We've covered a lot of ground in this chapter. Let's write a main entrypoint that performs the SDoF fit given a minimum number of arguments.

{==

:material-pencil-plus-outline: Write a function, called, `perform_SDoF_fit`, receiving the following three required arguments:

1.  The blade tip deflection DataFrame, `df_blade`.
2.  The revolution number indicating the start of the resonance, `n_start`.
3.  The revolution number indicating the end of the resonance, `n_end`.

The function should return the following values:

1.  The optimal solution, i.e. $\omega_n$, $\zeta$, $\delta_{st}$, and $\phi_0$
2.  The EO of vibration.

You may optionally accept other parameters to make the function more flexible.

==}

??? example "Reveal answer (Please try it yourself before revealing the solution)"
    ``` py linenums="1"
    def perform_SDoF_fit(
        df_blade : pd.DataFrame,
        n_start : int,
        n_end : int,
        EOs : List[int] = np.arange(1, 20),
        delta_st_max : int = 10,
        verbose : bool = False
    ) -> Dict[str, float]:
        """This function receives a blade tip deflection DataFrame, and returns 
        the SDoF fit model parameters after fitting.

        Args:
            df_blade (pd.DataFrame): The blade tip deflection DataFrame.
            n_start (int): The starting revolution number of the resonance 
                you want to fit.
            n_end (int): The ending revolution number of the resonance 
                you want to fit.
            EOs (List[int], optional): The list of EOs to search for. Defaults 
                to np.arange(1, 20).
            delta_st_max (int, optional): The maximum static deflection within our optimization 
                bounds. Defaults to 10.
            verbose (bool, optional): Whether to print the progress. Defaults to False.

        Returns:
            Dict[str, float]: The fitted model parameters.
        """
        df_resonance_window = df_blade.query(f"n >= {n_start} and n <= {n_end}")
        measured_tip_deflection_signals = [
            col 
            for col in df_resonance_window
            if col.endswith("_filt")
        ]
        PROBE_COUNT = len(measured_tip_deflection_signals)
        eo_solutions = []
        for EO in EOs:
            if verbose:
                print("NOW SOLVING FOR EO = ", EO, " of ", EOs)
            omega_n_min = df_resonance_window["Omega"].min() * EO
            omega_n_max = df_resonance_window["Omega"].max() * EO
            ln_zeta_min = np.log(0.0001)
            ln_zeta_max = np.log(0.3)
            delta_st_min = 0
            phi_0_min = 0
            phi_0_max = 2*np.pi
            bounds = [
                (omega_n_min, omega_n_max),
                (ln_zeta_min, ln_zeta_max),
                (delta_st_min, delta_st_max),
                (phi_0_min, phi_0_max),
            ]
            tip_deflections_set = []
            theta_sensor_set = []
            for i_probe in range(PROBE_COUNT):
                z_max = df_resonance_window[f"x_p{i_probe+1}_filt"].abs().max()
                z_min = -z_max
                bounds.extend(
                    [
                        (z_min, z_max),
                        (z_min, z_max)
                    ]
                )
                tip_deflections_set.append(
                    df_resonance_window[f"x_p{i_probe+1}_filt"].values
                )
                theta_sensor_set.append(
                    df_resonance_window[f"AoA_p{i_probe+1}"].median()
                )
            multiple_probes_solution = differential_evolution(
                func = SDoF_loss_multiple_probes,
                bounds=bounds,
                args=(
                    tip_deflections_set,
                    df_resonance_window['Omega'].values,
                    EO,
                    theta_sensor_set,
                    2
                ),
                seed=42
            )
            eo_solutions.append(multiple_probes_solution)
        best_EO_arg = np.argmin([solution.fun for solution in eo_solutions])
        best_EO = EOs[best_EO_arg]
        best_solution = eo_solutions[best_EO_arg]
        return {
            "omega_n" : best_solution.x[0] / (2*np.pi),
            "zeta" : np.exp(best_solution.x[1]),
            "delta_st" : best_solution.x[2],
            "phi_0" : best_solution.x[3],
            "EO" : best_EO,
        }
    ```

    Usage example:

    ``` py linenums="1"
    >>> SDoF_fit_parameters = perform_SDoF_fit(
        tip_deflection_dfs[0],
        n_start = 500,
        n_end = 600
    )
    ```

### 3. Comparing resonances on the upward and downward ramps

Throughout this chapter, we've focused on the EO=8 resonance on the ramp-up. We also have an EO=8 resonance on the ramp-down. Since nothing changed in the setup between the two resonances, we would expect the model parameters to be the same for both resonances.

{==

:material-pencil-plus-outline: Analyze the resonance on the downward ramp and compare the model parameters to the upward ramp.

Do the results make sense?

==}

??? example "Reveal answer (Please try it yourself before revealing the solution)"
    
    This is something you can do yourself. You're welcome to email me your results if you want me to check it.
