---
date: 2023-11-11
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
description: This chapter explains how we can use the Circumferential Fourier Fit method to estimate the vibration parameters of a rotor blade from BTT data.
robots: index, follow, Blade Tip Timing, BTT, Non Intrusive Stress Measurement, NSMS, Time of Arrival, Turbine blade,Mechanical Vibration
template: main_intro_to_btt.html
card_title: Intro to BTT Ch9 - Circumferential Fourier Fit Method
card_url: "ch9/"
---
# Circumferential Fourier Fit (CFF) Method

In the previous chapter, we learned how to use the SDoF fit method to infer the vibration properties of a blade during resonance. This method is based on a *physical model*. This means the equations we use are derived from the Equations of Motion of a harmonic oscillator.

The SDoF fit method, however, has a dark side: it is slow. It takes about 8 seconds to solve the equations for a single resonance in the previous chapter. This might not seem like a lot, but when we have many blades and resonances to analyze, it becomes impractical to do in real time. 

We need a faster way to estimate the vibration parameters for real time analysis. That's where the Circumferential Fourier Fit (CFF) method comes in. The CFF method is a phenomenological model, meaning it is only concerned with fitting the measured data well, not with being consistent with the underlying physics. 

In its most basic form, it fits a sinusoidal function to the measured data, using three coefficients: $A$, $B$, and $C$. The CFF method was first named as such, to my best knowledge, in [@joung2006analysis].

!!! question "Outcomes"

	:material-checkbox-blank-outline: Understand that the CFF method is a phenomenological method that fits a sinusoidal function to each revolution of the blade vibration.

	:material-checkbox-blank-outline: Understand how we can construct the CFF equations.

    :material-checkbox-blank-outline: Understand how the SDoF fit results and the CFF results can be compared to one another.
	
	:material-checkbox-blank-outline: Understand that we can iterate over many EOs and compare the sum of squared errors to identify the most likely EO.

## Derivation

The CFF method is similar to the SDoF method in that it assumes the blade tip exhibits sinusoidal vibration. The CFF method, however, completely ignores the modal properties, $\omega_n$, $\delta_{\text{st}}$, $\zeta$, and $\phi_0$. 

The CFF method calculates the vibration amplitude and phase directly for *each shaft revolution*. This results in linear equations, instead of the highly non-linear equations we had with the SDoF fit method.

The tip vibration of *in every shaft revolution* is expressed as follows:

<span id="eq_cff_fundamental_equation_blade_vibration"></span>

\begin{equation}
x(\theta_s) = A \sin(\theta_s \cdot EO) + B \cos(\theta_s \cdot EO) + C
\end{equation}

??? info "Symbols"
	| Symbol | Description | Unit| Domain|
	| :---: | :---: | :---: | :---: |
    | $x(\theta_s)$ | The tip deflection at the sensor located at $\theta_s$  | µm | $x(\theta_s) \in \mathbb{R}$ |
    | $\theta_s$ | The angular position of sensor $s$  | rad | $0 \leq \theta_s \leq 2\pi$ |
    | $A$ | The amplitude of the sine term  | µm | $A \in \mathbb{R}$ |
    | $B$ | The amplitude of the cosine term  | µm | $B \in \mathbb{R}$ |
    | $C$ | The offset of the tip deflection  | µm | $C \in \mathbb{R}$ |

We can write [Equation 1](#eq_cff_fundamental_equation_blade_vibration) for each sensor. This leads to a system of equations:

<span id="eq_cff_system_of_equations"></span>

\begin{align}
x(\theta_1) &= A \sin(\theta_1 \cdot EO) + B \cos(\theta_1 \cdot EO) + C \\
x(\theta_2) &= A \sin(\theta_2 \cdot EO) + B \cos(\theta_2 \cdot EO) + C \\
&\vdots \\
x(\theta_n) &= A \sin(\theta_n \cdot EO) + B \cos(\theta_n \cdot EO) + C
\end{align}

In the above equation, $\theta_1$ represents the position of probe 1, $\theta_2$ represents the position of probe 2, and so on. The variable $n$ represents the number of probes.

This system can be written in matrix form as follows:

<span id="eq_cff_matrix_form"></span>

\begin{equation}
\begin{bmatrix}
x(\theta_1) \\
x(\theta_2) \\
\vdots \\
x(\theta_n)
\end{bmatrix}
=
\begin{bmatrix}
\sin(\theta_1 \cdot EO) & \cos(\theta_1 \cdot EO) & 1 \\
\sin(\theta_2 \cdot EO) & \cos(\theta_2 \cdot EO) & 1 \\
\vdots & \vdots & \vdots \\
\sin(\theta_n \cdot EO) & \cos(\theta_n \cdot EO) & 1
\end{bmatrix}
\begin{bmatrix}
A \\
B \\
C
\end{bmatrix}
\end{equation}

We finally represent the system in a more compact form:

<span id="eq_cff_compact_form"></span>

\begin{equation}
\mathbf{x} = \mathbf{A} \mathbf{b}
\end{equation}

??? info "Symbols"
    | Symbol | Description | Unit| Domain|
    | :---: | :---: | :---: | :---: |
    | $\mathbf{x}$ | The vector of tip deflections at each sensor  | mm | $\mathbf{x} \in \mathbb{R}^n$ |
    | $\mathbf{A}$ | The matrix of the sine and cosine terms  | mm | $\mathbf{A} \in \mathbb{R}^{n \times 3}$ |
    | $\mathbf{b}$ | The vector of the vibration parameters  | mm | $\mathbf{b} \in \mathbb{R}^3$ |

We have now rephrased the problem as a linear algebra problem where we want to solve for $\mathbf{b}$. 

Once solved, we can calculate the amplitude and phase of the vibration within each revolution using the following equations:

<span id="eq_cff_amplitude"></span>

\begin{equation}
X = \sqrt{A^2 + B^2}
\end{equation}

<span id="eq_cff_phase"></span>

\begin{equation}
\phi = \arctan \left( \frac{A}{B} \right)
\end{equation}

The phase and amplitude can then be used in the following equation to calculate the tip deflections:

<span id="eq_cff_sinusoid"></span>

\begin{equation}
x(\theta_s) = X \sin(\theta_s \cdot EO - \phi) + C
\end{equation}

The above equation has exactly the same form as the SDoF fit method, only with a constant amplitude and phase within each revolution. 

!!! tip "Amplitude offset"
    The amplitude offset term, $C$, is generally kept as a part of the CFF model parameters. This stands in contrast to the SDoF fit method, where the correction factors are subtracted from the measured tip deflections before fitting the model.

## Following along

The worksheet for this chapter can be downloaded here <a href="https://github.com/Bladesight/bladesight-worksheets/blob/master/intro_to_btt/ch_09_worksheet.ipynb" target="_blank"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="Open In Github"/></a>.


You can open a Google Colab session of the worksheet by clicking here: <a href="https://colab.research.google.com/github/Bladesight/bladesight-worksheets/blob/master/intro_to_btt/ch_09_worksheet.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>.

You need to use one of the following Python versions to run the worksheet:
<img src="https://img.shields.io/badge/python-3.6-blue.svg">
<img src="https://img.shields.io/badge/python-3.7-blue.svg">
<img src="https://img.shields.io/badge/python-3.8-blue.svg">
<img src="https://img.shields.io/badge/python-3.9-blue.svg">
<img src="https://img.shields.io/badge/python-3.10-blue.svg">
<img src="https://img.shields.io/badge/python-3.11-blue.svg">

## Getting the dataset

To see the CFF method in action, we will use the same dataset that we applied the SDoF fit method to. This way, we can compare the results. The dataset contains blade tip displacements measured by a four sensors for a run-up and a run-down.

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
    np.cumsum(np.deg2rad(np.array([19.34, 19.34, 19.34]))),
    BLADE_COUNT
)
tip_deflection_dfs = []
for df_AoAs in rotor_blade_AoA_dfs:
    df_tip_deflections = get_blade_tip_deflections_from_AoAs(
        df_AoAs,
        RADIUS,
        11,
        2,
        0.5
    )
    tip_deflection_dfs.append(df_tip_deflections)
df_resonance_window = tip_deflection_dfs[0].query("n >= 500 and n <= 600")
EO = 8
```

## Single Revolution Case

The simplest implementation of the CFF method is given below:

``` py linenums="1"
def cff_method_single_revolution(
    df_blade : pd.DataFrame,
    theta_sensor_set : List[float],
    EO : int,
    signal_suffix : str = "_filt" #(1)!
) -> pd.DataFrame: #(2)!
    PROBE_COUNT = len(theta_sensor_set)
    tip_deflection_signals = [#(3)!
        f"x_p{i_probe + 1}{signal_suffix}" 
        for i_probe in range(PROBE_COUNT)
    ]
    theta_sensors = np.array(theta_sensor_set)
    A = np.ones((PROBE_COUNT, 3))#(4)!
    A[:, 0] = np.sin(theta_sensors * EO)
    A[:, 1] = np.cos(theta_sensors * EO)#(5)!
    
    A_pinv = np.linalg.pinv(A) #(6)!
    B = A_pinv.dot(
        df_blade.loc[:, tip_deflection_signals].values.T
    ) #(7)!
    df_cff = pd.DataFrame(B.T, columns=["A", "B", "C"]) #(8)!
    df_cff["X"] = np.sqrt(df_cff["A"]**2 + df_cff["B"]**2)
    df_cff["phi"] = np.arctan2(df_cff["A"], df_cff["B"])
    df_cff["n"] = df_blade["n"].values
    df_predicted_targets = pd.DataFrame(
        A.dot(B).T, 
        columns=[
            col + "_pred" 
            for col 
            in tip_deflection_signals
        ]
    ) #(9)!
    df_cff = pd.concat([df_cff, df_predicted_targets], axis=1)
    return df_cff
```

1.  Here, we specify the suffix of the tip deflection signals. We'll leave this as the default `"_filt"` most of the time. We may, however, set this to `""` in order to fit the CFF method on the unfiltered tip deflections.
2.  This function fits the CFF method for a resonance by 
    using a single revolution of data for each set of CFF parameters.

    Args:
        
        df_blade (pd.DataFrame): The dataframe containing the tip deflections.
        
        theta_sensor_set (List[float]): The sensor angles.
        
        EO (int): The Engine Order.
        
        signal_suffix (str, optional): The suffix of the tip deflection 
            signals. Defaults to "_filt".

    Returns:

        pd.DataFrame: A DataFrame containing the CFF parameters for 
            each shaft revolution. 

3.  We identify the signals that contain the tip deflections. This list is also used to determine how many probes were used for the present dataset.
4.  We initialize the matrix $\mathbf{A}$, which contain each sensor's sine, cosine and constant term coefficients. We initialize it as a `PROBE_COUNT X 3` matrix of ones, since the constant term is one throughout, and were going to override the other two columns.
5.  We calculate and assign the sine and cosine coefficients for each sensor.
6.  Here, we calculate the pseudo inverse of `A`. The pseudo inverse is a matrix that, when multiplied with `A`, results in the identity matrix. We only do this once, since the matrix `A` is constant for each revolution.
7.  This line is the reason why the CFF method is so fast. Because our `A` matrix is constant for each revolution, we can simply multiply its pseudo-inverse with the transpose of the observed tip deflections. This results in the CFF parameters for each revolution.
8.  From here to the end, its all cosmetic. We create a dataframe containing the CFF parameters, the amplitude, phase, and the shaft revolution number.
9.  We calculate the predicted tip deflections using the CFF parameters. This is for convenience so we don't need to recalculate it later to check our fits.

Implementing the method is simple:

``` py linenums="1"
%%timeit #(1)!
PROBE_COUNT = 4
df_cff_params = cff_method_single_revolution(
    df_resonance_window,
    [
        df_resonance_window[f"AoA_p{i_probe + 1}"].median()
        for i_probe in range(PROBE_COUNT)
    ],
    EO
)
```

1.  We use the `%%timeit` magic command to measure the execution time of the cell. This only works when you're in a Jupyter Notebook

You'll see that, once again, we use the median of the raw AoA values for the sensor location. 

The `%%timeit` command is something specific to Jupyter notebooks. It causes Python to run the entire cell multiple times. The mean execution time is then reported. This is useful for benchmarking code. This call runs for a total of... 3 ms! Contrast this to the SDoF fit method, which took approximately 8 seconds. The CFF method is therefore  ~2666 times faster than the SDoF fit method. This is a massive speed up ⏩!

Being faster is great, but does it actually work? Let's take a look at the predicted tip deflections vs the actual tip deflections:

=== "Probe 1 fit"
    
    <div>
        <div>
            <canvas id="c09_cff_simple_probe_1"></canvas>
        </div>
        <a onclick="window.fig_c09_cff_simple_probe_1_reset()" class='md-button'>Reset Zoom</a>
    </div>

=== "Probe 2 fit"

    <div>
        <div>
            <canvas id="c09_cff_simple_probe_2"></canvas>
        </div>
        <a onclick="window.fig_c09_cff_simple_probe_2_reset()" class='md-button'>Reset Zoom</a>
    </div>

=== "Probe 3 fit"

    <div>
        <div>
            <canvas id="c09_cff_simple_probe_3"></canvas>
        </div>
        <a onclick="window.fig_c09_cff_simple_probe_3_reset()" class='md-button'>Reset Zoom</a>
    </div>

=== "Probe 4 fit"

    <div>
        <div>
            <canvas id="c09_cff_simple_probe_4"></canvas>
        </div>
        <a onclick="window.fig_c09_cff_simple_probe_4_reset()" class='md-button'>Reset Zoom</a>
    </div>


<script src="c09_cff_simple_probe_1.js" > </script>
<script src="c09_cff_simple_probe_2.js" > </script>
<script src="c09_cff_simple_probe_3.js" > </script>
<script src="c09_cff_simple_probe_4.js" > </script>
<script>
    async function render_chart_c09_simple_cff() {
        const ctx_1 = document.getElementById('c09_cff_simple_probe_1');
        const ctx_2 = document.getElementById('c09_cff_simple_probe_2');
        const ctx_3 = document.getElementById('c09_cff_simple_probe_3');
        const ctx_4 = document.getElementById('c09_cff_simple_probe_4');

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
        window.fig_c09_cff_simple_probe_1 = new Chart(ctx_1, window.c09_cff_simple_probe_1);
        window.fig_c09_cff_simple_probe_1_reset = function resetZoomCFF_SimpleFig1() {
                window.fig_c09_cff_simple_probe_1.resetZoom();
        }
        window.fig_c09_cff_simple_probe_2 = new Chart(ctx_2, window.c09_cff_simple_probe_2);
        window.fig_c09_cff_simple_probe_2_reset = function resetZoomCFF_SimpleFig2() {
                window.fig_c09_cff_simple_probe_2.resetZoom();
        }
        window.fig_c09_cff_simple_probe_3 = new Chart(ctx_3, window.c09_cff_simple_probe_3);
        window.fig_c09_cff_simple_probe_3_reset = function resetZoomCFF_SimpleFig3() {
                window.fig_c09_cff_simple_probe_3.resetZoom();
        }
        window.fig_c09_cff_simple_probe_4 = new Chart(ctx_4, window.c09_cff_simple_probe_4);
        window.fig_c09_cff_simple_probe_4_reset = function resetZoomCFF_SimpleFig4() {
                window.fig_c09_cff_simple_probe_4.resetZoom();
        }
    }
    render_chart_c09_simple_cff();
</script>

<figure markdown>
  <figcaption>
        <strong><a name='figure_01'>Figure 1</a></strong>: The CFF predicted tip deflections vs the actual tip deflections for each probe. The CFF method is able to fit the tip deflections very well.
  </figcaption>
</figure>

Wow 👏! The CFF predicted values and the measured values are almost on top of one another. 

Contrast this with the SDoF method where we didn't get such a good fit. What do we conclude from this? Is the CFF method better than the SDoF fit method?

I don't think so. We have to remember that the CFF method is a phenomenological method. It has 3 parameters and 4 measured values, therefore only slightly overdetermined. The odds are therefore stacked in its favor to fit the data well. In fact, the CFF method even fits the data well near the start of the resonance, where we mostly have noise. This should make us uneasy. If our model reliably reproduces noise, it means we probably have overfitting.

Now for the big question, what is our natural frequency? The simple answer is that we don't have one! Our CFF is not concerned with the underlying physics, so it doesn't have a natural frequency. We can, however, take the shaft speed where the fitted amplitude is a maximum.

Let's compare the CFF amplitude and phase to the SDoF fit amplitude and phase for the same resonance:

=== "Amplitude"
    
    <div>
        <div>
            <canvas id="c09_cff_simple_amp"></canvas>
        </div>
        <a onclick="window.fig_c09_cff_simple_amp_reset()" class='md-button'>Reset Zoom</a>
    </div>

=== "Phase"

    <div>
        <div>
            <canvas id="c09_cff_simple_phase"></canvas>
        </div>
        <a onclick="window.fig_c09_cff_simple_phase_reset()" class='md-button'>Reset Zoom</a>
    </div>

<script src="c09_cff_simple_amp.js" > </script>
<script src="c09_cff_simple_phase.js" > </script>
<script>
    async function render_chart_c09_simple_cff_amp_phase() {
        const ctx_amp = document.getElementById('c09_cff_simple_amp');
        const ctx_phase = document.getElementById('c09_cff_simple_phase');

        // If this is a mobile device, set the canvas height to 400
        if (window.innerWidth < 500) {
            ctx_amp.height = 400;
            ctx_phase.height = 400;
        }
        while (typeof Chart == "undefined") {
            await new Promise(r => setTimeout(r, 1000));
        }
        Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
        window.fig_c09_cff_simple_amp = new Chart(ctx_amp, window.c09_cff_simple_amp);
        window.fig_c09_cff_simple_amp_reset = function resetZoomCFF_SimpleFigAmp() {
                window.fig_c09_cff_simple_amp.resetZoom();
        }
        window.fig_c09_cff_simple_phase = new Chart(ctx_phase, window.c09_cff_simple_phase);
        window.fig_c09_cff_simple_phase_reset = function resetZoomCFF_SimpleFigPhase() {
                window.fig_c09_cff_simple_phase.resetZoom();
        }
    }
    render_chart_c09_simple_cff_amp_phase();
</script>

<figure markdown>
  <figcaption>
        <strong><a name='figure_02'>Figure 2</a></strong>: The CFF amplitude and phase vs the SDoF fit amplitude and phase for the same resonance. We see the maximum amplitude between the two methods differ by approximately 200 μm. The phase, at least in the resonance region, is basically identical.
  </figcaption>
</figure>

From [Figure 2](#figure_02) above, we see that the maximum amplitude of the CFF method occurs at revolution number 568. We can use the rotor speed in this resonance, multiplied by the EO, to get the CFF natural frequency:

``` py linenums="1"
>>> omega_n_568 = df_resonance_window.query("n == 568")["Omega"].iloc[0]*EO
>>> omega_n_567 = df_resonance_window.query("n == 567")["Omega"].iloc[0]*EO
>>> print("CFF omega_n @ n=568: {:.3f} Hz".format(omega_n_568 / (2*np.pi)))
>>> print("CFF omega_n @ n=567: {:.3f} Hz".format(omega_n_567 / (2*np.pi)))
>>> print("SDoF omega_n       : {:.3f} Hz".format(SDoF_params["omega_n"]))
```

``` console
CFF omega_n @ n=568: 126.565 Hz
CFF omega_n @ n=567: 126.273 Hz
SDoF omega_n       : 126.270 Hz
```

Wee see that the CFF natural frequency at revolution 568 is 0.3 Hz higher than the SDoF natural frequency. This is a difference of 0.2%. I believe it is a small difference, showing correspondence between the two methods. We also used the shaft speed at the previous revolution, n=567, to calculate the natural frequency. Now, the natural frequency is almost identical to the SDoF natural frequency.

We see that the CFF method's maximum amplitude is approximately 200 μm lower than the SDoF method's maximum amplitude. This difference is large, approximately 28%. It is possible that the SDoF fit method may be overestimating the amplitude. If you did the coding exercise in the previous chapter, you would have implemented an SDoF fit method that rewards the ability to capture larger amplitudes.

When looking at the phase plot, we see that the two methods produce similar phase shifts where the amplitudes are largest. The CFF method produces larger phase shifts outside the resonance region.

Which one of the two is more accurate? I don't know. More effort needs to be put in to compensate for the zeroing artifacts in the CFF method. The only real way to judge which one is better is to have calibrated strain gauge data available. This falls outside the scope of the present tutorial.

## Estimating the EO
We have seen how the CFF method can fit the data well, as long as we know the correct EO. But what if we are not sure about the EO? How can we find out which one is the best for our data?

We can repeat what we did in the previous chapter. We can try different EO values and calculate the sum of squared errors between the predicted and the measured tip deflections. The lower the error, the better the fit. So, we can look for the EO that gives us the lowest error value. That is likely to be the correct one for our data.

The below code performs this calculation:

``` py linenums="1"
PROBE_COUNT = 4
EOs = np.arange(1, 17)
errors = []
for EO in EOs:
    df_cff_params = cff_method_single_revolution(
        df_resonance_window,
        [
            df_resonance_window[f"AoA_p{i_probe + 1}"].median()
            for i_probe in range(PROBE_COUNT)
        ],
        EO
    )
    error = 0
    for i_probe in range(PROBE_COUNT):
        error += np.sum(
            (
                df_cff_params[f"x_p{i_probe+1}_filt_pred"].values 
                - df_resonance_window[f"x_p{i_probe+1}_filt"].values
            )**2
        )
    errors.append(error)
print("Most likely EO:", EOs[np.argmin(errors)])
```

``` console
Most likely EO: 8
```

The error values are plotted in [Figure 3](#figure_03) below.

<div>
    <div>
        <canvas id="c09_resonance_1_EO_selection"></canvas>
    </div>
    <a onclick="window.fig_c09_resonance_1_EO_selection_reset()" class='md-button'>Reset Zoom</a>
</div>

<script src="c09_resonance_1_EO_selection.js" > </script>
<script>
    async function render_chart_c09_resonance_1_EO_selection() {
        const ctx = document.getElementById('c09_resonance_1_EO_selection');
        // If this is a mobile device, set the canvas height to 400
        if (window.innerWidth < 500) {
            ctx.height = 400;
        }
        while (typeof Chart == "undefined") {
            await new Promise(r => setTimeout(r, 1000));
        }
        Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
        window.fig_c09_resonance_1_EO_selection = new Chart(ctx, window.c09_resonance_1_EO_selection);
        window.fig_c09_resonance_1_EO_selection_reset = function resetZoomCFF_SimpleFig1() {
                window.fig_c09_resonance_1_EO_selection.resetZoom();
        }
    }
    render_chart_c09_resonance_1_EO_selection();
</script>
<figure markdown>
  <figcaption>
        <strong><a name='figure_03'>Figure 3</a></strong>: The sum of squared errors between the predicted and measured tip deflections for each EO. We can see that EO=8 results in the lowest sum of squared error value.
  </figcaption>
</figure>

From [Figure 3](#figure_03) above, we see that the EO with the lowest error value is 8. This is the correct value. You may need to zoom in a bit, since the error values for EOs 8 - 11 are close to one another.

## Conclusion
In this chapter, we have shown how to apply the CFF method to a resonance event. The CFF method is a powerful tool that can be used to fit the blade tip deflection data with high accuracy. We can also estimate the natural frequency and EO of vibration using it.

The main benefit of the CFF method is its speed. The CFF method is, in its rawest form, approximately 2666 times faster than the SDoF fit method. You should therefore be able to use it in real-time.

!!! question "Outcomes"

	:material-checkbox-marked:{ .checkbox-success .heart } Understand that the CFF method is a phenomenological method that fits a sinusoidal function to each revolution of the blade vibration.

	:material-checkbox-marked:{ .checkbox-success .heart } Understand how we can construct the CFF equations.

    :material-checkbox-marked:{ .checkbox-success .heart } Understand how the SDoF fit results and the CFF results can be compared to one another.
	
	:material-checkbox-marked:{ .checkbox-success .heart } Understand that we can iterate over many EOs and compare the sum of squared errors to identify the most likely EO.


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
            2023-11-11
        </p>
    </div>
</div>

## Coding exercises

### 1. Multiple Revolution Case
Solving the CFF parameters on a per revolution basis is great, but it may be prone to noise and outliers. In our example above, we have 3 unknown parameters and only 4 measurements. The system may be overdetermined, but we'd ideally like to have even more measurements to increase robustness to noise.

We can change to problem such that the parameters $A$, $B$, and $C$ are fit across *multiple revolutions*. This should increase the robustness of the fit.

{==

:material-pencil-plus-outline: Write a new function, `cff_method_multiple_revolutions` that receives the following arguments:

1.  `df_blade` : The DataFrame containing the tip deflections. 
2.  `theta_sensor_set` : The sensor angles.
3.  `EO` : The Engine Order.
4.  `extra_revolutions` : The number of revolutions before and after each "center" revolution to fit.
5.  `signal_suffix` (str, optional): The suffix of the tip deflection signals. Defaults to "_filt". 

The function should fit the CFF parameters over `1 + 2*extra_revolutions` consecutive revolutions. In other words, we should still receive one CFF set of parameters per revolution, but the values of $A$, $B$, and $C$ should be fit over 1 + 2*extra_revolutions revolutions.

==}

??? example "Reveal answer (Please try it yourself before peeking)"
    ``` py linenums="1"
    def cff_method_multiple_revolutions(
        df_blade : pd.DataFrame,
        theta_sensor_set : List[float],
        EO : int,
        extra_revolutions : int,
        signal_suffix : str = "_filt" 
    ) -> pd.DataFrame:
        """ This function fits the CFF method for a resonance by
        using multiple revolutions of data for each set of CFF parameters.

        Args:
            df_blade (pd.DataFrame): The dataframe containing the tip deflections.
            theta_sensor_set (List[float]): The sensor angles.
            EO (int): The Engine Order.
            extra_revolutions (int): The number of revolutions to use for the fit.
            signal_suffix (str, optional): The suffix of the tip deflection 
                signals. Defaults to "_filt".

        Returns:
            pd.DataFrame: A DataFrame containing the CFF parameters for 
                each shaft revolution.
        """
        PROBE_COUNT = len(theta_sensor_set)
        tip_deflection_signals = [
            f"x_p{i_probe + 1}{signal_suffix}" 
            for i_probe in range(PROBE_COUNT)
        ]
        theta_sensors = np.array(theta_sensor_set)

        A = np.ones((PROBE_COUNT*(2*extra_revolutions+1), 3))
        arr_multiple_thetas = np.array(
            list(theta_sensors)*(2*extra_revolutions+1)
        )
        A[:, 0] = np.sin(arr_multiple_thetas * EO)
        A[:, 1] = np.cos(arr_multiple_thetas * EO)
        A_pinv = np.linalg.pinv(A)
        new_obs_rows = df_blade.shape[0] - 2*extra_revolutions
        X_multiple_revos = np.zeros(
            (
                new_obs_rows, 
                PROBE_COUNT*(2*extra_revolutions+1)
            )
        )
        for n_revo in range(-extra_revolutions, extra_revolutions+1):
            for i_probe in range(PROBE_COUNT):
                mat_aoas_start = extra_revolutions + n_revo
                mat_aoas_end = mat_aoas_start + new_obs_rows
                i_col = i_probe + n_revo*PROBE_COUNT + extra_revolutions*PROBE_COUNT
                X_multiple_revos[:,i_col] = (
                    df_blade.iloc[mat_aoas_start:mat_aoas_end][tip_deflection_signals[i_probe]]
                )
        B = A_pinv.dot(X_multiple_revos.T)
        B_full = np.zeros((df_blade.shape[0], 3))
        B_full[extra_revolutions:-extra_revolutions, :] = B.T
        B_full[:extra_revolutions, :] = B_full[extra_revolutions, :]
        B_full[-extra_revolutions:, :] = B_full[-extra_revolutions-1, :]

        df_cff = pd.DataFrame(B_full, columns=["A", "B", "C"])
        df_cff["X"] = np.sqrt(df_cff["A"]**2 + df_cff["B"]**2)
        df_cff["phi"] = np.arctan2(df_cff["A"], df_cff["B"])
        df_cff["n"] = df_blade["n"].values
        target_matrix = (A.dot(B_full.T)).T
        predicted_deflections = target_matrix[:, extra_revolutions*PROBE_COUNT:(extra_revolutions+1)*PROBE_COUNT] 
        df_predicted_targets = pd.DataFrame(
            predicted_deflections, 
            columns=[
                col + "_pred" 
                for col 
                in tip_deflection_signals
            ]
        )
        df_cff = pd.concat([df_cff, df_predicted_targets], axis=1)
        return df_cff
    ```

    Usage example:

    ``` py linenums="1"
    >>> df_cff_params = cff_method_multiple_revolutions(
        df_resonance_window,
        [
            df_resonance_window[f"AoA_p{i_probe + 1}"].median()
            for i_probe in range(PROBE_COUNT)
        ],
        EO,
        2
    )
    ```

### 2. Writing a function we can use

Were now going to write a single entrypoint, receiving the minimum amount of arguments, that we can use to fit the CFF method and estimate the EO. This will make it easier to use the CFF method in the future.

{==

:material-pencil-plus-outline: Write a function, called, `perform_CFF_fit`, that receives the following three required arguments:

1.  The blade tip deflection DataFrame, `df_blade`.
2.  The revolution number indicating the start of the resonance, `n_start`.
3.  The revolution number indicating the end of the resonance, `n_end`.

The function should return the following values:

1.  The CFF parameters for each revolution.
2.  The EO of vibration.

You may optionally accept other parameters to make the function more flexible.

==}

??? example "Reveal answer (Please try it yourself before peeking)"
    ``` py linenums="1"
    def perform_CFF_fit(
        df_blade : pd.DataFrame,
        n_start : int,
        n_end : int,
        EOs : List[int] = np.arange(1, 20),
        extra_revolutions : int = 1
    ) -> Dict[str, Union[pd.DataFrame, int]]:
        """ This function performs the CFF method fit for a resonance. The function
        iterates over EOs and selects the EO that gives the lowest sum of squared
        errors between the measured and predicted tip deflections.

        Args:
            df_blade (pd.DataFrame): The dataframe containing the tip deflections.
            n_start (int): The start revolution number.
            n_end (int): The end revolution number.
            EOs (List[int], optional): The EOs to consider for this resonance. Defaults 
                to np.arange(1, 20).
            extra_revolutions (int, optional): How many extra revolutions to use for 
                the fit. Defaults to 1.

        Returns:
            Dict[str, Union[pd.DataFrame, int]]: A dictionary containing the CFF 
                parameters and the selected EO.
        """
        PROBE_COUNT = len(
            [
                col 
                for col in df_blade.columns
                if col.endswith("_filt")
            ]
        )
        theta_sensor_set = [
            df_blade[f"AoA_p{i_probe + 1}"].median()
            for i_probe in range(PROBE_COUNT)
        ]
        errors = []
        df_resonance_window = df_blade.query(f"n >= {n_start} and n <= {n_end}")
        for EO in EOs:
            df_cff_params = cff_method_multiple_revolutions(
                df_resonance_window,
                theta_sensor_set,
                EO,
                extra_revolutions
            )
            error = 0
            for i_probe in range(PROBE_COUNT):
                error += np.sum(
                    (
                        df_cff_params[f"x_p{i_probe+1}_filt_pred"].values 
                        - df_resonance_window[f"x_p{i_probe+1}_filt"].values
                    )**2
                )
            errors.append(error)
        EO = EOs[np.argmin(errors)]
        df_cff_params = cff_method_multiple_revolutions(
            df_resonance_window,
            theta_sensor_set,
            EO,
            extra_revolutions
        )
        return {
            "df_cff_params" : df_cff_params,
            "EO" : EO
        }
    ```

    Usage example:

    ``` py linenums="1"
    cff_params = perform_CFF_fit(
        tip_deflection_dfs[0],
        500,
        600
    )
    ```
