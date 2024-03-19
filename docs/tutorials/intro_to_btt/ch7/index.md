---
date: 2024-03-12
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
description: This chapter explains the physics behind rotor blade vibration and how we sample it.
robots: index, follow, Blade Tip Timing, BTT, Non Intrusive Stress Measurement, NSMS, Time of Arrival, Turbine blade,Mechanical Vibration
template: main_intro_to_btt_ch7.html
card_title: Intro to BTT Ch7 - Synchronous Vibration and Sampling
card_url: "ch7/"
---
??? abstract "You are here"
	<figure markdown>
	![BTT Tutorial Roadmap](BTT_Tutorial_Outline_Ch7.svg){ width="500" }
	</figure>

# Synchronous Vibration and Sampling
The previous chapters presented new signal processing steps along our journey to infer blade vibration, frequency, and phase from raw timestamps. In this chapter, we slow down a bit. 

Blades do not simply vibrate for no reason. Something causes it. 

In this chapter, we discuss *why blades are vibrating* instead of *how to process the signals*. This chapter will help us understand what the results mean.

We delve into the fundamentals of Single Degree of Freedom (SDoF) vibration. Most graduate students would have already encountered SDoF vibration in their studies. We, however, explain it through the lens of BTT. Hopefully, spending time on the fundamentals through a different lens will help you understand the subject better.

!!! question "Outcomes"

	:material-checkbox-blank-outline: Understand how obstructions in the flow path causes synchronous vibration.

	:material-checkbox-blank-outline: Understand that synchronous vibration occurs at an integer multiple of the shaft speed. This integer multiple is called the Engine Order (EO). 

    :material-checkbox-blank-outline: Understand what a Campbell diagram is and how we can use it to guess the mode associated with a resonance event we measured.

	:material-checkbox-blank-outline: Understand that we can use a Single Degree of Freedom (SDoF) oscillator to model a blade's vibration.
	
	:material-checkbox-blank-outline: Understand how BTT systems sample the blade's vibration waveform.

    :material-checkbox-blank-outline: Understand why synchronous vibration data is more difficult to process than asynchronous vibration data.

	:material-checkbox-blank-outline: Understand that BTT signals are generally aliased.

## Two kinds of vibration
This chapter focuses on a kind of vibration called *synchronous vibration*. We will, however, briefly categorize the two kinds of vibration here.

Here's a high-level overview of the two kinds of vibration:

1.  Obstruction-driven vibration, or *synchronous* vibration.

2.  Fluid-structure interaction-driven, or *asynchronous* vibration.

[Figure 1](#figure_01) below illustrates the respective driving mechanisms of two kinds of vibration.

<figure markdown>
  ![Two kinds of vibration](Ch7_why_vibrate.svg){ width="700" }
  <figcaption><strong><a name='figure_01'>Figure 1</a></strong>: The two kinds of driving mechanisms causing blade vibration. In A) the flow path is obstructed by struts, causing synchronous vibration. In B) the blade is vibrating in response to the fluid, and the fluid's pressure is oscillating because of the blade's vibration.
  </figcaption>
</figure>

In [Figure 1](#figure_01) A) there are 5 struts in the flow path upstream from the rotor. The struts obstruct the fluid, causing pressure fluctuations downstream. These pressure fluctuations occur *once per revolution for each obstruction*. This statement is so important, I've put it in a box 👇.

!!! success  "Obstruction-driven fluctuations cause synchronous vibration"
    
    The pressure fluctuations caused by obstructions occur once per revolution for each obstruction. If there are 5 obstructions, the pressure fluctuations occur 5 times per revolution. It is not possible to have 4.5 pressure fluctuations per revolution, just like you cannot have 2.4 children. 
    
    This kind of vibration is therefore called "synchronous vibration", because it occurs at an integer multiple of the shaft speed. This integer multiple is called the Engine Order (EO).

In [Figure 1](#figure_01) B) a blade's airfoil is shown as the fluid passes. On one of the fluid lines, I've added a conceptual spring and damper. It is supposed to convey the idea that the blade is vibrating because of the fluid, and the fluid is vibrating because of the structure. These two systems feed off one another, and can result in significant vibration. This kind of vibration is called "asynchronous vibration", because it does not occur at an integer multiple of the shaft speed, it is dependant on the fluid and the blade's structural properties.

!!! warning "Asynchronous vibration is not well understood... by me"

    Please do not take my explanation above as authoritative. I am not an expert in fluid-structure interaction, Computational Fluid Dynamics (CFD), or flutter. BTT is concerned with measuring the vibrations, regardless of what caused them. 

    If you are an expert in this field and you would like to contribute to this tutorial, please get in touch.😀

    PS. Obviously, I do not believe there is anything close to a linear relationship between the blade's vibration and the fluid's pressure.

This chapter focuses on synchronous vibration because it is more difficult to measure than asynchronous vibration. We'll explain why this is the case later in the chapter. 

For now, let's focus on synchronous vibration.

## Synchronous vibration
Synchronous vibration is defined as vibration occurring at an *integer multiple* of the shaft speed. This integer multiple is called the Engine Order (EO). 

The relationship between the excitation frequency and the shaft speed is given below:

$$
f = \Omega \cdot EO
$$
<figure markdown>
  <figcaption><strong><a name='equation_01'>Equation 1</a></strong></figcaption>
</figure>

??? info "Symbols"
	| Symbol | Description |
	| :---: | :--- |
    | $f$ | Excitation frequency experienced by a blade [rad/s] |
    | $\Omega$ | Shaft speed [rad/s] |
    | $EO$ | Engine Order  |

    The domains of the variables are:
    
    $$
    \begin{align}
    f &\gt 0 \\
    \Omega & \gt 0 \\
    EO & \in [1,2,3...] \\
    \end{align}
    $$

Synchronous vibration can therefore only take on a discrete set of values. If, for instance, the rotor speed is 3000 RPM (50 Hz), the possible excitation frequencies are:

<figure markdown>
  <figcaption><strong><a name='table_01'>Table 1</a></strong></figcaption>
  </figcaption>
</figure>

| EO | $f$ [Hz]|
| :---: | :---: |
| 1 |  50 |
| 2 |  100 |
| 3 |  150 |
| 4 |  200 |
| 5 |  250 |
| ... | ...|

Asynchronous vibration occurs when there is a non-integer relationship between the shaft speed and the excitation frequency. We typically do not express asynchronous vibration as a non-integer EO, but would simply report the natural frequency without relation to the shaft speed. 

## At which frequency are blades excited?
A popular textbook [@rao1991turbomachine] on rotor blade vibration offers one explanation. It is said that the main source of dangerous excitations are the stator vanes. Though the premise is sound, lets conduct a thought experiment to weigh the theory.

The number of stator vanes multiplied by the shaft speed gives us the excitation frequency, also called the Nozzle Passing Frequency (NPF). The presence of stator vanes definitely cause obstruction-driven vibration, but does it actually cause damage? Damage can only accrue if a blade is responding at one of its natural frequencies.

??? info "Why can damage only occur when blades are in resonance?"

    In short, if a blade's design allows for fatigue damage to accumulate in normal operating conditions, I do not want to be in the room when that blade goes into resonance. It is a bad design. The engineers that design blades know what they are doing, and they design blades to be safe.

Structures theoretically have an infinite number of natural frequencies. However, the first natural frequencies typically have the least damping associated with them, and the highest frequency response function (FRF) amplitudes. It is a generally accepted practice to disregard all modes except the first few when modal analysis is performed. The first few modes are therefore the most likely to cause damage.

Let's consider a rotor with 80 blades, and half the amount of stator blades, 45. The rotor runs at 3000 RPM. The NPF is therefore:

$$
NPF = \frac{3000}{60} \times 45 \approx 2250 \text{ Hz}
$$

Even though this is a thought experiment and we don't have real natural frequencies to compare the NPF to, 2250 Hz is a suspiciously high frequency to be among the first few modes. This is well above the typical range within which the lower natural frequencies of large blades occur. In my work, I'm interested in natural frequencies well below 2000 Hz. 

!!! note "Note"
    Blade natural frequencies are not usually made public by blade manufacturers. No doubt there are exceptions to the 2000 Hz cutoff used above. In my experience, however, this is a good rule of thumb. The first natural frequency of rotor blades generally occur far below 2250 Hz.

!!! note "What is the highest frequency you can measure with BTT?"

    BTT signals are inherently aliased. You'll read about this later in the chapter. If you are designing an experimental setup and would like to calculate the positions for your probes, you can check out my paper about it [@diamond2018novel]. This space seems to have received attention recently. Here's a link to the Google Scholar page of citing articles to my paper: <a href="https://scholar.google.com/scholar?cites=10600910125193632917&as_sdt=2005&sciodt=0,5&hl=en" target="_blank">Google Scholar</a>.

The vibrations caused by the stator vanes are therefore not the culprit.

How, then, does damaging vibration arise?

??? warning "Do your own research"

    I have said here that Nozzle Passing Frequencies (NPF) are not the main culprits for blade vibration. This is my take on it, based on my experience and understanding of vibration.
    
    There are, however, papers and sources claiming the opposite. A recent example that claims NPF is - in fact - a key concern, is Chapter 7 of the book "Forsthoffer's Proven Guidelines for Rotating Machinery Excellence" [@forsthoffer2021forsthoffer]. Another recent source does, however, share my sentiment that non NPF-related phenomena are the main culprits [@tanuma2022advances]. 
    
    The statements attributed to these books can be found here: <a href="https://www.sciencedirect.com/topics/engineering/campbell-diagram" target="_blank">Science Direct: Campbell Diagram</a>.

    Please do your own research and come to your own conclusions. If you have a different take on this, please get in touch. I would love to include your take on this tutorial.

## A simple forcing function
The aerodynamic behavior inside a turbomachine is a complex discipline. We will not attempt to explain how specific aerodynamic flow patterns arise. Instead, we rather discuss why it does not take much to cause excitations at the lowest EOs.

Let's suppose we have a turbomachine afflicted with a *single discontinuity* in the working fluid's flow path upstream of the blades. The discontinuity could be a supporting structural element, such as a strut. The discontinuity will cause a *pressure fluctuation* downstream of it. As the blades rotate, they pass through this pressure fluctuation **once every revolution**. This, in turn, causes a force to be exerted on the blade **once every revolution**.

Let's model this forcing function as a *unit impulse* when the blade is in the path of the discontinuity, and zero otherwise. We'll assume the downstream effects of the discontinuity occur between $\frac{2 \pi}{10}$ and $\frac{3 \pi}{10}$ radians. The forcing function can be expressed as:

$$
f(\theta) = \begin{cases}
1 & \text{if }  \frac{2 \pi}{10} \lt \theta \leq \frac{3 \pi}{10} \\
0 & \textrm{else} \\
\end{cases}
$$

We plot the forcing function experienced by each blade over multiple shaft revolutions in [Figure 2](#figure_02) below.
<script src="once_per_revolution_force_edt.js" > </script>
<div>
	<div>
		<canvas id="once_per_revolution_force_edt"'></canvas>
	</div>
	<script>
		async function render_chart_once_per_revolution_force_edt() {
			const ctx = document.getElementById('once_per_revolution_force_edt');
			// If this is a mobile device, set the canvas height to 400
			if (window.innerWidth < 500) {
				ctx.height = 400;
			}
			while (typeof Chart == "undefined") {
				await new Promise(r => setTimeout(r, 1000));
			}
			Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
			window.fig_once_per_revolution_force_edt = new Chart(ctx, window.once_per_revolution_force);
			window.fig_blade_once_per_revolution_force_edt_reset = function resetZoomFig1() {
					window.fig_once_per_revolution_force_edt.resetZoom();
				}
			}
		render_chart_once_per_revolution_force_edt();
	</script>
	<a onclick="window.fig_blade_once_per_revolution_force_edt_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_02'>Figure 2</a></strong>: The forcing function caused by an idealized discontinuity in the flow path.</figcaption>
  </figcaption>
</figure>

The forcing function is clearly periodic. A brief force is experienced by each blade once per revolution.

To understand which frequencies are excited by this forcing function, we plot the magnitudes of the first few positive Fourier coefficients in [Figure 3](#figure_03) below:

<script src="fft_once_per_revolution_force_edt.js" > </script>
<div>
    <div>
        <canvas id="fft_once_per_revolution_force_edt"'></canvas>
    </div>
    <script>
        async function render_chart_fft_once_per_revolution_force_edt() {
            const ctx = document.getElementById('fft_once_per_revolution_force_edt');
            // If this is a mobile device, set the canvas height to 400
            if (window.innerWidth < 500) {
                ctx.height = 400;
            }
            while (typeof Chart == "undefined") {
                await new Promise(r => setTimeout(r, 1000));
            }
            Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
            window.fig_fft_once_per_revolution_force_edt = new Chart(ctx, window.fft_once_per_revolution_force);
            window.fig_blade_fft_once_per_revolution_force_edt_reset = function resetZoomFig2() {
                    window.fig_fft_once_per_revolution_force_edt.resetZoom();
                }
            }
        render_chart_fft_once_per_revolution_force_edt();
    </script>
    <a onclick="window.fig_blade_fft_once_per_revolution_force_edt_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_03'>Figure 3</a></strong>:  The magnitudes of the first few positive Fourier coefficients of the forcing function.
  </figcaption>
</figure>

The frequency domain representation of the forcing function 👆 shows that, although the force occurs once per revolution, all EOs are excited by it.

!!! tip "Why does the frequency domain look like this?"

    When I first saw this result, it was completely counter intuitive. Why should a once per revolution disturbance excite all EOs? 
    
    The answer is the forcing function is not a simple sinusoid. The fourier transform stipulates you can represent any time-domain signal as the sum of infinite sinusoidal terms. Our forcing function is therefore made up of an infinite number of sinusoids. It happens to be the case for periodic signals that each sinusoid at an integer multiple of the shaft speed is significant, and the other ones are not.

    I intended to derive this result analytically, but time caught me. If you can derive this result analytically, please get in touch. I would love to include your derivation here.


!!! note "Energy of the excitation and damping ratios"
    The energy of the excitation diminishes as the EO increases. This is one reason why the first few EOs are the most likely to cause damage. Another reason is because higher modes usually have larger damping ratios. This means they are less likely to cause damage than the lower modes.

This explains why a simple discontinuity in the flow path can excite the first few EOs of vibration. Obviously, the forcing function inside a turbomachine is not as simple as the one we've modeled above. But the principle remains the same. A non-sinusoidal periodic forcing function will excite some or all of the low EOs.

!!! note "Discontinuities can be anything"

    We intuitively understand struts or stator vanes count as discontinuities. In fact, there are many possible discontinuities in a turbomachine. A nonconcentric casing might be a discontinuity. A noncircular inlet might introduce a discontinuity.

    I believe, given the breadth of possible obstructions, there will always be at least a single once per revolution excitation source in a turbomachine. 
    
    This is an inkling I have, not a conclusion based on evidence. 😁

## Campbell diagram
Synchronous vibration can only occur when the excitation frequency coincides with a blade natural frequency. It is straightforward to calculate the shaft speed that will cause excitation at a natural frequency. We simply substitute the EOs we expect may occur into [Equation 1](#equation_01) and solve for $\Omega$. 

If, for instance, a blade has a natural frequency of 120 Hz, we can calculate the shaft speeds at which the blade will be excited in [Equation 2](#equation_02) below:

$$
\begin{align}
\Omega_{EO=2} &= \frac{120}{2} = 60 \text{Hz or 3600 RPM }\\
\Omega_{EO=3} &= \frac{120}{3} = 40 \text{Hz or 2400 RPM }\\
\Omega_{EO=4} &= \frac{120}{4} = 30 \text{Hz or 1800 RPM }\\
\dots &
\end{align}
$$
<figure markdown>
  <figcaption><strong><a name='equation_02'>Equation 2</a></strong></figcaption>
</figure>

Is it really so simple? Alas, the physics throws another curveball at us here. 

Rotating objects experience centrifugal force. The centrifugal force experienced by a rotor blade acts radially from the center of the shaft toward the tip of the blade.

This force causes the stiffness of the blade to increase, a phenomenon known as *centrifugal stiffening*. Elevated stiffness leads to increased natural frequencies. The natural frequencies of the blades are therefore not constant, but generally increase as the rotor speeds up.

We need to take this effect into account when we calculate the possible resonance shaft speeds. The Campbell diagram is a handy tool to visually solve the problem. A Campbell diagram contains the natural frequencies of the blades as a function of rotor speed. The excitation frequencies for each EO are also plotted. 

An illustrative Campbell diagram for a rotor blade's first three natural frequencies are displayed in [Figure 4](#figure_04) below.

<script src="campbell_diagram.js" > </script>
<div>
    <div>
        <canvas id="campbell_diagram"'></canvas>
    </div>
    <script>
        async function render_chart_campbell_diagram() {
            const ctx = document.getElementById('campbell_diagram');
            // If this is a mobile device, set the canvas height to 400
            if (window.innerWidth < 500) {
                ctx.height = 400;
            }
            while (typeof Chart == "undefined") {
                await new Promise(r => setTimeout(r, 1000));
            }
            Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
            window.fig_campbell_diagram = new Chart(ctx, window.campbell_diagram);
            window.fig_blade_campbell_diagram_reset = function resetZoomFig3() {
                    window.fig_campbell_diagram.resetZoom();
                }
            }
        render_chart_campbell_diagram();
    </script>
    <a onclick="window.fig_blade_campbell_diagram_reset()" class='md-button'>Reset Zoom</a>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_04'>Figure 4</a></strong>:  An illustrative Campbell diagram for a fictional rotor blade's first three natural frequencies.
  </figcaption>
</figure>

[Figure 4](#figure_04) above illustrates the Campbell diagram's central concepts. The three mode lines indicate three natural frequencies as they change with rotor speed. The dotted lines indicate the excitation frequency associated with each EO of interest. You'll notice the EO lines are perfectly straight. This is because the EO is directly proportional to the rotor speed.

Dark star-shaped markers on [Figure 4](#figure_04) represent shaft speeds where one of the blade's natural frequencies coincide with an EO excitation frequency. Synchronous vibration can only occur at these discrete shaft speeds. We'll call these shaft speeds *resonance speeds*. 

!!! note "FEM and Campbell diagrams"
    The variation of natural frequencies with rotor speed is usually known from Finite Element Analysis (FEA) of the blades. It is almost inconceivable for a commercial rotor blade manufacturer to design a blade without also producing a Campbell diagram. 
    
    In the rare case where you don't have access to one, you'll have to infer the natural frequencies algorithmically. Many methods have been proposed to do this. They are all outside the scope of this tutorial.

## A simple vibration model
The simplest, and often completely sufficient, way of expressing a rotor blade's vibration is to assume the blade is a damped single degree of freedom oscillator under harmonic excitation. The equation of motion for such a system is:

$$
m \ddot{x} + c \dot{x} + k x = F_0 \cdot \cos(\omega t)\\
$$
<figure markdown>
  <figcaption><strong><a name='equation_03'>Equation 3</a></strong></figcaption>
</figure>

Now we divide by $m$. This yields a new equation of motion:

$$
\ddot{x} + 2 \zeta \omega_n \dot{x} + \omega_n^2 x = f_0 \cdot \cos(\omega t) \\
$$
<figure markdown>
  <figcaption><strong><a name='equation_04'>Equation 4</a></strong></figcaption>
</figure>
where

$$
\omega_n = \sqrt{\frac{k}{m}}
$$
<figure markdown>
  <figcaption><strong><a name='equation_05'>Equation 5</a></strong></figcaption>
</figure>
and 

$$
\zeta = \frac{c}{2 \sqrt{k \cdot m}} \\
$$
<figure markdown>
  <figcaption><strong><a name='equation_06'>Equation 6</a></strong></figcaption>
</figure>
and

$$
f_0 = \frac{F_0}{m}
$$

<figure markdown>
  <figcaption><strong><a name='equation_07'>Equation 7</a></strong></figcaption>
</figure>

??? info "Symbols"
    
    | Symbol | Meaning | SI Unit | Domain|
    | :---: | :---: | :---: | :---: |
    | $m$ | Mass of the single degree of freedom system | kg | $m \gt 0$ |
    | $c$ | Damping coefficient | $\frac{Ns}{m}$ | $c \geq 0$ | 
    | $k$ | Stiffness  | $\frac{N}{m}$ | $k \gt 0$ |
    | $F_0$ | Amplitude of the excitation force | $N$ | $F_0 \in \mathbb{R}$ |
    | $\omega$ | Excitation frequency | rad/s | $\omega \gt 0$ |
    | $x$ | Tip Displacement  | $m$ | $x \in \mathbb{R}$ |
    | $\zeta$ | Damping ratio  | $\sqrt{\frac{ N }{ m \cdot kg } } s$ | $\zeta \geq 0$ |
    | $\omega_n$ | Natural frequency | rad/s | $\omega_n \gt 0$ |
    

This equation is a second order ordinary differential equation. A derivation of the solution can be found in Rao's [@rao2003vibrations] excellent text book on mechanical vibrations (Chapter 3). 

The solution is: 

$$
x(t) = X(\omega) \cos (\omega t - \phi(\omega))
$$
<figure markdown>
  <figcaption><strong><a name='equation_08'>Equation 8</a></strong></figcaption>
</figure>

where

$$
\frac{X(\omega)}{\delta_{\text{st}}} = \frac{1}{ \sqrt{ (1 - r^2)^2 + (2 \zeta r)^2 } } 
$$
<figure markdown>
  <figcaption><strong><a name='equation_09'>Equation 9</a></strong></figcaption>
</figure>
and

$$
\phi(\omega) = \arctan \left( \frac{2 \zeta r}{1 - r^2} \right)
$$
<figure markdown>
  <figcaption><strong><a name='equation_10'>Equation 10</a></strong></figcaption>
</figure>

and

$$
r = \frac{\omega}{\omega_n}
$$
<figure markdown>
  <figcaption><strong><a name='equation_11'>Equation 11</a></strong></figcaption>
</figure>

??? info "Symbols"
        
    | Symbol | Meaning | SI Unit | Domain|
    | :---: | :---: | :---: | :---: |
    | $\delta_{\text{st}}$ | Deflection under the static force $F_0$ | $m$ | $\delta_{\text{st}} \in \mathbb{R}$ |
    | $r$ | Excitation frequency ratio | - | $r \gt 0$ |

Each blade will have different values for $\omega_n$, $\delta_{\text{st}}$, and $\zeta$. These values determine the vibration response of the blade. Intuition about the solution can be gained by fixing $\omega_n=125$ Hz and $\delta_{\text{st}} = 1$. We can plot the solution for different values of $\zeta$ and $\omega$.

!!! note "Natural frequency unit"
    Normally, the unit you use for natural frequency (Hz or rad/s) depends on where you want to use it. Here, however, the natural frequency gets absorbed into the excitation frequency ratio, $r$. It therefore does not matter which unit you use here. We'll use Hz for convenience.

The *slider* below 👇 allows you to change the value of $\zeta$. The resulting vibration amplitude and phase as a function of excitation frequency are plotted in [Figure 4](#figure_04) below.

<div>
	<div>
        <div>
            <div class="slidecontainer" >
                <input 
                    type="range" 
                    width="100%" 
                    min="0.01" 
                    max="0.3"
                    step="0.002" 
                    value="0.1"
                    style="width:100%" 
                    id="zeta_slider"
                >
                <p>Current ζ: <strong><span id="current_zeta"></span> </strong></p>
            </div>
            <div>
                <span><strong>A)</strong></span>
                <canvas height=100 id="vibration_amplitude"></canvas>
                <span><strong>B)</strong></span>
                <canvas height=100 id="vibration_phase"></canvas>
            </div>
        </div>
	</div>
	<script>
        function makeArr(startValue, stopValue, cardinality) {
            var arr = [];
            var step = (stopValue - startValue) / (cardinality - 1);
            for (var i = 0; i < cardinality; i++) {
                arr.push(startValue + (step * i));
            }
            return arr;
        }
        function calculate_sdof_oscillator_params_from_omegas(zeta, omegas) {
            // This function calculates the following two parameters
            // of an SDoF oscillator:
            // X(ω) and φ(ω)
            // where the oscillator is given by x(t) = X(ω) * sin(ω * t + φ(ω)
            // and the natural frequency of the system us 125 Hz
            var Xs = [];
            var phis = []
            for (omega of omegas) {
                var r = omega / 125;
                var X = (
                    1 
                    / Math.sqrt(
                        (1 - r**2)**2
                        + (2*zeta*r)**2
                    )
                )
                var phi = Math.atan2(
                    (2*zeta*r) 
                    ,(1 - r**2)
                )
                Xs.push(X);
                phis.push(phi);
            }
            return [Xs, phis];
        }
        
		async function render_vibration_model() {
            var ctx_vibration_amplitude = document.getElementById('vibration_amplitude').getContext('2d');
            var ctx_vibration_phase = document.getElementById('vibration_phase').getContext('2d');
			// If this is a mobile device, set the canvas height to 400
			if (window.innerWidth < 500) {
				ctx_vibration_amplitude.height = 400;
                ctx_vibration_phase.height = 400;
			}
			while (typeof Chart == "undefined") {
				await new Promise(r => setTimeout(r, 1000));
			}
			Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
        const ZETAS = makeArr(0.01, 0.5);
        const OMEGAS = makeArr(100, 150, 3*50 + 1);
        
        var vibration_amplitude = new Chart(ctx_vibration_amplitude, {
            type: 'line',
            data : {
                labels: OMEGAS,
                datasets: [{
                    pointStyle: false,
                    type: "line",
                    label: 'X(ω)',
                    data: [{}],
                    borderColor: 'rgba(255, 99, 132)',
                    borderWidth: 3,
                    fill: false,
                }]
            },
            options:{
                plugins :{

                },
                scales: {
                    y: {// Label this axis as "Real Tip Vibration"
                        title: {
                            display: true,
                            text: ['X(ω)'],
                            font: {
                                size: 20,
                            }
                        },
                        min:0,
                        max : 50
                    },
                    x : {
                        title : {
                            display : true,
                            text : "ω [Hz]",
                            font : {
                                size : 20
                            }
                        },
                        "ticks": {
                            callback : (value, index, values) => {
                                var current_value = OMEGAS[value]; 
                                // If current value is within 0.00001 of an integer, return it
                                if (Math.abs(current_value - Math.round(current_value)) < 0.0001) {
                                    return current_value;
                                }
                            }
                        }
                    }
                }
            }
        });
        
        var vibration_phase = new Chart(ctx_vibration_phase, {
            type: 'line',
            data : {
                labels: OMEGAS,
                datasets: [{
                    pointStyle: false,
                    type: "line",
                    label: 'φ(ω)',
                    data: [{}],
                    borderColor: 'rgba(255, 99, 132)',
                    borderWidth: 3,
                    fill: false,
                }]
            },
            options:{
                plugins :{

                },
                scales: {
                    y: {// Label this axis as "Real Tip Vibration"
                        title: {
                            display: true,
                            text: ['φ(ω) [rad]'],
                            font: {
                                size: 20,
                            }
                        },
                        min:0,
                        max : Math.PI*1.1
                    },
                    x : {
                        title : {
                            display : true,
                            text : "ω [Hz]",
                            font : {
                                size : 20
                            }
                        },
                        "ticks": {
                            callback : (value, index, values) => {
                                var current_value = OMEGAS[value]; 
                                // If current value is within 0.00001 of an integer, return it
                                if (Math.abs(current_value - Math.round(current_value)) < 0.0001) {
                                    return current_value;
                                }
                            }
                        }
                    }
                }
            }
        });


        var zeta_slider = document.getElementById("zeta_slider");
        zeta_slider.oninput = function() {
            // Calculate X and phi
            var [Xs, phis] = calculate_sdof_oscillator_params_from_omegas(this.value, OMEGAS);
            // Update the chart
            vibration_amplitude.data.datasets[0].data = Xs;
            vibration_phase.data.datasets[0].data = phis;
            vibration_amplitude.update();
            vibration_phase.update();
            document.getElementById("current_zeta").innerHTML = this.value;
        }
        // Trigger a change event to initialize the chart
        zeta_slider.value = 0.1;
        zeta_slider.oninput();
		}
    render_vibration_model()
	</script>
</div>
<figure markdown>
  <figcaption><strong><a name='figure_05'>Figure 5</a></strong>: The amplitude and phase of a single degree of freedom oscillator as a function of excitation frequency. We've fixed the natural frequency to 125 Hz and the static deflection to 1.
  </figcaption>
</figure>

Two observations from [Figure 5](#figure_05) are highlighted below:

* Larger damping ratios lead to smaller amplitudes.
* The phase of the vibration *always shifts* by $\pi$ radians as the resonance is traversed. This is a fundamental law of vibration. The rate at which this shift occurs  is controlled by the damping ratio. The larger the damping ratio, the slower the phase shift.

## Sampling
We now have a mathematical expression that describes the shape of a blade tip's vibration response. Theoretically, we can use the expression to calculate the tip deflection at *any point in time*. However, *we cannot measure* the tip deflection at any point in time. We can only measure the tip deflection each time a blade passes a probe. 

In other words, despite the fact that the blade's vibration response is continuous, we only get *one sample of the continuous waveform* each time a blade passes a probe.

To illustrate this concept, we've simulated the vibration response of a blade and *artificially placed three proximity probes* into a "casing". The *slider* below 👇 allows you to change the shaft speed, and observe *both* the continuous vibration response in [Figure 6](#figure_06) __A)__ *and* the samples taken by our BTT system in [Figure 6](#figure_06) __B)__.
<div>
	<div>
        <div>
            <div class="slidecontainer" >
                <input 
                    type="range" 
                    width="100%" 
                    min="1150" 
                    max="1350"
                    step="1" 
                    value="1150" 
                    style="width:100%" 
                    id="speed_slider"
                >
                <p>Current shaft speed: <strong><span id="shaft_speed"></span> RPM</strong></p>
            </div>
            <div>
                <span><strong>A)</strong></span>
                <canvas height=100 id="sinusoid_chart"></canvas>
                <span><strong>B)</strong></span>
                <canvas height=100 id="sampled_chart"></canvas>
            </div>
        </div>
	</div>
	<script>

        function makeArr(startValue, stopValue, cardinality) {
            var arr = [];
            var step = (stopValue - startValue) / (cardinality - 1);
            for (var i = 0; i < cardinality; i++) {
                arr.push(startValue + (step * i));
            }
            return arr;
        }
        function calculate_sdof_oscillator_params(zeta, EO, Omega) {
            // This function calculates the following two parameters
            // of an SDoF oscillator:
            // X(ω) and φ(ω)
            // where the oscillator is given by x(t) = X(ω) * sin(ω * t + φ(ω))
            // and the natural frequency of the system us 125 Hz
            var r = (EO * Omega) / (125 * 2 * Math.PI);
            var X = (
                1 
                / Math.sqrt(
                    (1 - r**2)**2
                    + (2*zeta*r)**2
                )
            )
            var phi = Math.atan2(
                (2*zeta*r) 
                ,(1 - r**2)
            )
            return {"X":X, "phi":phi};
        }
        function rpm_to_rad_p_sec(rpm){
            return rpm / 60 * Math.PI * 2;
        }
        function make_sinusoid_vs_angle(EO, Omega, zeta, angles) {
            /* This function creates a sinusoid with frequency
                f = Omega * EO and phase Phi. 

            Args:
                EO (int): The Engine Order of vibration.
                Omega (float): The shaft speed in rad/s.
                zeta (float): The damping ratio of the system.
                angles (array): The angles at which the sinusoid
                    should be evaluated

            Returns:
                array: An array of objects with the following
                    structure:
                    {
                        x: angle,
                        y: sin(EO * Omega * angle + phi)
                    }
            */
            var data = []
            var sdof_params = calculate_sdof_oscillator_params(zeta, EO, Omega);
            for (var i = 0; i < angles.length; i++) {
                data.push({
                    x : angles[i],
                    y : sdof_params["X"] * Math.sin(EO * 2 * Math.PI * angles[i]/360 - sdof_params["phi"])
                });
            }
            return data;
        }
        function make_sinusoid_vs_omega(EO, omegas, zeta, angle) {
            /* This function creates a sinusoid with frequency
                f = Omega * EO and phase Phi. 

            Args:
                EO (int): The Engine Order of vibration.
                omegas (array[float]): The shaft speeds in RPM.
                zeta (float): The damping ratio of the system.
                angle (float): The angle at which the sinusoid
                    should be evaluated

            Returns:
                array: An array of objects with the following
                    structure:
                    {
                        x: shaft speed,
                        y: sin(EO * Omega * angle + phi)
                    }
            */
            var data = []
            for (var i = 0; i < omegas.length; i++) {
                var sdof_params = calculate_sdof_oscillator_params(
                    zeta, 
                    EO, 
                    rpm_to_rad_p_sec(omegas[i])
                );
                data.push({
                    x : omegas[i],
                    y : sdof_params["X"] * Math.sin(EO * 2 * Math.PI * angle/360 - sdof_params["phi"])
                });
            }
            return data;
        }
        

		async function render_sampling_illustration() {
			var ctx_sinusoid = document.getElementById('sinusoid_chart').getContext('2d');
            var ctx_sampled = document.getElementById('sampled_chart').getContext('2d');
			// If this is a mobile device, set the canvas height to 400
			if (window.innerWidth < 500) {
				ctx_sinusoid.height = 400;
                ctx_sampled.height = 400;
			}
			while (typeof Chart == "undefined") {
				await new Promise(r => setTimeout(r, 1000));
			}
			Chart.defaults.font.family = "Literata, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif";
            const SINUSOID_X = makeArr(0, 360, 361);
            const PROBE_ANGLES = [45, 145, 275];
            const SENSOR_WIDTH = 15;
            var SINUSOIDAL_CHART_ANNOTATIONS = {};
            const PROBE_COLORS = [
                "rgba(255, 99, 132)",
                "rgba(54, 162, 235)",
                "rgba(255, 206, 86)",
            ]
            for (var i = 0; i < PROBE_ANGLES.length; i++) {
                SINUSOIDAL_CHART_ANNOTATIONS["probe_" + i] = {
                    type: 'box',
                    xMin: PROBE_ANGLES[i]-SENSOR_WIDTH,
                    xMax: PROBE_ANGLES[i]+SENSOR_WIDTH,
                    yMin: 55,
                    yMax: 73,
                    backgroundColor: 'yellow',
                    borderWidth: 2,
                    borderColor: PROBE_COLORS[i],
                    label: {
                        display: true,
                        content: "P" + (i + 1)
                    },
                    borderRadius : {
                        bottomLeft : 14,
                        bottomRight : 14,
                    }
                }
                SINUSOIDAL_CHART_ANNOTATIONS["probe_" + i + "line"] = {
                    type: 'line',
                    yMin: -50,
                    yMax: 55,
                    xMin: PROBE_ANGLES[i],
                    xMax: PROBE_ANGLES[i],
                    borderColor: 'black',
                    borderWidth: 5,
                    borderDash: [5, 5],
                    borderColor: PROBE_COLORS[i],
                    borderRadius : 5,
                }
            }
            var sinusoidal_chart = new Chart(ctx_sinusoid, {
                type: 'line',
                data : {
                    labels: SINUSOID_X,
                    datasets: [{
                        pointStyle: false,
                        type: "line",
                        label: 'Tip vibration',
                        data: [{}],
                        borderColor: 'rgba(255, 99, 132)',
                        borderWidth: 1,
                        fill: false,
                    }]
                },
                options:{
                    plugins :{
                        annotation : {
                            annotations : SINUSOIDAL_CHART_ANNOTATIONS
                        }
                    },
                    scales: {
                        y: {// Label this axis as "Real Tip Vibration"
                            title: {
                                display: true,
                                text: ["True tip"," vibration"],
                                font: {
                                    size: 20,
                                }
                            },
                            min:-50,
                            max : 75
                        },
                        x : {
                            title : {
                                display : true,
                                text : "Angle [deg]",
                                font : {
                                    size : 20
                                }
                            }
                        }
                    }
                }
            });
            
            var PROBE_FRF_DATASETS = [];
            const ZETA = 0.01;
            const EO = 6;
            const SHAFT_SPEEDS = makeArr(1150, 1350, 201);
            for (var i = 0; i < PROBE_ANGLES.length; i++) {
                PROBE_FRF_DATASETS.push({
                    pointStyle: false,
                    type: "line",
                    label: 'P' + (i + 1),
                    data: make_sinusoid_vs_omega(EO, SHAFT_SPEEDS, ZETA, PROBE_ANGLES[i]),
                    borderWidth: 1,
                    fill: false,
                    borderColor: PROBE_COLORS[i],
                })
            }
            var PROBE_FRF_CURRENT_SAMPLE = {}
            for (var i = 0; i < PROBE_ANGLES.length; i++) {
                PROBE_FRF_CURRENT_SAMPLE['probe_' + i + "_current_sample" ] = {
                    type: 'point',
                    xValue: 0,
                    yValue: 0,
                    backgroundColor: PROBE_COLORS[i],
                    radius : 7
                }
            }
            var sampled_chart = new Chart(ctx_sampled, {
                type: 'line',
                data : {
                    labels: SHAFT_SPEEDS,
                    datasets: PROBE_FRF_DATASETS,
                },
                options : {
                        plugins : {
                            annotation : {
                                annotations : PROBE_FRF_CURRENT_SAMPLE
                        }
                    },
                    scales : {
                        y : {
                            title : {
                                display : true,
                                text : ["Sampled tip"," vibration"],
                                font : {
                                    size : 20
                                }
                            }
                        },
                        x : {
                            title : {
                                display : true,
                                text : "Shaft speed [RPM]",
                                font : {
                                    size : 20
                                }
                            }
                        }
                    }
                }
            });
            
            var speed_slider = document.getElementById("speed_slider");
            speed_slider.oninput = function() {
                // Calculate main sinusoid
                sinusoid = make_sinusoid_vs_angle(EO, rpm_to_rad_p_sec(this.value), ZETA, SINUSOID_X);
                sinusoidal_chart.data.datasets[0].data = sinusoid;
                // Calculate probe sampling
                probe_samples = make_sinusoid_vs_angle(EO, rpm_to_rad_p_sec(this.value), ZETA, PROBE_ANGLES);
                //console.log(sinusoidal_chart)
                let index_no = this.value - 1150;
                for (var i = 0; i < probe_samples.length; i++) {
                    sinusoidal_chart.options.plugins.annotation.annotations["probe_" + i + "line"]["yMin"] = probe_samples[i]["y"];
                    sampled_chart.options.plugins.annotation.annotations["probe_" + i + "_current_sample"]["yValue"] = probe_samples[i]["y"];
                    sampled_chart.options.plugins.annotation.annotations["probe_" + i + "_current_sample"]["xValue"] = index_no;
                }
                sampled_chart.update();
                sinusoidal_chart.update();
                document.getElementById("shaft_speed").innerHTML = this.value;
            }
            // Trigger a change event to initialize the chart
            speed_slider.value = 1150;
            speed_slider.oninput();
		}
    render_sampling_illustration()
	</script>

</div>
<figure markdown>
  <figcaption><strong><a name='figure_06'>Figure 6</a></strong>: A) A blade's continuous vibration response as a function of the blade's angular position. B) The samples taken by our BTT system as a function of shaft speed. The shaft speed can be controlled by the slider. If you're uncomfortable because we've expressed the continuous vibration vs angle instead of time, just keep reading.
  </figcaption>
</figure>

??? info "Simulation parameters"
    * $\omega_n = 125$
    * $\delta_{\text{st}}$ = 1
    * $\Omega \in [1150, 1350]$ RPM
    * $EO = 6$ 
    * $\zeta = 0.01$
    * $\text{Sensor locations} = [45, 145, 275]$ deg

In [Figure 6](#figure_06) above, the continuous tip deflection is presented in __A)__. We've also *placed* three proximity probes, prefixed by P, in the *casing* above the signal. Each proximity probe will *sample* the continuous waveform at the value corresponding to the vertical dotted line that stretches from the probe to the waveform. As you move the slider, you'll notice the continuous waveform changes in both amplitude and phase. You'll also notice the values sampled by each probe change.

In [Figure 6](#figure_06) __B)__, the sampled values of the BTT system as a function of the entire shaft speed range are plotted. The instantaneous samples for the shaft speed as it is currently set is indicated by large markers on __B__. The corresponding samples are indicated on __A)__ at the ends of the vertical dotted lines.

We cannot stress the implication of this figure enough, we're therefore going to use a fancy box 👇 to highlight it.

!!! info "Continuous vs Sample signals"
    We __are not__ measuring the continuous signal shown in [Figure 6](#figure_06) __A)__ above. We __only measure__ the sampled values for each probe as indicated in __B__).

    The task of BTT frequency inference is to *__infer the continuous waveform from the samples__*.

## Substitute angle for time
Back in [Equation 4](#equation_04), we expressed the tip deflection as a function of time: $x(t)$. In [Figure 6](#figure_06), however, we plotted the tip deflection as a function of angle. Why the apparent conflict?

This is because we can substitute time for angle in our equations.

In reality the tip deflection *does* vary with time, but it is a faux dependance. To show why, the definition of synchronous vibration is repeated below:

$$
\omega = \Omega \cdot EO
$$
<figure markdown>
  <figcaption><strong><a name='equation_12'>Equation 12</a></strong></figcaption>
</figure>

recall the shaft speed, $\Omega$, can be expressed as the distance traveled by the rotor from the start of a revolution until it reaches a sensor's position, $\theta_s$:

$$
\Omega = \frac{\theta_s}{t}
$$
<figure markdown>
  <figcaption><strong><a name='equation_13'>Equation 13</a></strong></figcaption>
</figure>

We can substitute this into  [Equation 12](equation_12) above to get:

$$
\omega = \frac{\theta_s}{t} \cdot EO
$$
<figure markdown>
  <figcaption><strong><a name='equation_14'>Equation 14</a></strong></figcaption>
</figure>

Finally, we substitute the above equation into [Equation 8](equation_08) to get:

<span id='eq_final_sdof_eq'></span>

\begin{align}
x(t) &= X(\omega) \cos \left( \omega \cdot t - \phi(\omega) \right) \\
x(t) &= X(\omega) \cos \left( \frac{\theta_s}{t} \cdot EO \cdot t - \phi(\omega) \right) \\
&= X(\omega) \cos \left( \theta_s \cdot EO - \phi(\omega) \right)
\end{align}
<figure markdown>
  <figcaption><strong><a name='equation_15'>Equation 15</a></strong></figcaption>
</figure>

The tip deflection is only dependant on the EO, the location of the sensor, and the shaft speed (since $\omega=\Omega \cdot EO$). The tip deflection is therefore *not* dependant on time.

The implications of this are profound. Normally in vibration measurement, the longer you measure something, the more information you get. Our equations reveal, however, if you keep the shaft speed constant and you measure the tip deflections for all eternity, you will measure the exact same deflections over and over again. You effectively only have as many unique samples as there are sensors. 


!!! note "This is why synchronous vibration is more difficult to measure than asynchronous vibration."

    Recall we said at the beginning of the chapter that synchronous vibration is more difficult to measure than asynchronous vibration? This is the reason why. 
    
    The continuous waveform will be sampled at approximately the same points over and over again. This makes it difficult to infer the continuous waveform from the samples. Asynchronous vibration, on the other hand, is not dependant on the shaft speed. The continuous waveform will be sampled at different points each revolution. This makes it easier to infer the continuous waveform from the samples.


## Aliasing
It is often pointed out that BTT signals are aliased. This means BTT systems sample at a rate below the Nyquist frequency of the blade response. 

The Nyquist frequency is double the natural frequency we want to measure:

$$
\begin{align}
f_{s,N} &= \omega_n \cdot 2\\
&= 125 \cdot 2\\
&= 250 Hz
\end{align}
$$
<figure markdown>
  <figcaption><strong><a name='equation_16'>Equation 16</a></strong></figcaption>
</figure>

We can calculate our BTT system's sampling rate at the EO 6 resonance speed of 1250 RPM:

$$
\begin{align}
f_s &= \Omega \cdot S\\
&= \frac{\omega}{EO} \cdot S\\
&= \frac{1250}{60} \cdot 3\\
&= 62.5\\
\end{align}
$$

??? info "Symbols"
    | Symbol | Meaning | SI Unit | Domain|
    | :---: | :---: | :---: | :---: |
    | $f_{s,N}$ | Nyquist frequency | $Hz$ | $f_{sN} \gt 0$ |
    | $f_s$ | Sampling frequency | $Hz$ | $f_s \gt 0$ |
    | $\Omega$ | Shaft speed | $Hz$ | $\Omega \gt 0$ |

We only measure 62.5 samples per second, whereas the required rate is 250 samples per second. This is why BTT signals are said to be aliased.

!!! note
    Although the above method provides intuition, I do not believe it is a mathematically sound deduction. We normally associate aliasing and the Nyquist frequency with signals that can be transformed using the Discrete Fourier Transform (DFT) . One requirement of the DFT is the samples must be equidistant along the discretization axis, like time or angle. BTT sensors are generally not equally far apart from one another. Even if you attempted to install them equidistantly, manufacturing errors would render the samples non-equidistant.

    You can read about this in more detail in [@vanderplas2018understanding].

## Conclusion

In this chapter, we've spent some time to understand the fundamentals behind synchronous vibration. We've shown that BTT systems sample a continuous vibration waveform, and we need to infer the true vibration behavior from these samples.   

The final two chapter describe two ways of completing our promised journey. The two remaining chapters describe different ways to infer vibration frequency, amplitude, and phase from the samples:

* Ch8: The Single Degree of Freedom (SDoF) fit method, and
* Ch9: The Circumferential Fourier Fit (CFF) method;

!!! question "Outcomes"

	:material-checkbox-marked:{ .checkbox-success .heart } Understand how obstructions in the flow path causes synchronous vibration.

	:material-checkbox-marked:{ .checkbox-success .heart } Understand that synchronous vibration occurs at an integer multiple of the shaft speed. This integer multiple is called the Engine Order (EO). 

    :material-checkbox-marked:{ .checkbox-success .heart } Understand what a Campbell diagram is and how we can use it to guess the mode associated with a resonance event we measured.

	:material-checkbox-marked:{ .checkbox-success .heart } Understand that we can use a Single Degree of Freedom (SDoF) oscillator to model a blade's vibration.
	
	:material-checkbox-marked:{ .checkbox-success .heart } Understand how BTT systems sample the blade's vibration waveform.

    :material-checkbox-marked:{ .checkbox-success .heart } Understand why synchronous vibration data is more difficult to process than asynchronous vibration data.

	:material-checkbox-marked:{ .checkbox-success .heart } Understand that BTT signals are generally aliased.


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
            2024-03-12
        </p>
    </div>
</div>