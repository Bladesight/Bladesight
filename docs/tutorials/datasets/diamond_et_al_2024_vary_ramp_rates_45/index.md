---
date: 2024-10-28
tags:
  - Blade Tip Timing
  - BTT
  - Non Intrusive Stress Measurement
  - NSMS
  - Turbine blade
  - Mechanical Vibration
hide:
  - tags
description: This chapter explains the setup and method used to collect the Diamond et. al. 2024 dataset 45 degree dataset.
robots: index, follow, Blade Tip Timing, BTT, Non Intrusive Stress Measurement, NSMS, Turbine blade,Mechanical Vibration
template: main_datasets.html
card_title: Dataset - Diamond et al. 2024 Varying Ramp Rates
card_url: "datasets_diamond_et_al_2024_varying_ramp_rates/"
---
# Varying Ramp Rates with a 45 degree angle pipe as a forcing mechanism

I will now beat resistance.

## TL;DR

How to use this dataset:

```python

from bladesight import Datasets

ds = Datasets["Diamond_et_al_2024/ramp_rates_45_degrees"]

ds.query(
  """
  SELECT *
  """
)

```

## Why did we do conduct these tests?

I've had a stone-in-my foot feeling about transient resonances in BTT for a very long time. I've always felt there was something off about how transient resonances were not being analyzed. I've felt this way primarily about the way I analyzed them.

What has bothered me most, however, is the nonchalant attitude that pervades published literature. The attitude is one of: "this has been sorted already, let's move on to something more interesting." 

Why, then, is BTT not yet a standard measurement technique for condition monitoring in the power generation industry? I believe it is because it has not pitched up as a reliable method to find cracks. Cracks are what we are concerned about. Cracks are what we are trying to find. What does it take to find cracks?

...

## Experimental setup and method

The main investigation is to investigate how the relationship $f = ΩEO$, between blade vibration frequency and
shaft speed, changes through different varying speed conditions.

The BTT test bench at the University of Pretoria’s Sasol laboratory was used for this investigation.

### Rotor, motor, and blade excitation

The test bench consists of a five blade blisk. The blades of the blisk have an outer radius of 164 mm. These blades
are rectangular, allowing us to study the effects of pure bending modes. The motor is driven by a Variable Speed Motor
which can be supplied with an analogue signal that varies between 0 - 10 Volts. The maximum speed of the rotor is
approximately 1450 RPM. Compressed air was used to excite the blades. The pipe, captured in Figure 1, was placed
at approximately a 45 degree angle with respect to the blades’ chord. A 45 degree angle is selected as to observe all
resonances most prominently. The rotor blades’ first three natural frequencies at rest are 125 Hz ,5xx Hz, 7xx Hz.


### Sensors

Three MEGGIT TQ401 eddy current sensors are used for the experiment. Two are used as proximity probes and
one is used as an OPR sensors. An Optel Thevon 152G8 optical fibre tachometer is also installed in conjunction with
the zebra-tape shaft speed encoder. Strain gauge attached using a slip ring.

### Data acquisition and control

This experiment was part of a test to investigate a variety of sensors and Data Acquisition System (DAQ)-related
questions, as such three different electronic DAQ systems were used.
An under-development BTT DAQ, the Bladesight Kraken X4 was used to acquire the Time-of-Arrival (ToA)s
from the TQ401 eddy current sensors. This system digitizes the incoming samples at 125 MHz. This is a DAQ of
the corresponding author’s own make during its final commissioning run before being installed for the continuous
monitoring of a 600 MW steam turbine. This box was used to condition the TQ401 sensors from the -24 V - 0V range
to -1.5 to 1.5 V range and to extract the ToA.

A HBK Quantum (MX410) was used to measure the strain gauge and output the signal. A <Confirm model number
and manufacturer, ill do this when I am next at the lab> slip ring is used to transmit the strain gauge signal to the
Quantum DAQ.
An OROS OR35 Data acquisition system was used to measure the Multiple Per Revolution (MPR), strain gauge,
and three other inductive probes (TURCK BI10U-M18-AP6X) which did not make part of this study.
CoCo-80x analyzer
Laptops were used to communicate to and from the DAQs. A CoCo 80x Analyser and PCB Piezotronics 086C03
modal impulse hammer is used for the tap tests
ADD PHOTOS OF THE 4 SYSTEMS: BLADESIGHT KRAKEN X4, OROS ANALYER, HBK QUANTUM,
and OROS OR35 ANALYZER.

## Experimental method

The CoCo-80X Analyser was used to generate triangular waveforms, for each speed profile, that vary from 0 - 10
V.
The ramp rates, defined as the time taken to move from 0 RPM to 1450 RPM, for different test cases were as
follows: 30s, 40s, 50s, 60s, 70s, 80s, 90s, 100s, 110s, 120s
For each test, acquisition was performed for at least 4 cycles. Giving 4 run-ups and run-downs per test.
The below chart shows the ramp rates of every test case. - [ ] Just a question, not sure if its the axes (and that
they not "square") but the 30s RR (and maybe some others but I’m not sure) seems to take less time on the ramp down
compared to the ramp up? I don’t know if this is negligible or not - i would think so. Good point. I’ll explicitly calculate
the ramp rate when doing the in-depth analysis. The ramp rates indicated in Figure ?? above are calculated from the
OPR pulses. Only the first cycle is shown. All BTT data analyses were conducted in Python using the open source
Bladesight package. step-by-step guide to reproduce the results presented in this paper is presented at

## Multiple Triggering levels

I've always thought it nonsensical to favour a single threshold level for BTT....

## Strain gauge data

The strain gauge attached to the blades are beautiful. We endeavored to calibrate the strain gauge after the tests. As things go, we put this off, and at some stage Murphy intervened and we tore the strain gauge off the blade.

There is no calibration data that we can use to convert the strain gauge data to tip displacement. This is a pity. We will have to make do with the raw strain gauge data.

Fortunately, it is still an extremely rich dataset.

## Getting started 👇

### Using the OPR as a reference

### Using the MPR as a reference

### Use multiple triggering levels to uncover something interesting?

## Conclusion

## Acknowledgements

A big thanks to the University of Pretoria for providing the test bench and the laboratory space to conduct these tests.

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
            2024-11-XX
        </p>
    </div>
</div>
