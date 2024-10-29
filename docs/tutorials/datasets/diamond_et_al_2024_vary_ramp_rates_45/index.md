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
# Diamond et al 2024 ramp rates at 45 degrees

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

I've always felt like there's a stone in my shoe while fitting curves to transient synchronous resonances. The resonant frequencies recovered by curve fitting algorithms have always struck me as imprecise. A prime example of this is when you measure the resonant frequency during a run-up vs the same frequency during a run-down. Logic dictates that such frequencies should be the same. 

I've found they are never the same.

We then set about to perform tests to investigate the effect of transient ramp rate and direction on the *apparent* resonant frequency.

Unfortunately, about a fortnight prior to completing our first draft, I found a paper by [@zhi2022error] that investigated this phenomenon. It could be that most BTT practitioners know about this. But I doubt it. 

I believe most people subscribe to the classical view that synchronous resonances are integrally proportional to the shaft speed. This is not the case.

I believe the tests we conducted were proper and also provides a good dataset to investigate the effect of ramp rate on the apparent resonant frequency. It may not be the groundbreaking study I thought it would be, but it is still a good dataset.

I've uploaded the dataset to Bladesight and released it under the Creative Commons license. 

I hope it will be of use to someone.

## Experimental setup and method

The main investigation is to investigate how the relationship $f = ΩEO$, between blade vibration frequency and
shaft speed, changes through different varying speed conditions.

The BTT test bench at the University of Pretoria's Sasol Laboratory was used for this investigation.

### Rotor, motor, and blade excitation

The test bench consists of a five blade blisk. The blades of the blisk have an outer radius of 164 mm. These blades
are rectangular, thereby allowing us to study the effects of pure bending modes. The motor is controlled with an analogue signal that varies between 0 - 10 Volts, corresponding to the minimum and maximum speed, respectively. 

The maximum speed of the rotor is approximately 1450 RPM. Compressed air was used to excite the blades. The pipe, captured in [Figure 1](#figure_01), was placed at approximately a 45 &deg; angle with respect to the blades' chord. 

!!! note

    The choice of 45 degrees was actually accidental. Prior to positioning the pipe at 45 &deg; angle, we placed it close to a 90 &deg; angle. I hypothesized that, because the air was directed perpendicular to the blade's chord, the largest tip deflections would be observed. 

    We then conducted a single test with the pipe at a 45 &deg; angle. The tip deflections were significantly larger than those observed at 90 &deg;. It was like finding out there's another 5 km left after running a marathon. 
    
    We repeated all the tests at 45 &deg;, yielding the dataset were presenting here.


A 45 degree angle is selected as to observe all resonances most prominently. The rotor blades’ first three natural frequencies at rest are 125 Hz ,5xx Hz, 7xx Hz.

<figure markdown>
  ![Experimental setup](fig_1_experimental_setup.png){ width="500" }
    <figcaption><strong><a name='figure_01'>Figure 1</a></strong>: The experimental setup at the University of Pretoria's Sasol Laboratory. The image has been taken from the front.  
    </figcaption>
</figure>


!!! note

  We had every intent to return to the test bench and take better photos. We never did. The photos we have are not the best. We apologize for this.

### Sensors

Three <a href='https://catalogue.meggittsensing.com/shop/sensors-and-signal-conditioners/proximity-probes/tq401-proximity-sensor/' target='_blank'> MEGGIT TQ401 eddy current proximity probes</a> were used. Two were used as proximity probes and one as an OPR sensors. 

An Optel Thevon 152G8 optical fibre sensor was also installed in conjunction with the zebra-tape to serve as a Multiple Pulses per Encoder (MPR) speed encoder. 

We instrumented the blade with a strain gauge at the root of the blade. The strain gauge was installed to be sensitive to bending.

### Data acquisition and control

In addition to understanding the effects of ramp rates and directions on blade vibration, we also included new/unproven sensors and Data Acquisition Systems (DAQs). Ultimately, four different DAQs and control systems were used.

The four DAQs and control systems are shown in [Figure 2](#figure_02) below.

<figure markdown>
  ![Experimental setup](4_daqs.png){ width="800" }
    <figcaption><strong><a name='figure_02'>Figure 2</a></strong>: The four DAQs and control systems used in the tests. 
    </figcaption>
</figure>

#### Kraken X4
My under-development BTT DAQ, the Bladesight Kraken X4, was used to acquire the Time-of-Arrivals (ToAs)
from the TQ401 eddy current sensors. The Kraken digitizes the incoming samples at 125 MHz. The Kraken was programmed to trigger the ToAs at multiple thresholds, not simply one. We'll discuss this in more detail later.

#### HBK Quantum
An HBK Quantum (MX410) was used to excite and measure the strain gauge. Excitation and communication to and from the strain gauge ocurred a slip ring.

#### OROS OR35
An OROS OR35 Data acquisition system was used to measure the Multiple Per Revolution (MPR), strain gauge, and three other inductive probes (TURCK BI10U-M18-AP6X). 

The OROS' sampling rate was set to 25.6 kHz.

#### CoCo-80x analyzer
CoCo-80x analyzer Laptops were used to communicate to and from the DAQs. A CoCo 80x Analyser and PCB Piezotronics 086C03

## Experimental method

The CoCo-80X Analyser was used to generate triangular waveforms, for each speed profile, that vary from 0 - 10
V.

The ramp rates, defined as the time taken to move from 0 RPM to 1450 RPM, for different test cases were as
follows: 30s, 40s, 50s, 60s, 70s, 80s, 90s, 100s, 110s, 120s. 

For each test, acquisition was performed for at least 4 cycles. Giving 4 run-ups and run-downs per test.

The below chart shows the ramp rates of every test case. - [ ] Just a question, not sure if its the axes (and that
they not "square") but the 30s RR (and maybe some others but I’m not sure) seems to take less time on the ramp down
compared to the ramp up? I don’t know if this is negligible or not - i would think so. Good point. I’ll explicitly calculate
the ramp rate when doing the in-depth analysis. The ramp rates indicated in [[Figure 2](#figure_02)] above are calculated from the OPR pulses. 

Only the first cycle is shown. All BTT data analyses were conducted in Python using the open source Bladesight package.

<figure markdown>
  ![Experimental setup](fig_2_ramp_rates.jpeg){ width="800" }
    <figcaption><strong><a name='figure_02'>Figure 2</a></strong>: The ramp rates of each test case.
    </figcaption>
</figure>

## Multiple Triggering levels

I've always thought it nonsensical to favour a single threshold level for BTT. Some blades are longer than others. It makes sense that you should use different triggering level for different blades.

The Kraken X4 was programmed to trigger at 8 levels on the upwards slope and downwards slope.

The levels were ...

## Strain gauge data

The strain gauge attached to the blades are beautiful. We endeavored to calibrate the strain gauge after the tests. As things go, we put this off, and at some stage Murphy intervened and we tore the strain gauge off the blade.

There is no calibration data that we can use to convert the strain gauge data to tip displacement. This is a pity. We will have to make do with the raw strain gauge data.

Fortunately, it is still an extremely rich dataset. The strain gauge values are in units of millivolts.

## Dataset usage

## Tables
 
The tables containing the toas are:

* `toas_100_s`
* `toas_110_s`
* `toas_120_s`
* `toas_30_s`
* `toas_40_s`
* `toas_50_s`
* `toas_60_s`
* `toas_70_s`
* `toas_80_s`
* `toas_90_s`

Each toa table has the following channels:
 
* `Kraken OPR`
* `OROS OPR`
* `OROS MPR`
* `Kraken Probe `
* `Kraken Probe 2`
* `OROS IP Probe `
* `OROS IP Probe 2`
* `OROS IP Probe 3`

The tables containing the strain gauge data are:

* `sg_100_s`
* `sg_110_s`
* `sg_120_s`
* `sg_30_s`
* `sg_40_s`
* `sg_50_s`
* `sg_60_s`
* `sg_70_s`
* `sg_80_s`
* `sg_90_s`

## Getting started 👇

### Querying the strain gauge data

The strain gauge data is not stored with a time column. You need to create a timestamp column either after reading the data, or during the read operation.

For instance, if you want to read the strain gauge data for the 30s ramp rate test, you can do the following:

```python
from bladesight import Datasets

ds = Datasets["data/diamond_et_al/2024_ramp_rates_45_deg"]

df_sg_1 = ds.query(
  """
  SELECT
    (row_number() OVER () -  1)*(1/25.6e3) as t,
    voltage,
  FROM 
    sg_30_s
  """
)
```

### Using the OPR as a reference
You have two choices regarding the OPR sensor. The first is to use the OPR zero-crossing times as measured using the Kraken X4. This channel is called 'Kraken OPR' in the toa datasets. 

There are 6 voltage levels you can choose from (-0.3, -0.2, 0.1, 0.2) and you can either recover the zero-crossing times from the rising slope or the falling slope.

For instance, say you want to recover the zero-crossing times from the falling slope of OPR sensor at -0.3 V, you can use the following query:

```python
from bladesight import Datasets

ds = Datasets["data/diamond_et_al/2024_ramp_rates_45_deg"]

df_opr_kraken = ds.query(
  """
  SELECT
    toa
  FROM 
    toas_30_s
  WHERE
    channel = 'Kraken OPR'
  AND 
    voltage = -0.3
  AND is_rising = False
  """
)
```

Alternatively, you can select all from the OROS OPR channel. There is only one voltage level, 0 V, for this channel.

```python
from bladesight import Datasets

ds = Datasets["data/diamond_et_al/2024_ramp_rates_45_deg"]

df_opr_oros = ds.query(
  """
  SELECT
    toa
  FROM 
    toas_30_s
  WHERE
    channel = 'OROS OPR'
  """
)
```

### Using the MPR as a reference

Only the OROS was used to capture the MPR data. The MPR data is stored in the 'OROS MPR' channel.

```python
from bladesight import Datasets

ds = Datasets["data/diamond_et_al/2024_ramp_rates_45_deg"]

df_mpr = ds.query(
  """
  SELECT
    toa
  FROM 
    toas_30_s
  WHERE
    channel = 'OROS MPR'
  """
)
```

There are 78 sections in the MPR data.


## Conclusion


## Acknowledgements

A big thanks to the University of Pretoria for providing the test bench and the laboratory space to conduct these tests.

\bibliography

<div style='display:flex;flex-direction:column'>
    <div style='display:flex;flex-direction:row'>
      <div>
          <a target="_blank" href="https://www.bladesight.com" class="" title="Dawie Diamond" style="border-radius:100%;"> 
              <img src="https://github.com/Bladesight.png?size=300" alt="Dawie Diamond" style="border-radius: 100%;width: 4.0rem;">
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
    
    <div style='display:flex;flex-direction:row'>
    <div>
        <a target="_blank" href="https://www.linkedin.com/in/justin-s-507338116/" class="" title="Justin Smith" style="border-radius:100%;"> 
            <img src="https://lh3.googleusercontent.com/a-/ALV-UjXWpiVzpdjYOvSQxPJMjYoqfg8a6oXzxJ9rGWJC0m8TCNFkksin=s80-p-k-rw-no" alt="Justin Smith" style="
            border-radius: 100%;
            width: 4.0rem;
        ">
        </a>
    </div>
    <div style='margin-left:2rem'>
        <p>
            <strong>Justin Smith</strong>
        </p>
        <p>
            2024-11-XX
        </p>
    </div>


    </div>

</div>
