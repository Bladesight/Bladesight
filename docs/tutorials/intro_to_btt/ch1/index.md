---
date: 2023-09-18
tags:
  - Blade Tip Timing
  - BTT
  - Non Intrusive Stress Measurement
  - NSMS
  - Turbine blade
  - Mechanical Vibration
hide:
  - tags
description: This chapter explains how Blade Tip Timing is derived from Time of Arrival's.
robots: index, follow, Blade Tip Timing, BTT, Non Intrusive Stress Measurement, NSMS, Turbine blade,Mechanical Vibration
template: main_intro_to_btt.html
card_title: Intro to BTT Ch1 - Blade Tip Timing's Fundamental Principle
card_url: "ch1/"
---
# How BTT works
George Berkely, a 17th century philosopher, created beautiful literature while critiquing Newton's approach to calculus. Referring to one of the entities in Newton's equations, he wrote:

!!! quote "Quote [@georgeberkeleyquote]"

    And what are these fluxions? The velocities of evanescent increments? They are neither finite quantities, nor quantities infinitely small, nor yet nothing. May we not call them __ghosts of departed quantities__?

He likely expressed this viewpoint because the infinitesimally small entities required to make calculus work could not be understood through conventional algebraic techniques.

The same critique can be applied to Blade Tip Timing (BTT).

Our objective is to quantify __tip deflection__, yet conventional methods like placing an accelerometer on the blade, directing a Laser Doppler Vibrometer (LDV) towards it, or using a strain gauge are impractical.

Consequently, we must adopt an indirect approach. We must measure a *related quantity* from which the tip deflection can be *inferred*. The challenges encountered in BTT largely emerge from using this indirect approach.

We therefore start with what we *can* measure, which is __time__. To paraphrase George Berkeley, we can say that BTT derives tip deflections using the *ghosts of departed timestamps*.


## BTT's Fundamental Principle

Suppose you have a rotor, and your goal is to assess the stress experienced by its blades. We have two crucial pieces of information about this structure:

1. The rotor blades maintain fixed shapes.
2. The blades are firmly affixed to the shaft.

The rigid connection between the shaft and the blades grants us the ability to calculate the blade tip positions based on the shaft's angular position.

However, if we were to calculate the anticipated position of a blade over numerous shaft revolutions and compare that to the actual position, we would discover that our simple yet elegant theory above is flawed. 

In reality, the blade is almost never  found at the expected location. Our first assumption is, in fact, incorrect. The blade's shape constantly changes because it *vibrates*.

<figure markdown>
  ![fundamental_btt_principle](./ch01_fundamental_principle.png){ width="700" }
  <figcaption><strong><a name='figure_01'>Figure 1</a></strong>:The core concept underlying Blade Tip Timing (BTT) is as follows: As the rotor blades revolve, they pass by a sensor installed into the rotor's casing. When a blade traverses this sensor, it generates a pulse in the probe's signal. The timing of this pulse reveals the precise position of the blade's tip. If the pulse occurs ahead of the expected time, we can infer that the blade's tip is positively displaced, while the reverse is true for a negative displacement. </figcaption>
</figure>


If you're new to BTT, I suggest a thorough examination of [Figure 1](#figure_01) above. In it, we observe two fictional blades exhibiting different behaviors. First, consider blade "A," which solely undergoes rotation without any vibration. On every occasion the blade rotates past the sensor, we register the arrival time of the blade at the sensor. These timestamps can be regarded as the *expected* arrival times.

Now, let's shift our focus to blade "B." Like blade "A," it also generates a sequence of pulses in the sensor. Blade "B" is, however, experiencing vibration on top of its standard rotation. The pulses stemming from blade "B" occur at different times compared to those from blade "A." We call these timestamps from Blade "B" the *actual* arrival times. Consequently, its pulses either precede or lag behind the expected timing. The greater the displacement of the blade tip, the larger this time difference becomes.

If you grasp [Figure 1](#figure_01), you grasp the fundamental concept at the heart of BTT. However, don't underestimate the challenge that lies ahead. If tip deflections are *ghosts of departed timestamps*, they do not reveal themselves easily.    

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
