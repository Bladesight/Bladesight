---
date: 2024-01-30
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
card_title: Intro to BTT Ch1 - Blade Tip Timing's fundamental principle
card_url: "ch1/"
---
??? note "Reviews"

    Please read through the chapter prior to activating a review as I have kept the chapters deliberately terse. 
    
    Click to activate a review 👇 for more detailed comments from other experts.
    
    <a onclick="toggleBtnClicked('Mirosław Witoś')" class='md-button md-small review-toggle' data-author="Mirosław Witoś">
        Mirosław Witoś
    </a>

# Blade Tip Timing's fundamental principle

## Who's this tutorial for?
This tutorial is for graduate students who want to use Blade Tip Timing (BTT) to measure rotor blade vibration. One of the earliest prominent BTT references is a 1967 article by Rudolph Hohenberg [@hohenberg1967detection]. There are, however, much older references found in patents [@campbell1924elastic,@mershon1937vibration,@hardigg1949apparatus,@shapiro1958vibration]. BTT has therefore been around for approximately 100 years. It is strange that so few getting-started resources for BTT exist. There have been academic publications that serve this end [@bouckaert2007tip,@witos2007tip], but I am unaware of a single book or tutorial dedicated to BTT. 

<div class='review' data-author='Mirosław Witoś' data-innerhtml="<p>“The earliest BTT reference I could find is a 1967 article by Rudolph Hohenberg.”<p><p>BTT is much older. See e.g.:</p><ul>
    <li>Campbell W. (1924): Elastic fluid turbine rotor and method of avoiding tangential bucket vibration therein, Patent US 1,502,904.
    <li>Mershon A.V. , Schenectady N.Y. (1940): Vibrator indicator, General Electric Company, Patent US 2,204,425.
    <li>Hardigg G.H., Swarthmore P.A. (1951): Apparatus for measuring rotor blade vibration, Westinghouse Electric Corporation, Patent US 2,575,710.
    <li>Shapiro H. (1962): Vibration detector and measuring instrument. Curtiss-Wright Corporation, Propulsion Products Division, Patent US 3058339.</li></ul>" >
</div>

<div class='review' data-author='Mirosław Witoś' data-innerhtml="<p>“I am unaware of a single book or tutorial dedicated to BTT.”</p> <p>For many years, BTT was protected by numerous patents, hence the trace of publications revealing details of the method. There are, however, publications on BTT. See e.g.:</p><il><li> Bouckaert J.F. (ed.) Tip timing and tip clearance problem in turbomachines, VKI Lecture Series 2007-03, von Karman Institute for
Fluid Dynamics 2007.</li><li>Witos M.,  Increasing the durability of turbine engines through active diagnostics and control. Research Works of Air Force Institute of Technology, Issue 29, 2011, p. 1-324,  (pol.), http://dx.doi.org/10.13140/RG.2.1.4341.4560.</li></ol>">
</div>

Instead, students are expected to deep dive into old journal articles and figure it out for themselves. I have no doubt that most mechanical engineering graduate students are intellectually gifted enough to achieve this. The pressure to finish your studies on time, however, can be immense. BTT is a great discipline. You should be excited to learn it. We cannot permit submission dates to ruin the subject.

That's why I've created this tutorial. To help others learn the fundamentals as fast as possible.

I have compiled the most fundamental BTT concepts into a single resource. The core concepts are explained alongside code and example datasets. The Python :simple-python: programming language won the challenge as the language of choice. Python's popularity has surged in recent years, mostly due to its adoption in Machine Learning (ML) and Artificial Intelligence (AI). It is well poised to also take over the other engineering disciplines. Any time spent coding in Python is an excellent investment... and its value scales well beyond BTT.

The coding exercises might take more time than simply reading through the text and nodding along. However, when you grant yourself permission to spend time on the fundamentals, it leads to exponential progress later on.

By the end of this tutorial, you'll have been exposed to the fundamental theory and code that will enable you to process raw BTT data into vibration frequency, amplitude, and phase estimates.

What qualifies me to write a tutorial about BTT?

I do not claim to be the authority on BTT data processing. I have, however, wrestled with BTT for longer than 10 years. I have rewritten BTT algorithms and approaches from scratch several times. Each time, I uncovered something more fundamental. I learned how to cut away unnecessary steps and focus on the essentials. 

Hopefully, you can use what I've started with here... and take it further.

!!! question "Chapter outcomes"

	:material-checkbox-blank-outline: Understand that this tutorial is for graduate students who seek a code-first treatment of the fundamentals of BTT :simple-python:.

    :material-checkbox-blank-outline: Understand that the entire discipline of BTT is built from time stamps.

	:material-checkbox-blank-outline: Understand that each time stamp has several attributes associated with it. BTT is concerned with figuring out these attributes.
	
## Blade Tip Timing and Time Stamps
George Berkely, an 18th century philosopher, created beautiful literature while criticizing Newton's approach to calculus. Puzzled by one of the entities introduced by Newton, he wrote:

!!! quote "Quote [@berkeley1754analyst]"

    And what are these Fluxions? The velocities of evanescent increments? ... They are neither finite Quantities nor Quantities infinitely small, nor yet nothing. May we not call them the __Ghosts of departed Quantities__?

I admire the way he confers a sense of mystery onto infinitesimal entities (i.e. fluxions). This quote often comes to mind when I process BTT data. BTT measurements are not *directly proportional* to the output of the measurement device. This stands in contrast to, for example, accelerometer or strain gauge measurements. You only need to scale these raw voltages to produce the engineering quantity of interest. 

BTT measurements are the *__ghosts of departed time stamps__*.

<div class='review' data-author="Mirosław Witoś" data-innerhtml="<p>“BTT measurements are the ghosts of departed time stamps.”</p><p>Each measurement result is the spirit of an event that was observed and passed.</p> <p>BTT measurements record the effects of mass and aerodynamic forcing acting on rotating blades (encoder transmitter). The deflections and forced vibrations of the blade feathers are expected to be observed by sensors attached to the rotating machine housing.</p>">
</div>

### Time shifts and tip deflections
Rotor blades, being rigid structures, vibrate in response to dynamic loads.  They vibrate because of the aerodynamic forces experienced during operation. BTT exploits the fact that the tips move relative to the shaft's position.

The fundamental concept behind BTT is illustrated in [Figure 1](#figure_01) below. 
<figure markdown>
  ![BTT Fundamentals](BTT_Fundamental_illustration.jpeg){ width="700" }
    <figcaption><strong><a name='figure_01'>Figure 1</a></strong>: An illustration of the fundamental principle behind BTT. </figcaption>
</figure>

On the left hand side of [Figure 1](#figure_01), a rotor is shown. Positions A and B indicate the top blade's minimum and maximum deflections due to vibration. 

A sensor is mounted in the rotor's casing. The sensor generates a voltage pulse in its output signal in response to the presence of the passing blade.

Tip deflections cause the pulse to shift *relative to* the signal produced by a "non-vibrating" blade. If the tip is deflected backward (position A), it causes the pulse to lag. If the tip is deflected forward (position B), it causes the pulse to lead.

The Time of Arrival (ToA) is extracted from each pulse. A ToA is the exact time, in seconds, that a blade is said to have *arrived* at the sensor.

<div class='review' data-author="Mirosław Witoś" data-innerhtml="<p>The BTT method is based on the theoretical underpinnings of a low angular resolution encoder - there is an extensive literature in
robotics, among others. The difference in BTT data processing is to take into account the fact that:</p><ol><li> the phase markers (blade tip) are oscillating, and the encoder (rotating blade disc) may be subject to additional errors in pitch, misalignment, phase marker geometry, physical characteristics of the material (impact of degradation in service) - these components you left
out of the tutorial;</li><li>the shape of the signal from the sensor (the receiving part of the encoder) depends on the type of sensor and its coupling to the rotating blades - this was left out of the tutorial. </li></ol>">

</div>

<div class='review' data-author="Mirosław Witoś" data-innerhtml="<p>You rightly pointed out that in BTT the measurement is indirect - we only record the time of arrival of the blades (TOA). To get from TOA information about the deflection of the blade and the vibration parameters of the blades, first of all:</p><ol><li> correctly perform the measurement (the aspect of pulse triggering, including optimization of the analog bandwidth of the input circuit and
timing accuracy);</li><li>solve the encoder inverse problem with unknown instantaneous speed.</li><li>solve the inverse problem with the assumed model of the observed phenomenon. </li> </ol><p>Task 1) requires knowledge of the metrological characteristics of the sensor signal and the relationship between the characteristics of the electrical signal and the instantaneous position of the blade under the sensor. The purpose of BTT measurement is TOA of the blade, not electrical pulses.</p><p> Task 2) is not trivial for an encoder with a small number (N <128) of rigid phase markers (the encoder measures angular position, not instantaneous speed). Even more so, task 2) is not trivial for an encoder with oscillating phase markers and the aforementioned errors. </p><p> Task 3): In the rest of the tutorial, you assume a linear system describing blade vibration and ignore the effect of a given blade
vibration modulus on the tip amplitude. In this way, you limit the possibility of detecting cracks and the effects of progressive blade
degradation (cyclic strengthening, cyclic weakening). If your tutorial is to be useful in practice, it is worth showing the real capabilities of the BTT method, not just its substitute. </p>">

</div>

### Each ToA is a puzzle to be solved

ToAs have other attributes that must be determined. Each ToA has 6 attributes associated with it. 

These attributes are best defined by a series of questions, as shown in [Figure 2](#figure_02) below.

<figure markdown>
  ![BTT Fundamentals](BTT_Tutorial_Outline_Ch1.svg){ width="500" }
    <figcaption><strong><a name='figure_02'>Figure 1</a></strong>: The outline of this tutorial. Each ToA  has 6 attributes associated with it. Extraction of each ToA is covered in Chapter 2. Chapters 3 - 9 determines the other attributes. </figcaption>
</figure>

I've structured this tutorial to sequentially answer these questions. Each chapter will focus on one or more of the ToA's attributes.

## Conclusion

If you grasp [Figure 1](#figure_01), you grasp the fundamental concept at the heart of BTT. However, don't underestimate the challenge that lies ahead. If tip deflections are, in fact, the ghosts of departed timestamps, they do not reveal themselves easily.    

!!! success "Chapter Outcomes"

	:material-checkbox-marked:{ .checkbox-success .heart } Understand that this tutorial is for graduate students who seek a code-first treatment of the fundamentals of BTT :simple-python:.

    :material-checkbox-marked:{ .checkbox-success .heart } Understand that the entire discipline of BTT is built from time stamps.

	:material-checkbox-marked:{ .checkbox-success .heart } Understand that each time stamp has several attributes associated with it. BTT is concerned with figuring out these attributes.

## Acknowledgements
A big thanks to <a href="https://www.linkedin.com/in/justin-s-507338116/" target="_blank">Justin Smith</a> and <a href="https://www.linkedin.com/in/alex-brocco-70218b25b/" target="_blank">Alex Brocco</a> for their feedback and suggestions regarding this chapter.

A special thanks to <a href="https://www.researchgate.net/profile/Miroslaw-Witos-2" target="_blank">Mirosław Witoś</a> for his detailed review of this chapter.

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
            2024-01-30
        </p>
    </div>
</div>
