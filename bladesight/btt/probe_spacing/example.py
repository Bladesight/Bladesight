"""
This is an example of how to use the probe_spacing module to calculate the probe spacing following 
from the example presented in 
[1] D. H. Diamond and P. S. Heyns, 'A Novel Method for the Design of Proximity Sensor Configuration for Rotor Blade Tip Timing',
Journal of Vibration and Acoustics, vol. 140, no. 6, p. 061003, Dec. 2018, doi: 10.1115/1.4039931.
"""

# Locations of the Fixtures, see Table 5 in reference [1]:
Fixture_1 = np.deg2rad([0, 80]) #degrees
Fixture_2 = np.deg2rad([123, 130]) #degrees
Fixture_3 = np.deg2rad([256, 330]) #degrees

print("Fixture 1 = ", Fixture_1)
print("Fixture 2 = ", Fixture_2)
print("Fixture 3 = ", Fixture_3)

difference_1_to_2 = Fixture_2[1] - Fixture_1[1]
difference_2_to_3 = Fixture_3[0] - Fixture_2[1]
print("Difference 1 to 2 = ", np.rad2deg(difference_1_to_2), 'degrees')
print("Difference 1 to 2 = ", difference_1_to_2, 'radians')
print("Difference 2 to 3 = ", np.rad2deg(difference_2_to_3), 'degrees')
print("Difference 2 to 3 = ", difference_2_to_3, 'radians')
print("-"*100)

lower_bound = [0.0477]*number_of_probes # min spacing between each probe
lower_bound[0] = float(Fixture_1[1]) #enforcing probe 1 to be after first fixture

# upper_bound = [float((difference_2_to_3)/number_of_probes)]*number_of_probes #Keeping each probe between end of fixture 2 and start of fixture 3
upper_bound = [0.5*np.pi]*number_of_probes #As per recommendations in reference [1]
upper_bound[0] = float(Fixture_2[0]) #enforcing probe 1 to be before second fixture

print("Lower Bound = ", np.rad2deg(lower_bound), 'degrees')
print("Lower Bound = ", lower_bound, 'radians')

print("Upper Bound = ", np.rad2deg(upper_bound), 'degrees')
print("Upper Bound = ", upper_bound, 'radians')

print("-"*100)
print("Lower Bound Probe Positions:", np.rad2deg(np.cumsum(lower_bound)), 'degrees')
print("Upper Bound Probe Positions:", np.rad2deg(np.cumsum(upper_bound)), 'degrees')


# Perform the optimization
best_probe_spacing, best_cost = pso(func=save_loss_progress_PSO, args=(use_equidistant_spacing_bool, EO_array, number_of_probes),
                                    lb=lower_bound, ub=upper_bound, 
                                    swarmsize=num_particles, maxiter=num_iterations,
                                    # f_ieqcons = constraints_PSO, 
                                    minstep = 1e-8, minfunc = 1E-12,
                                    debug=True
                                    )