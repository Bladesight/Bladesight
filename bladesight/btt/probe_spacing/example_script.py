"""
This is an example of how to use the probe_spacing module to calculate the probe spacing following 
from the example presented in 
[1] D. H. Diamond and P. S. Heyns, 'A Novel Method for the Design of Proximity Sensor Configuration for Rotor Blade Tip Timing',
Journal of Vibration and Acoustics, vol. 140, no. 6, p. 061003, Dec. 2018, doi: 10.1115/1.4039931.
"""
import numpy as np
from bladesight.btt.probe_spacing import perform_minimization#, plot_best_PSO_loss_per_iter
from bladesight.btt.probe_spacing.optimiser import create_probe_constraints


NUMBER_OF_PROBES = 6
NUMBER_OF_PARTICLES = 100
NUMBER_OF_ITERATIONS = 30
USE_EQUIDISTANT_SPACING_BOOL = False

EO_array = np.arange(1, 32)

MIN_DIST_BETWEEN_SENSORS = 0.0477 # Recommended by reference [1].
MAX_DIST_BETWEEN_SENSORS = 0.5*np.pi # Recommended by reference [1].

# Locations of the Fixtures, see Table 5 in reference [1]:
# Fixture definitions (in radians)
Fixture_1 = np.deg2rad([0, 80])    # forbidden region: (0,80) => we allow positions AFTER the upper bound: 80 deg
Fixture_2 = np.deg2rad([123, 130]) # forbidden region: (123,130) => we want probes either before 123 or after 130
Fixture_3 = np.deg2rad([256, 330]) # forbidden region: (256,330) => we want probes before 256 in this example
MAX_PROBE_POSITION = 2 * np.pi   # 360 degrees

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

FORBIDDEN_INTERVALS = [
        (Fixture_1[0], Fixture_1[1]),
        (Fixture_2[0], Fixture_2[1]),
        (Fixture_3[0], Fixture_3[1])
    ]

# --- Set up an initial guess and bounds ---
lower_bound = [MIN_DIST_BETWEEN_SENSORS] * NUMBER_OF_PROBES
upper_bound = [MAX_DIST_BETWEEN_SENSORS] * NUMBER_OF_PROBES

# NOTE THESE BOUNDS ARE A BIT CONSTRICTIVE IN THE SENSE THAT THEY DO NOT FULLY REPRESENT EQUATION (16) IN REFERENCE [1]
# BUT THEY ARE ADEQUATE TO SOLVE THE PROBLEM.
lower_bound = [MIN_DIST_BETWEEN_SENSORS]*NUMBER_OF_PROBES # min spacing between each probe
lower_bound[0] = float(Fixture_1[1]) #enforcing probe 1 to be after first fixture

# upper_bound = [float((difference_2_to_3)/NUMBER_OF_PROBES)]*NUMBER_OF_PROBES #Keeping each probe between end of fixture 2 and start of fixture 3
# upper_bound = [MAX_DIST_BETWEEN_SENSORS]*NUMBER_OF_PROBES #As per recommendations in reference [1]
# upper_bound[0] = float(Fixture_2[0]) #enforcing probe 1 to be before second fixture
# upper_bound[1] = float(Fixture_2[0]) #enforcing probe 2 to be before first fixture
# upper_bound[2] = float(Fixture_3[1]) #enforcing probe 3 to be before third fixture


print("Lower Bound = ", np.rad2deg(lower_bound), 'degrees')
print("Lower Bound = ", lower_bound, 'radians')

print("Upper Bound = ", np.rad2deg(upper_bound), 'degrees')
print("Upper Bound = ", upper_bound, 'radians')

print("-"*100)
print("Lower Bound Probe Positions:", np.rad2deg(np.cumsum(lower_bound)), 'degrees')
print("Upper Bound Probe Positions:", np.rad2deg(np.cumsum(upper_bound)), 'degrees')

# Create bounds as a list of tuples
bounds = list(zip(lower_bound, upper_bound))
# print("Bounds:", bounds)

# Creating constraints
constraints = create_probe_constraints(
    forbidden_intervals=FORBIDDEN_INTERVALS,
    min_spacing=MIN_DIST_BETWEEN_SENSORS,
    max_spacing=MAX_DIST_BETWEEN_SENSORS,
    overall_max=MAX_PROBE_POSITION,
    number_of_probes=NUMBER_OF_PROBES,
    delta=1e-1  # example delta; adjust as needed
)
# Perform the optimization
# best_probe_spacing, best_cost = pso(func=save_loss_progress_PSO, args=(USE_EQUIDISTANT_SPACING_BOOL, EO_array, NUMBER_OF_PROBES),
#                                     lb=lower_bound, ub=upper_bound, 
#                                     swarmsize=NUMBER_OF_PARTICLES, maxiter=NUMBER_OF_ITERATIONS,
#                                     # f_ieqcons = constraints_PSO, 
#                                     minstep = 1e-8, minfunc = 1E-12,
#                                     debug=False
#                                     )

optimal_result, loss_list = perform_minimization(
    use_equidistant_spacing_bool = USE_EQUIDISTANT_SPACING_BOOL,
    EO_array = EO_array,
    number_of_probes = NUMBER_OF_PROBES,
    x0 = (np.array(lower_bound) + np.array(upper_bound))/2,
    # lower_bound = lower_bound,
    # upper_bound = upper_bound,
    bounds = bounds,
    constraints = constraints,
    # number_of_particles = NUMBER_OF_PARTICLES,
    # number_of_iterations = NUMBER_OF_ITERATIONS,
    debug = False,
    # num_iterations = NUMBER_OF_ITERATIONS
    optimizer = 'trust-constr',
    
    options = {
            'maxiter': 1000,
            # 'tol': 1e-16,
            # 'xatol': 1e-16,
            }
    )
print("optimal_result:", optimal_result)
print("="*200)
best_probe_spacing, best_cost = optimal_result.x, optimal_result.fun

# plot_best_PSO_loss_per_iter(loss_list, NUMBER_OF_PARTICLES)


if USE_EQUIDISTANT_SPACING_BOOL == True:
    best_probe_spacing_list = []
    best_probe_spacing_list.append(0)
    for i in range(NUMBER_OF_PROBES-1):
        best_probe_spacing_list.append(best_probe_spacing[0])
else: 
    best_probe_spacing_list = best_probe_spacing
# Print the best theta values and the corresponding objective function value
print("Best Cost:", best_cost)
print("Best Spacing:", np.rad2deg(best_probe_spacing), 'Degrees')
print("Best Spacing:", best_probe_spacing_list, 'Radians')
print("Best Probe Positions:", np.cumsum(best_probe_spacing_list), 'Radians')
print("Best Probe Positions:", np.rad2deg(np.cumsum(best_probe_spacing_list)), 'Degrees')
# print("Min PSR From Best Probe Position:", np.min(get_PSR(EO_array, (np.sum(best_probe_spacing_list) - best_probe_spacing_list[0]))))
# print("Max PSR From Best Probe Position:", np.max(get_PSR(EO_array, (np.sum(best_probe_spacing_list) - best_probe_spacing_list[0]))))