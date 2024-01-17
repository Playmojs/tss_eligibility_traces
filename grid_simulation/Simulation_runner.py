from LIF_theta_model3 import *

output_files = "data/ThetaMSimuls"
n_simulations = 1

for i in range(n_simulations):
    print(f"Starting simulation {i + 1} out of {n_simulations}")
    distrib = 'regular'
    print(f"Distribution: {distrib}")
    gridSimulation(24, 13, 0.1, 6000, False, distrib, 1000000, False, False, False, True, 10000, output_files + "_" + distrib + str(i))
