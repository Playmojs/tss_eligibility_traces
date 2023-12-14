from LIF_theta_model3 import *

output_files = "data/december/ThM3_6000s_"
n_simulations = 3

for i in range(n_simulations):
    print(f"Starting simulation {i + 1} out of {n_simulations}")
    gridSimulation(48, 13, 0.1, 3, False, False, 1000000, False, True, 1000, output_files + str(i))
