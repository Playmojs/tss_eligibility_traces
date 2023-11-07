import numpy as np
import sys
import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import BoundaryVectorCells

def generateTrajectory(boundary_vecs, dt, duration_s, output_file):
    Env = Environment(params= {'boundary': boundary_vecs})
    Ag = Agent(Environment= Env, params = {'dt': dt})
    BVCs = BoundaryVectorCells(Agent = Ag)
    dists = np.zeros((int(duration_s/Ag.dt), 180))
    for i in range(int(duration_s/Ag.dt)):
        sys.stdout.write("\rStatus: %3.4f" % ((i+1) * Ag.dt /duration_s))
        sys.stdout.flush()

        Ag.update()
        dists[i] = BVCs.get_state(evaluate_at = 'agent', dist = True)
    np.savez_compressed(f"grid_simulation/Trajectories/{output_file}",\
             positions = Ag.history['pos'], \
             vels = Ag.history['vel'], \
             boundaries = boundary_vecs, \
             boundary_distances = dists)

boundary_square = [[-0.5,-0.5], [0.5,-0.5], [0.5,0.5], [-0.5, 0.5]]
boundary_circular = [[0.5*np.cos(t),0.5*np.sin(t)] for t in np.linspace(0,2*np.pi,100)]
boundary_trapezoid = [[0, -0.2], [0, 0.2], [1.5, 0.5], [1.5, -0.5]]

generateTrajectory(boundary_trapezoid, 0.1, 900, "Trapezoid/test")
