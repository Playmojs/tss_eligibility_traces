import numpy as np
import sys
import ratinabox as rb
import utils
import matplotlib.path as mpath
import matplotlib.pyplot as plt

def generateTrajectory(dt, duration_s, output_file, boundary_shape = 'square', pos = 'rand', save_to_f = True):
    match boundary_shape:
        case 'square':
             boundary_vecs = [[-0.5,-0.5], [0.5,-0.5], [0.5,0.5], [-0.5, 0.5]]
        case 'circular':
            boundary_vecs = [[0.5*np.cos(t),0.5*np.sin(t)] for t in np.linspace(0,2*np.pi,100)]
        case 'trapezoid':
            boundary_vecs = [[0, -0.2], [0, 0.2], [1.5, 0.5], [1.5, -0.5]]
        case _:
            raise Exception("Invalid boundary shape")


    Env = rb.Environment(params= {'boundary': boundary_vecs})
    Ag = rb.Agent(Environment= Env, params = {'dt': dt})
    BVCs = rb.BoundaryVectorCells(Agent = Ag)
    dists = np.zeros((int(duration_s/Ag.dt), 180))
    for i in range(int(duration_s/Ag.dt)):
        
        if type(pos) == str and pos == 'rand':
            Ag.update()
        else:
            Ag.pos = pos[i]
        dists[i] = BVCs.get_state(evaluate_at = 'agent', dist = True)
    sys.stdout.write("\n")
    if save_to_f:
        np.savez_compressed(f"grid_simulation/Trajectories/{output_file}",\
                positions = Ag.history['pos'], \
                vels = Ag.history['vel'], \
                boundaries = boundary_vecs, \
                boundary_distances = dists)
    return dists

def global_bvc_act(boundary_cells, NBVCs, boundary_shape, pxs):
    match boundary_shape:
        case 'square':
            boundary_vecs = [[-0.5,-0.5], [0.5,-0.5], [0.5,0.5], [-0.5, 0.5]]
            x_min, x_max, y_min, y_max = -0.5, 0.5, -0.5, 0.5
        case 'circular':
            boundary_vecs = [[0.5*np.cos(t),0.5*np.sin(t)] for t in np.linspace(0,2*np.pi,100)]
            x_min, x_max, y_min, y_max = -0.5, 0.5, -0.5, 0.5
        case 'trapezoid':
            boundary_vecs = [[0, -0.2], [0, 0.2], [1.5, 0.5], [1.5, -0.5]]
            x_min, x_max, y_min, y_max = 0, 1.5, -0.5, 0.5
        case _:
            raise Exception("Invalid boundary shape")
    plot_dims = np.array([int(pxs*(y_max - y_min)), int(pxs*(x_max-x_min))])

    mask = utils.maskArea(pxs, boundary_vecs, [x_min, x_max], [y_min, y_max])

    Env = rb.Environment(params={'boundary': boundary_vecs, 'dx': 1/pxs})
    Ag = rb.Agent(Environment= Env, params = {'dt': 0.01})
    BVCs = rb.BoundaryVectorCells(Agent = Ag)
    dists = BVCs.get_state(evaluate_at = 'all', dist = True)
    masked_dists = dists*mask # Mask the distances to avoid overflow later
    acts = utils.BVC_act(boundary_cells, masked_dists, NBVCs, 0)
    y = np.quantile(acts, 0.9)
    masked_acts = acts * mask
    masked_acts[masked_acts<0.92] = 0
    return masked_acts, plot_dims


if __name__ == "__main__":
    #generateTrajectory(0.1, 7200, "Square/7200s", "square")
    Nthetas = 12
    Ndists = 11
    max_dist = np.sqrt(2)
    Nbvcs = Nthetas*Ndists
    boundary_cells = np.meshgrid(np.arange(0,180, int(180/Nthetas), dtype =int), np.linspace(0, max_dist, Ndists))
    boundary_cells = np.reshape(boundary_cells, (2,-1))    

    dists = global_bvc_act(boundary_cells, Nbvcs, 'trapezoid', 25)
    print(np.shape(dists))  