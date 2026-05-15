import sys
sys.path.append("/opt/qtcad-2.1.4/src/qtcad")
__copyright__ = "Copyright 023, Nanoacademic Technologies Inc."

import numpy as np
import pathlib
from device import io
from matplotlib import pyplot as plt
from progress.bar import ChargingBar as Bar
from device import constants as ct
from device import analysis as an
from device.mesh3d import Mesh, SubMesh
from device import materials as mt
from device import Device, SubDevice
from device.poisson import Solver as PoissonSolver
from device.poisson import SolverParams as PoissonSolverParams
from device.schrodinger import Solver as SchrodingerSolver
from device.schrodinger import SolverParams as SchrodingerSolverParams
from device.many_body import Solver as ManyBodySolver
from device.many_body import SolverParams as ManyBodySolverParams
from device.leverarm_matrix import Solver as LeverArmSolver
from transport.junction import Junction
from transport.mastereq import add_spectrum

# Set up paths
script_dir = pathlib.Path(__file__).parent.resolve()

# Mesh path
path_mesh = script_dir / "meshes"
path_out = script_dir / "output"

path_hdf5 = script_dir / "output" /     "DQD.hdf5"
path_device = script_dir / "output" /   "DQD-device.hdf5"
path_vtu = script_dir / "output" /      "DQD.vtu"
path_vtu_n = script_dir / "output" /    "DQD-n.vtu"
path_vtu_p = script_dir / "output" /    "DQD-p.vtu"
path_vtu_phi = script_dir / "output" /  "DQD-phi.vtu"
path_vtu_EC = script_dir / "output" /   "DQD-EC.vtu"
path_vtu_EV = script_dir / "output" /   "DQD-EV.vtu"
path_vtu_q = script_dir / "output" /    "DQD-q.vtu"
path_energies = script_dir / "output" / "DQD-energies.txt"


# Load the mesh
scaling = 1e-9
mesh = Mesh(scaling, str(path_mesh / "dqdfdsoi.msh2"))

# Define the device object
dvc = Device(mesh, conf_carriers='e')
dvc.set_temperature(0.1)

# Create the regions
dvc.new_region("oxide", mt.SiO2)
dvc.new_region("oxide_dot", mt.SiO2)
dvc.new_region("gate_oxide", mt.HfO2)
dvc.new_region("gate_oxide_dot", mt.HfO2)
dvc.new_region("buried_oxide", mt.SiO2)
dvc.new_region("buried_oxide_dot", mt.SiO2)
dvc.new_region("channel", mt.Si)
dvc.new_region("channel_dot", mt.Si)
dvc.new_region("source", mt.Si, ndoping=1e20*1e6)
dvc.new_region("drain", mt.Si, ndoping=1e20*1e6)

# Define some device parameters
back_gate_bias = -0.5
barrier_gate_1_bias = 0.5
plunger_gate_1_bias = 0.6
barrier_gate_2_bias = 0.5+100e-6
plunger_gate_2_bias = 0.6
barrier_gate_3_bias = 0.5

# Set up boundary conditions
Ew = mt.Si.Eg/2 + mt.Si.chi # Midgap
dvc.new_gate_bnd("barrier_gate_1_bnd", barrier_gate_1_bias, Ew)
dvc.new_gate_bnd("plunger_gate_1_bnd", plunger_gate_1_bias, Ew)
dvc.new_gate_bnd("barrier_gate_2_bnd", barrier_gate_2_bias, Ew)
dvc.new_gate_bnd("plunger_gate_2_bnd", plunger_gate_2_bias, Ew)
dvc.new_gate_bnd("barrier_gate_3_bnd", barrier_gate_3_bias, Ew)
dvc.new_ohmic_bnd("source_bnd")
dvc.new_ohmic_bnd("drain_bnd")
dvc.new_frozen_bnd("back_gate_bnd", back_gate_bias, mt.Si, 1e15*1e6, 
   "n", 46*1e-3*ct.e)

# Create the double quantum dot region
dot_region_list = ["oxide_dot", "gate_oxide_dot", "buried_oxide_dot", "channel_dot"]
dvc.set_dot_region(dot_region_list)

# Configure the non-linear Poisson solver
params_poisson = PoissonSolverParams()
params_poisson.tol = 1e-3 # Convergence threshold (tolerance) for the error
params_poisson.initial_ref_factor = 0.1
params_poisson.final_ref_factor = 0.75
params_poisson.min_nodes = 50000
params_poisson.max_nodes = 1e5
params_poisson.maxiter_adapt = 30
params_poisson.maxiter = 200
params_poisson.dot_region = dot_region_list
params_poisson.h_dot = 0.8
params_poisson.size_map_filename = str(path_mesh / "refined_dqdfdsoi.pos")
params_poisson.refined_mesh_filename = str(path_mesh / "refined_dqdfdsoi.msh2")

# Number of single-electron orbital states to be considered
num_states = 4

# Instantiate Schrodinger solver's parameters
params_schrod = SchrodingerSolverParams()
params_schrod.num_states = num_states    # Number of states to consider

# Instantiate Poisson solver
poisson_slv = PoissonSolver(dvc, solver_params=params_poisson, 
   geo_file=str(path_mesh/"dqdfdsoi.geo_unrolled"))

# Solve Poisson's equation
poisson_slv.solve()

# Get the potential energy from the band edge for usage in the Schrodinger
# solver
dvc.set_V_from_phi()

# Create a submesh including only the dot region
submesh = SubMesh(dvc.mesh, dot_region_list)

# Create a subdevice for the dot region
subdvc = SubDevice(dvc, submesh)

# Create a Schrodinger solver
schrod_solver = SchrodingerSolver(subdvc, solver_params=params_schrod)

# Solve Schrodinger's equation
schrod_solver.solve()

# Output and save single-particle energy levels
subdvc.print_energies()
energies = subdvc.energies
np.save(path_out/"dqdfdsoi_energies.npy",energies)

arrays_dict = {"n" : dvc.n/1e6, "p" : dvc.p/1e6, "phi" : dvc.phi, "EC" : dvc.cond_band_edge()/ct.e, "EV": dvc.vlnce_band_edge()/ct.e}
io.save(str(path_hdf5), arrays_dict)
io.save(str(path_vtu_n), arrays_dict, dvc.mesh)

arrays_dict = {"p": dvc.p/1e6}
io.save(str(path_vtu_p), arrays_dict, dvc.mesh)

arrays_dict = {"phi": dvc.phi}
io.save(str(path_vtu_phi), arrays_dict, dvc.mesh)

arrays_dict = {"EC": dvc.cond_band_edge()/ct.e}
io.save(str(path_vtu_EC), arrays_dict, dvc.mesh)

arrays_dict = {"EV": dvc.vlnce_band_edge()/ct.e}
io.save(str(path_vtu_EV), arrays_dict, dvc.mesh)


# Schroedinger
out_dict = {"Ground": np.abs(subdvc.eigenfunctions[:,0])**2,
    "1st excited": np.abs(subdvc.eigenfunctions[:,1])**2,
    "2nd excited": np.abs(subdvc.eigenfunctions[:,2])**2,
    "3rd excited": np.abs(subdvc.eigenfunctions[:,3])**2,
    "conf-potential": subdvc.V/ct.e}
io.save(str(path_vtu_q), out_dict, subdvc.mesh)
with open(path_energies, 'w') as f:
    f.write('Energy eigenvalues [eV]')
    f.write(str(subdvc.energies/ct.e))

subdvc.print_energies() 