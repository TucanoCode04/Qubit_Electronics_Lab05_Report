import sys
sys.path.append('/opt/qtcad-2.1.4/qtcad/')


from qtcad.device import constants as ct
from qtcad.device.mesh3d import Mesh, SubMesh
from qtcad.device import io
from qtcad.device import analysis
from qtcad.device import materials as mt
from qtcad.device import Device, SubDevice
from qtcad.device.poisson import Solver as PoissonSolver
from qtcad.device.schrodinger import Solver as SchrodingerSolver
from qtcad.device.schrodinger import SolverParams as SchrodingerSolverParams 
from qtcad.device.schrodinger_poisson import Solver as PoissonSchrodingerSolver
from qtcad.device.poisson import SolverParams as PoissonSolverParams

from qtcad.device.leverarm_matrix import Solver as LeverArmSolver
import numpy as np
import pathlib

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#
#       Input physical parameters
#
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Input voltages
V_Y_gate_1                   = 0.535
V_Y_gate_2                   = 0.535
V_region_Reservoir_Left = 1.3 # Left reservoir voltage [V]
V_region_Reservoir_Right = 1.3 # Right reservoir voltage [V]
V_Barrier_1          = 0.040 # Left barrier voltage [V]
V_Barrier_2          = 0.4 # Interdot barrier voltage [V]
V_Barrier_3          = 0.04 # Right barrier voltage [V]
V_Plunger_Left          = 0.70# Left dot voltage [V]
V_Plunger_Right          = 0.70+0.0001990577078 # Right dot voltage [V]
#Vbottom=0 #ground
mWF= mt.Si.chi+ mt.Si.Eg/2 #electron affinity plus half of energy gap

#doping
n_doping=5e18*1e6
p_doping=0
#Temperature K
T=0.015



script_dir = pathlib.Path(__file__).parent.resolve()
path_mesh = script_dir / "meshes" / "DQD_SiMOS.msh2"
path_geo = script_dir / "meshes" / "DQD_SiMOS.geo_unrolled"
path_hdf5 = script_dir / "output" / "DQD_SiMOS.hdf5"
path_vtu_EV = script_dir / "output" / "DQD_SiMOS_EV.vtu"
path_vtu_n = script_dir / "output" / "DQD_SiMOS_n.vtu"
path_vtu_EC = script_dir / "output" / "DQD_SiMOS_EC.vtu"
path_vtu_p = script_dir / "output" / "DQD_SiMOS_p.vtu"
path_vtu_phi = script_dir / "output" / "DQD_SiMOS_phi.vtu"
path_vtu_q = script_dir / "output" /    "DQD_SiMOS_q.vtu"
path_energies = script_dir / "output" / "DQD_SiMOS_energies.txt"



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#
#       Mesh
#
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Load Mesh
scalingFactor = 1e-9
mesh = Mesh(scalingFactor, str(path_mesh))
#mesh.show()
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#
#     Device
#
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
d = Device(mesh, conf_carriers="e")
d.set_temperature(T)
d.statistics = "FD_approx" #Fermi Dirac statistic for carrier densities

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#
#    Different Region and Boundaries Condition Definition
#
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

d.new_region('region_bulk_bottom',mt.Si)

d.new_region("region_pre_quantum_A_dop_left",       mt.Si, pdoping=p_doping, ndoping=n_doping)
d.new_region("region_pre_quantum_A_mid",            mt.Si)
d.new_region("region_pre_quantum_A_dop_right",      mt.Si, pdoping=p_doping, ndoping=n_doping)
d.new_region("region_pre_quantum_B_dop_left",       mt.Si)
d.new_region("region_pre_quantum_B_left",           mt.Si)
d.new_region("region_pre_quantum_B_mid",            mt.Si)
d.new_region("region_pre_quantum_B_right",          mt.Si)
d.new_region("region_pre_quantum_B_dop_right",      mt.Si)
d.new_region("region_pre_quantum_C_dop_left",       mt.Si, pdoping=p_doping, ndoping=n_doping)
d.new_region("region_pre_quantum_C_mid",            mt.Si)
d.new_region("region_pre_quantum_C_dop_right",      mt.Si, pdoping=p_doping, ndoping=n_doping)

d.new_region("region_quantum_A_dop_left",       mt.Si, pdoping=p_doping, ndoping=n_doping)
d.new_region("region_quantum_A_mid",            mt.Si)
d.new_region("region_quantum_A_dop_right",      mt.Si, pdoping=p_doping, ndoping=n_doping)
d.new_region("region_quantum_B_dop_left",       mt.Si)
d.new_region("region_quantum_B_left",           mt.Si)
d.new_region("region_quantum_B_mid",            mt.Si)
d.new_region("region_quantum_B_right",          mt.Si)
d.new_region("region_quantum_B_dop_right",      mt.Si)
d.new_region("region_quantum_C_dop_left",       mt.Si, pdoping=p_doping, ndoping=n_doping)
d.new_region("region_quantum_C_mid",            mt.Si)
d.new_region("region_quantum_C_dop_right",      mt.Si, pdoping=p_doping, ndoping=n_doping)

d.new_region("region_post_quantum_A_dop_left",       mt.Si, pdoping=p_doping, ndoping=n_doping)
d.new_region("region_post_quantum_A_mid",            mt.SiO2)
d.new_region("region_post_quantum_A_dop_right",      mt.Si, pdoping=p_doping, ndoping=n_doping)
d.new_region("region_post_quantum_B_dop_left",       mt.SiO2)
d.new_region("region_post_quantum_B_left",          mt.SiO2)
d.new_region("region_post_quantum_B_mid",           mt.SiO2)
d.new_region("region_post_quantum_B_right",          mt.SiO2)
d.new_region("region_post_quantum_B_dop_right",      mt.SiO2)
d.new_region("region_post_quantum_C_dop_left",       mt.Si, pdoping=p_doping, ndoping=n_doping)
d.new_region("region_post_quantum_C_mid",           mt.SiO2)
d.new_region("region_post_quantum_C_dop_right",     mt.Si, pdoping=p_doping, ndoping=n_doping)

d.new_region("region_bulk_top_A_dop_left",       mt.Si, pdoping=p_doping, ndoping=n_doping)
d.new_region("region_bulk_top_A_dop_right",      mt.Si, pdoping=p_doping, ndoping=n_doping)
d.new_region("region_bulk_top_B_dop_left",       mt.SiO2)
d.new_region("region_bulk_top_B_dop_right",      mt.SiO2)
d.new_region("region_bulk_top_C_dop_left",      mt.Si, pdoping=p_doping, ndoping=n_doping)
d.new_region("region_bulk_top_C_dop_right",      mt.Si, pdoping=p_doping, ndoping=n_doping)
d.new_region("region_bulk_top_mid",             mt.SiO2)
d.new_region('region_top_oxide', mt.Al2O3)

d.new_region('region_Y1_gate', mt.Al2O3)
d.new_region('region_Y2_gate', mt.Al2O3)
d.new_region('region_Reservoir_L_X_gate', mt.Al2O3)
d.new_region('region_Reservoir_R_X_gate', mt.Al2O3)
d.new_region('region_Barrier_L_X_gate', mt.Al2O3)
d.new_region('region_Barrier_M_X_gate', mt.Al2O3)
d.new_region('region_Barrier_R_X_gate', mt.Al2O3)
d.new_region('region_Plunger_L_X_gate', mt.Al2O3)
d.new_region('region_Plunger_R_X_gate', mt.Al2O3)

#d.new_ohmic_bnd('surface_ohmic_left')
#d.new_ohmic_bnd('surface_ohmic_right')
d.new_gate_bnd('surface_Y1_gate',           V_Y_gate_1                   , mWF)
d.new_gate_bnd('surface_Y2_gate',           V_Y_gate_2                   , mWF)
d.new_gate_bnd('surface_Reservoir_L_X_gate',V_region_Reservoir_Left , mWF)
d.new_gate_bnd('surface_Reservoir_R_X_gate',V_region_Reservoir_Right , mWF)
d.new_gate_bnd('surface_Barrier_L_X_gate',  V_Barrier_1          , mWF)
d.new_gate_bnd('surface_Barrier_M_X_gate',  V_Barrier_2          , mWF)
d.new_gate_bnd('surface_Barrier_R_X_gate',  V_Barrier_3          , mWF)
d.new_gate_bnd('surface_Plunger_L_X_gate',  V_Plunger_Left          , mWF)
d.new_gate_bnd('surface_Plunger_R_X_gate',  V_Plunger_Right          , mWF)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#
#      Defining the quantum dot
#
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

dot_region=["region_pre_quantum_B_mid","region_quantum_B_mid","region_post_quantum_B_mid"]
# Set up the dot region in which no classical charge is allowed
d.set_dot_region(dot_region)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#
#      Poisson Solver
#
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Configure the non-linear Poisson solver
params_poisson = PoissonSolverParams()
params_poisson.tol = 1e-7 # Convergence threshold (tolerance) for the error
params_poisson.initial_ref_factor = 0.55
params_poisson.final_ref_factor = 0.85

params_poisson.min_nodes = 0
params_poisson.maxiter = 1000
params_poisson.refined_region = dot_region
params_poisson.h_refined = 0.3
params_poisson.max_nodes=1e10

# Create an adaptive-mesh non-linear Poisson solver
p = PoissonSolver(d, solver_params=params_poisson, geo_file=path_geo)

# Self-consistent solution
p.solve()

# Schrodinger configuration
num_states = 5
params_schrod = SchrodingerSolverParams()
params_schrod.num_states = num_states
params_schrod.tol = 1e-9

d.set_V_from_phi()

submesh = SubMesh(d.mesh, dot_region)
subd = SubDevice(d,submesh)

schrod_solver = SchrodingerSolver(subd, solver_params=params_schrod)
schrod_solver.solve()

subd.print_energies()
energies = subd.energies
np.save(script_dir/"output"/"DQD_SiMOS_energies.npy",energies)

# To save throughout the device: n, p, phi, EC, EV
arrays_dict = {"n": d.n/1e6, "p": d.p/1e6, "phi": d.phi,
   "EC": d.cond_band_edge()/ct.e, "EV": d.vlnce_band_edge()/ct.e}
io.save(str(path_hdf5), arrays_dict)
#io.save(str(path_vtu), arrays_dict, d.mesh)
arrays_dict = {"n": d.n/1e6}
io.save(str(path_vtu_n), arrays_dict, d.mesh)

arrays_dict = {"p": d.p/1e6}
io.save(str(path_vtu_p), arrays_dict, d.mesh)

arrays_dict = {"phi": d.phi}
io.save(str(path_vtu_phi), arrays_dict, d.mesh)

arrays_dict = {"EC": d.cond_band_edge()/ct.e}
io.save(str(path_vtu_EC), arrays_dict, d.mesh)

arrays_dict = {"EV": d.vlnce_band_edge()/ct.e}
io.save(str(path_vtu_EV), arrays_dict, d.mesh)

# Schrodinger
out_dict = {"Ground": np.abs(subd.eigenfunctions[:,0])**2,
    "1st excited": np.abs(subd.eigenfunctions[:,1])**2,
    "2nd excited": np.abs(subd.eigenfunctions[:,2])**2,
    "3rd excited": np.abs(subd.eigenfunctions[:,3])**2,
    "4th excited": np.abs(subd.eigenfunctions[:,4])**2,
    "conf-potential": subd.V/ct.e}
io.save(str(path_vtu_q), out_dict, subd.mesh)
with open(path_energies, "w") as f:
   f.write('Energy eigenvalues [eV]')
   f.write(str(subd.energies/ct.e))
   
subd.print_energies()


