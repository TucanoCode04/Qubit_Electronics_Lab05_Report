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


def nodal_volume_weights(mesh):
   # Tetrahedron orientation can make volumes signed; integration weights
   # must use positive physical volumes.
   volumes = np.abs(np.asarray(mesh.tetra_volumes(), dtype=float))
   connectivity = np.asarray(mesh.connectivity, dtype=int)
   if connectivity.ndim != 2:
      raise ValueError("Expected tetrahedral mesh connectivity to be a 2D array")
   if len(volumes) != len(connectivity):
      raise ValueError("Number of tetra volumes does not match mesh connectivity")

   node_number = getattr(mesh, "node_number", int(np.max(connectivity)) + 1)
   weights = np.zeros(node_number)
   local_node_number = connectivity.shape[1]
   np.add.at(
      weights,
      connectivity.ravel(),
      np.repeat(volumes/local_node_number, local_node_number),
   )
   return weights


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
path_out = script_dir / "output"
path_mesh = script_dir / "meshes" / "DQD_SiMOS.msh2"
path_geo = script_dir / "meshes" / "DQD_SiMOS.geo_unrolled"
path_hdf5 = path_out / "DQD_SiMOS.hdf5"
path_vtu_EV = path_out / "DQD_SiMOS_EV.vtu"
path_vtu_n = path_out / "DQD_SiMOS_n.vtu"
path_vtu_EC = path_out / "DQD_SiMOS_EC.vtu"
path_vtu_p = path_out / "DQD_SiMOS_p.vtu"
path_vtu_phi = path_out / "DQD_SiMOS_phi.vtu"
# Signed-wavefunction output: keep it separate from the |psi|^2 q file.
path_vtu_psi = path_out / "DQD_SiMOS_psi_refined_test.vtu"
path_energies = path_out / "DQD_SiMOS_psi_energies.txt"

path_out.mkdir(exist_ok=True)



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
params_poisson.h_refined = 0.1
params_poisson.max_nodes=1e10

# Create an adaptive-mesh non-linear Poisson solver
p = PoissonSolver(d, solver_params=params_poisson, geo_file=str(path_geo))

# Self-consistent solution
p.solve()

# Schrodinger configuration
num_states = 5
params_schrod = SchrodingerSolverParams()
params_schrod.num_states = num_states
params_schrod.tol = 1e-9
params_schrod.method = "robust"
params_schrod.maxiter = 3000

d.set_V_from_phi()

submesh = SubMesh(d.mesh, dot_region)
subd = SubDevice(d,submesh)

schrod_solver = SchrodingerSolver(subd, solver_params=params_schrod)
schrod_solver.solve()

subd.print_energies()
energies = subd.energies
np.save(path_out/"DQD_SiMOS_psi_energies.npy",energies)

# Full-device outputs are not needed to create the Schrodinger q file.
# They are disabled for this refinement test to avoid writing several GB.
# arrays_dict = {"n": d.n/1e6, "p": d.p/1e6, "phi": d.phi,
#    "EC": d.cond_band_edge()/ct.e, "EV": d.vlnce_band_edge()/ct.e}
# io.save(str(path_hdf5), arrays_dict)
# arrays_dict = {"n": d.n/1e6}
# io.save(str(path_vtu_n), arrays_dict, d.mesh)
#
# arrays_dict = {"p": d.p/1e6}
# io.save(str(path_vtu_p), arrays_dict, d.mesh)
#
# arrays_dict = {"phi": d.phi}
# io.save(str(path_vtu_phi), arrays_dict, d.mesh)
#
# arrays_dict = {"EC": d.cond_band_edge()/ct.e}
# io.save(str(path_vtu_EC), arrays_dict, d.mesh)
#
# arrays_dict = {"EV": d.vlnce_band_edge()/ct.e}
# io.save(str(path_vtu_EV), arrays_dict, d.mesh)

# Schrodinger
state_labels = ["Ground", "1st excited", "2nd excited", "3rd excited", "4th excited"]
out_dict = {}
for i, label in enumerate(state_labels):
   psi = np.asarray(subd.eigenfunctions[:, i])
   if np.iscomplexobj(psi):
      imag_max = np.max(np.abs(np.imag(psi)))
      real_max = np.max(np.abs(np.real(psi)))
      if real_max != 0 and imag_max/real_max > 1e-8:
         print(f"Warning: {label} has a non-negligible imaginary component.")
      psi = np.real(psi)

   out_dict[f"{label} psi"] = psi

# Uncomment this line if you want the confinement potential in the same VTU.
# out_dict["conf-potential"] = subd.V/ct.e
io.save(str(path_vtu_psi), out_dict, subd.mesh)
with open(path_energies, "w") as f:
   f.write("Energy eigenvalues [eV]\n")
   for label, energy in zip(state_labels, subd.energies/ct.e):
      f.write(f"{label}: {energy:.12g}\n")
   f.write("\nThe VTU fields contain the signed real wavefunctions psi, not |psi|^2.\n")
   f.write("The overall sign of each eigenfunction is arbitrary.\n")
   
subd.print_energies()
