from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent

OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
POTENTIAL_DIR = FIG_DIR / "potential"
CHARGE_DENSITY_DIR = FIG_DIR / "charge_density"
DOS_DIR = FIG_DIR / "dos"
LDOS_DIR = FIG_DIR / "ldos"
CONV_FIG_DIR = ROOT / "convergence" / "figures"
CONV_DATA_DIR = ROOT / "convergence" / "data"

# Ensure folders exist
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
POTENTIAL_DIR.mkdir(exist_ok=True)
CHARGE_DENSITY_DIR.mkdir(exist_ok=True)
DOS_DIR.mkdir(exist_ok=True)
LDOS_DIR.mkdir(exist_ok=True)
CONV_FIG_DIR.mkdir(parents=True, exist_ok=True)
CONV_DATA_DIR.mkdir(parents=True, exist_ok=True)


"""
RUN
False for efficient run if data from previous run present, True if running for the first time or want to create new files with new parameters
"""
RUN_QE = True
RUN_POTENTIAL = RUN_QE
RUN_CHARGE_DENSITY = RUN_QE
RUN_DOS = RUN_QE
RUN_LDOS = RUN_QE
RUN_CONVERGENCE = False


"""
RUNNER
"""
NPROC = int(os.environ.get("QE_NPROC", "16"))
NK = int(os.environ.get("QE_NK", "8"))  # split kgrid computations into n pools with


"""
vdW SCHEME
"""
VDW_CORR = "grimme-d3"
XC = [None, "vdw-df2-c09"]


"""
CONSTANTS AND CONVERSIONS
"""
BOHR_TO_ANGSTROM = 0.529177210903  # Bohr radius
RY_TO_EV = 13.605693122994
EV_TO_RY = 1 / RY_TO_EV


"""
STRUCTURE
"""
VACUUM = 10.0  # QE requires 2D Coulomb truncation of the cell to have min z-length ~10.58 A
INTERLAYER_DISTANCE = 3.35  # for bilayer (angstrom)
STACKING = "AB"

LATTICE_CONSTANT_GRAPHENE = 2.46
# https://onlinelibrary.wiley.com/doi/epdf/10.1002/andp.201400128
LATTICE_CONSTANT_MOSE2 = 3.288
LATTICE_CONSTANT_WSE2 = 3.260
LATTICE_CONSTANT_MOS2 = 3.160
LATTICE_CONSTANT_WS2 = 3.153

SE_HEIGHT_MOSE2 = 1.664#A
SE_HEIGHT_WSE2  = 1.657#A
S_HEIGHT_MOS2 = 1.586#A
S_HEIGHT_WS2  = 1.571#A


"""
Carbon pseudopotential
from https://sssp.materialscloud.org/pseudopotentials/PBE/efficiency
"""
PSEUDO_C = "C.us.pbe.z_4.uspp.gbrv.v1.2.upf"
PSEUDO_Mo = "Mo.paw.pbe.z_14.atompaw.jth.v1.1-std.upf"
PSEUDO_W = "W.nc.pbe.z_14.oncvpsp4.spms.v1.upf"
PSEUDO_S = "S.nc.pbe.z_6.oncvpsp4.spms.v1.upf"
PSUEDO_Se = "Se.us.pbe.z_6.uspp.gbrv.v1.upf"

PSEUDOPOTENTIALS = {"C":PSEUDO_C, "Mo":PSEUDO_Mo, "W":PSEUDO_W, "Se":PSUEDO_Se, "S":PSEUDO_S}


"""
pw.x
"""
ECUTWFC = 90.0
ECUTRHO = 4 * ECUTWFC
CONV_THRESHOLD = 1.0e-8
DEGAUSS_PW = 0.01
SMEARING = "gauss"
FORCE_CONVERGENCE_THRESHOLD = 0.001
BANDPATH = 'GMKG'
Z_VALENCE = {"C": 4, "Mo": 14, "W": 14, "S": 6, "Se": 6}
EXTRA_BANDS = {"relax": 0.2, "scf": 0.2, "nscf": 3.5} 


"""
pp.x
"""
DEGAUSS_LDOS = 0.2
WINDOW_LDOS = (-10, 10)
DELTA_E_LDOS = 0.05
VACUUM_LEVEL_TOLERANCE = 0.0000001  # do not go lower -> error
INTERFACE_BOX_WIDTH = 1.0


"""
dos.x
"""
DEGAUSS_DOS = DEGAUSS_LDOS * EV_TO_RY
WINDOW_DOS = WINDOW_LDOS
DELTA_E_DOS = 0.01


"""
PLOTTER
"""
WINDOW = WINDOW_LDOS
PLOT_WINDOW_LDOS = (-10, 3)
LDOS_GRID_DELTA_E = 0.01
FIGSIZE_CHARGE_DENSITY = (5, 8)
LDOS_WEIGHT = 5


"""
CONVERGENCE
"""
ECUTWFC_BOUND = 110


"""
ALIGNMENT
"""
VACUUM_REFERENCE_WINDOW = (3, 6)
Z_REFERENCE = 5
