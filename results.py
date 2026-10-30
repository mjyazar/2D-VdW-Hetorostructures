from dataclasses import dataclass
from pathlib import Path
from ase import Atoms
import numpy as np


@dataclass
class Potential:
    data: np.ndarray
    atoms: object
    averaged: np.ndarray
    z: np.ndarray


@dataclass
class PotentialAveraged:
    coordinates: np.ndarray
    planar: np.ndarray


@dataclass
class ChargeDensity:
    data: np.ndarray
    atoms: object
    averaged: np.ndarray
    z: np.ndarray

@dataclass
class ChargeDensityAveraged:
    coordinates: np.ndarray
    planar: np.ndarray


@dataclass
class LDOS:
    averaged: np.ndarray
    z: np.ndarray
    energies: np.ndarray
    atoms: Atoms
    box_z: np.ndarray
    fft: np.ndarray  # shape: (n_energies, n1, n2, n_z_box)

@dataclass
class LDOSAveraged:
    energies: np.ndarray
    coordinates: np.ndarray
    planar: np.ndarray


@dataclass
class DOS:
    energy: np.ndarray
    dos: np.ndarray
    idos: np.ndarray

@dataclass
class System:
    name: str
    atoms: Atoms
    path: Path
    fermi_energy: float
    
    potential: Potential
    charge_density: ChargeDensity
    dos: DOS
    ldos: LDOS


@dataclass
class Results:
    coupled: System
    bottom: System
    top: System


@dataclass
class Grid3D:
    data: np.ndarray
    origin: np.ndarray
    cell: np.ndarray
    atoms: object
    
    @property
    def z(self):        
        # number of grid points along z
        nz = self.data.shape[2]

        # total height of the simulation cell
        Lz = self.cell[2, 2]
        
        # distance between neighbouring grid points
        dz = Lz / nz

        # physical z-position of each grid point
        return self.origin[2] + np.arange(nz) * dz
