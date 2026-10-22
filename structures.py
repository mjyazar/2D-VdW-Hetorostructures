import numpy as np
from ase import Atoms
# from ase.constraints import FixCartesian

from config import *

def _hexagonal_cell(a, c):
    """
    Hexagonal lattice vectors, same orientation as the graphene cell
    https://mc2d.materialscloud.org/details/mc2d-71
    """
    return np.array([[a, 0, 0],
                     [-a/2, a*np.sqrt(3)/2, 0],
                     [0, 0, c]])


class Graphene:
    def __init__(self, lattice_const=LATTICE_CONSTANT_GRAPHENE, vacuum=VACUUM):
        self.a = lattice_const
        self.vacuum = vacuum


    def cell(self, dz=0.):
        """
        Construct unit cell lattice vectors
        https://mc2d.materialscloud.org/details/mc2d-71
        
        dz: thickness of the layer
        """
                        
        return _hexagonal_cell(self.a, 2*self.vacuum + dz)
    
    
    def monolayer(self):
        """
        Create Monolayer Graphene as Atoms object: honeycomb, 2 atoms/unit cell
        """
        
        # atomic positions in units of the unit cell 
        atomic_positions = np.array([[0, 0, 0.5],      # atom 1
                                     [2/3, 1/3, 0.5]]) # atom 2
        
        return Atoms(symbols="C2", cell=self.cell(), scaled_positions=atomic_positions, pbc=True)
    
    
    def bilayer(self, interlayer_dist=INTERLAYER_DISTANCE, stacking=STACKING):
        """
        Create Bilayer Graphene as an Atoms object
        """
        
        # calculating z coordinates in units of unit cell (i.e. 0 to 1)
        dz = interlayer_dist / (self.vacuum * 2 + interlayer_dist)
        
        # centre the bilayer around around z=0.5
        z1 = 0.5 - dz/2
        z2 = 0.5 + dz/2
            
        if stacking == "AB":
            atomic_positions = np.array([[0, 0, z1],      # atom 1
                                         [2/3, 1/3, z1],  # atom 2
                                         [2/3, 1/3, z2],  # atom 3
                                         [1/3, 2/3, z2]]) # atom 4

        elif stacking == "BA":
            atomic_positions = np.array([[0, 0, z1],
                                         [2/3, 1/3, z1],
                                         [1/3, 2/3, z2],
                                         [0, 0, z2]])

        # must be AA otherwise
        else:
            atomic_positions = np.array([[0, 0, z1],
                                         [2/3, 1/3, z1],
                                         [0, 0, z2],
                                         [2/3, 1/3, z2]])
            
        return Atoms(symbols="C4", cell=self.cell(interlayer_dist), scaled_positions=atomic_positions, pbc=True)

    
    def isolate_bilayer(self, bilayer):
        
        # returns atoms 1 and 2 - in bottom layer
        z = bilayer.positions[:, 2]
        
        mid = (z.min() + z.max()) / 2
        
        bottom = bilayer[np.where(z < mid)[0]].copy()
        top = bilayer[np.where(z >= mid)[0]].copy()

        return bottom, top


class MetalDiselenide:
    def __init__(self, metal, lattice_constant, height, vacuum=VACUUM):
        self.metal = metal
        self.a = lattice_constant
        self.height = height  # distance between metal and Se planes
        self.vacuum = vacuum


    def cell(self, dz=0.):
        """
        dz: z-distance between the lowest and highest atomic planes
        """
        
        return _hexagonal_cell(self.a, 2*self.vacuum + dz)


    def monolayer(self):
        """
        Create monolayer MSe2 as Atoms object, centred at z=0.5
        """
        
        dz = 2 * self.height
        h = self.height / (2*self.vacuum + dz)  # in units of the unit cell
        
        atomic_positions = np.array([[1/3, 2/3, 0.5],      # metal
                                     [2/3, 1/3, 0.5 - h],  # Se bottom
                                     [2/3, 1/3, 0.5 + h]]) # Se top
        
        return Atoms(symbols=f"{self.metal}Se2", cell=self.cell(dz), scaled_positions=atomic_positions, pbc=True)


class GrapheneMDBilayer:
    def __init__(self, bottom, top, interlayer_dist=INTERLAYER_DISTANCE, vacuum=VACUUM):
        self.bottom = bottom.copy()
        self.top = top.copy()
        self.interlayer_dist = interlayer_dist  # distance between facing atomic planes
        self.vacuum = vacuum

    def bilayer(self):
        # match the a of top and bottom layers - numbers assume bottom layer is graphene
        bottom = self.bottom.repeat((4, 4, 1))
        top = self.top.repeat((3, 3, 1))

        # Stretch top layer so its supercell is exactly the same size as the bottom supercell
        cell = top.cell.copy()
        cell[0] = bottom.cell[0]
        cell[1] = bottom.cell[1]
        top.set_cell(cell, scale_atoms=True)

        bottom_height = np.ptp(bottom.positions[:, 2])
        top_height = np.ptp(top.positions[:, 2])

        # shift bottom /top layers such that min/max sits just above/below 10A of vacuum
        bottom.positions[:, 2] += self.vacuum - bottom.positions[:, 2].min()
        top.positions[:, 2] += bottom_height + self.interlayer_dist
        
        bilayer = bottom + top
        cell = bottom.cell.copy()
        cell[2, 2] = 2*self.vacuum + bottom_height + self.interlayer_dist + top_height
        bilayer.set_cell(cell)
        
        self.n_bottom = len(bottom)
        self.n_top = len(top)
        
        return bilayer

    def isolate_bilayer(self, bilayer):
        bottom = bilayer[:self.n_bottom].copy()
        top = bilayer[self.n_bottom:].copy()
        
        return bottom, top


class MoSe2(MetalDiselenide):
    def __init__(self, vacuum=VACUUM):
        super().__init__("Mo", LATTICE_CONSTANT_MOSE2, SE_HEIGHT_MOSE2, vacuum)


class WSe2(MetalDiselenide):
    def __init__(self, vacuum=VACUUM):
        super().__init__("W", LATTICE_CONSTANT_WSE2, SE_HEIGHT_WSE2, vacuum)
