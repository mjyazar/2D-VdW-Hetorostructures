from pathlib import Path
import numpy as np
from ase.io.xsf import read_xsf

from results import * 
from config import INTERFACE_BOX_WIDTH

def read_qe_xsf(path):
    with open(path, "r") as file:
        data, origin, span_vectors, atoms = read_xsf(file, read_data=True)
    
    cell = np.asarray(atoms.cell)
    data = np.asarray(data)
    origin = np.asarray(origin)

    # check endpoints are repeated - default qe xsf output repeats periodic boundary planes
    for axis in range(3):
        if not np.array_equal(np.take(data, 0, axis=axis), np.take(data, -1, axis=axis)):
            raise ValueError(f"NO DUPLICATED PERIODIC ENDPOINT ON AXIS {axis} - {path}")

    # remove (repeated) endpoint
    data = data[:-1, :-1, :-1]

    if not np.allclose(np.asarray(span_vectors), cell, rtol=1e-4, atol=1e-4):
        print(f"XSF SPAN VECTORS DIFFER FROM ATOMIC CELL: {path}")
        
    return Grid3D(data=data, origin=origin, cell=cell, atoms=atoms)


def common_grid(coupled, bottom, top):
    """
    Construct a common grid for the coupled/uncoupledd
    Uncoupled data subtracted from coupled
    """
    for layer in (bottom, top):
        if coupled.data.shape != layer.data.shape:
            raise ValueError("GRID SHAPES DIFFER")

        np.testing.assert_allclose(coupled.cell, layer.cell, atol=1e-6, rtol=0)
        np.testing.assert_allclose(coupled.origin, layer.origin, atol=1e-6, rtol=0)

    data = coupled.data - bottom.data - top.data
    
    return Grid3D(data=data, origin=coupled.origin, cell=coupled.cell, atoms=coupled.atoms)


def fft(data):
    data = np.asarray(data)

    if data.ndim != 3:
        raise ValueError("DATA NOT 3-DIMENSIONAL")

    # normalise by dividing by N=data.size
    return np.fft.fftn(data) / data.size


def reciprocal_cell(cell):
    return 2 * np.pi * np.linalg.inv(cell).T


def interface_mask(z, z_mid, width=INTERFACE_BOX_WIDTH):
    """
    Boolean mask for the interlayer box centred at z_mid
    """
    return (z >= z_mid - width / 2) & (z <= z_mid + width / 2)


def save_induced_charge_fft(coupled_path, bottom_path, top_path, output_path, z_mid):
    coupled = read_qe_xsf(coupled_path)
    bottom = read_qe_xsf(bottom_path)
    top = read_qe_xsf(top_path)

    grid = common_grid(coupled, bottom, top)
    
    box_mask = interface_mask(grid.z, z_mid)
    spectrum = fft(grid.data[:, :, box_mask])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # zip file with multiple arrays
    np.savez_compressed(output_path, coefficients=spectrum, cell=grid.cell, origin=grid.origin, box_z=grid.z[box_mask], z_mid=z_mid)

    return grid, spectrum
