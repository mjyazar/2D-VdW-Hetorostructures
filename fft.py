from pathlib import Path
import numpy as np
from ase.io.xsf import read_xsf

from results import * 

def read_qe_xsf(path):
    with open(path, "r") as file:
        data, origin, span_vectors, atoms = read_xsf(file, read_data=True)
    
    cell = np.asarray(atoms.cell)
    data = np.asarray(data)
    origin = np.asarray(origin)

    # check endpoints are repeated - default qe xsf output repeats periodic boundary planes
    duplicate = False
    for axis in range(3):
        if not np.allclose(np.take(data, 0, axis=axis), np.take(data, -1, axis=axis)):
            duplicate = True
            print(f"NO DUPLICATED PERIODIC ENDPOINTS ON AXIS {axis} - {path}")
    
    if duplicate:
        data = data[:-1, :-1, :-1]
    
    if np.allclose(np.asarray(span_vectors), cell, rtol=1e-4, atol=1e-4):
        duplicate = True
    else:
        print(f"XSF SPAN VECTORS DIFFER FROM ATOMIC CELL: {path}")
        
    # remove (repeated) endpoint
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


def save_induced_charge_fft(coupled_path, bottom_path, top_path, output_path):
    coupled = read_qe_xsf(coupled_path)
    bottom = read_qe_xsf(bottom_path)
    top = read_qe_xsf(top_path)

    grid = common_grid(coupled, bottom, top)
    spectrum = fft(grid.data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # zip file with multiple arrays
    np.savez_compressed(output_path, coefficients=spectrum, cell=grid.cell, origin=grid.origin, shape=np.array(grid.data.shape))

    return grid, spectrum
