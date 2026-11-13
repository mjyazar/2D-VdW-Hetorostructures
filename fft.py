import numpy as np
from ase.io.xsf import read_xsf
from scipy.ndimage import map_coordinates

from results import * 
from config import *
from alignment import *


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


def _commond_grid(coupled, bottom, top):
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
    """
    A * B.T = 2pi * I --> B = 2pi * (A^-1).T
    """
    return 2 * np.pi * np.linalg.inv(np.asarray(cell)).T


def interface_mask(z, z_mid, width=INTERFACE_BOX_WIDTH):
    """
    Boolean mask for the interlayer box centred at z_mid
    """
    return (z >= z_mid - width / 2) & (z <= z_mid + width / 2)


def save_induced_charge_fft(coupled_path, bottom_path, top_path, output_path, z_mid):
    coupled = read_qe_xsf(coupled_path)
    bottom = read_qe_xsf(bottom_path)
    top = read_qe_xsf(top_path)

    grid = _commond_grid(coupled, bottom, top)
    
    box_mask = interface_mask(grid.z, z_mid)
    spectrum = fft(grid.data[:, :, box_mask])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # zip file with multiple arrays
    np.savez_compressed(output_path, coefficients=spectrum, cell=grid.cell, origin=grid.origin, box_z=grid.z[box_mask], z_mid=z_mid)

    return grid, spectrum


def g_vectors(cell, shape):
    """
    Physical wavevector of every coefficient of the box FFT.
    q_par = m1*b1 + m2*b2 and q_z = 2*pi*m3 / (n3*dz), with m = 0, 1, ..., -2, -1 (numpy FFT ordering).
    Uses the cell vectors, so works for any primitive cell, supercell or twisted cell.
    """
    n1, n2 = shape
    b = reciprocal_cell(cell)

    m1 = np.fft.fftfreq(n1, d=1/n1)  # integer indices in FFT order
    m2 = np.fft.fftfreq(n2, d=1/n2)
    M1, M2 = np.meshgrid(m1, m2, indexing="ij")

    # G = m1*b1 + m2*b2
    G_inplane = M1[:, :, None] * b[0, :2] + M2[:, :, None] * b[1, :2]
    
    return G_inplane
    # dz = box_z[1] - box_z[0]
    # qz = 2 * np.pi * np.fft.fftfreq(n3, d=dz)  # box length n3*dz, not INTERFACE_BOX_WIDTH
    
    return G_inplane, qz


def fourier_peaks(F, G, threshold=FFT_PEAK_THRESHOLD):
    """
    Extract G Fourier peaks
    """
    F_intensity = np.abs(F) ** 2
    G_norm = np.linalg.norm(G, axis=-1)
    
    mask = G_norm > 0
    relative = F_intensity / F_intensity[mask].max()
    peaks = mask & (relative >= threshold)

    peak_indices = np.argwhere(peaks)
    # sort by intensity first (descending hence the minus) then magnitude of G peaks
    # lexsort does not reorder your actual data directly
    order = np.lexsort((G_norm[peaks], -F_intensity[peaks]))
    ordered_peaks = peak_indices[order]
    i1, i2 = ordered_peaks.T

    return {"peaks": ordered_peaks,
            "G": G[i1, i2],
            "norm": G_norm[i1, i2],
            "wavelength": 2 * np.pi / G_norm[i1, i2],
            "intensity": F_intensity[i1, i2],
            "relative": relative[i1, i2],
            "phase": np.degrees(np.angle(F[i1, i2]))}


def on_lattice(G, layer_cell):
    """
    Returns true where G is a reciprocal lattice vector of a layer
    i.e. evaluated n_i for a_i is and integer
    """
    a = np.asarray(layer_cell)[:2, :2]  # discards a3 of the cell
    n = G @ a.T / (2 * np.pi)
    
    return np.all(np.abs(n - np.round(n)) < FFT_LATTICE_TOL, axis=-1)


def g_origin(G, bottom_cell, top_cell):
    """
    Split G into four classes categorising which layer's reciprocal lattice they belong to
    """
    bottom = on_lattice(G, bottom_cell)
    top = on_lattice(G, top_cell)
    
    return {"both": bottom & top,
            "bottom": bottom & ~top,  # exclude 
            "top": top & ~bottom,
            "neither": ~bottom & ~top}


def sample_columns(data, positions):
    """
    Periodic grid data (n1, n2, nz) along vertical lines through in-plane points frac (n, 2) in cell fractions
    Linear interpolation in-plane, grid planes in z. Returns (n, nz)
    """
    n1, n2, nz = data.shape
    positions = np.asarray(positions, dtype=float)
    
    # Create an array holding the coordinates of every sampling location
    # (3, N_columns, Nz)
    coords = np.empty((3, len(positions), nz))
    coords[0] = positions[:, 0, None] * n1  # i
    coords[1] = positions[:, 1, None] * n2  # j
    coords[2] = np.arange(nz)               # k

    # broadcast output shape (N_columns, Nz)
    return map_coordinates(data, coords, order=1, mode="grid-wrap")


def ripple_groups(G, classes, tol=FFT_G_TOL):
    """
    Group G by (|G|, lattice class): ripples with the same wavelength owned by the same layer(s)
    Returns a list of (|G|, class, number of G) per group and the group index of every FFT seat (n1, n2)
    """
    names = list(classes)
    code = np.zeros(G.shape[:2], dtype=int)
    for i, mask in enumerate(classes.values()):
        code[mask] = i

    # shell number: sort |G| and start a new shell wherever |G| jumps by more than tol
    norm = np.linalg.norm(G, axis=-1).ravel()
    order = np.argsort(norm)
    shell = np.empty(norm.size, dtype=int)
    shell[order] = np.cumsum(np.diff(norm[order], prepend=norm[order][0]) > tol)

    keys, index, counts = np.unique(np.stack([shell, code.ravel()], axis=1), axis=0,
                                    return_inverse=True, return_counts=True)
    groups = [(norm[shell == s].mean(), names[c], int(n)) for (s, c), n in zip(keys, counts)]

    return groups, index.reshape(code.shape)

