import numpy as np

from config import *


def _vacuum_reference(system, window=VACUUM_REFERENCE_WINDOW, z_reference=Z_REFERENCE):
    z = system.potential.z
    potential = system.potential.averaged * RY_TO_EV

    mask = (z >= window[0]) & (z <= window[1])

    slope, intercept = np.polyfit(z[mask] - z_reference, potential[mask], 1)

    return slope, intercept


def _energy_axes(system):
    _, reference = _vacuum_reference(system)
    
    dos_energy = system.dos.energy - reference

    # ldos energies centred around fermi energy in qe
    ldos_energy = (system.ldos.energies + system.fermi_energy - reference)

    return dos_energy, ldos_energy


def ldos_energy_axis(system, reference):
    if reference == "fermi":
        return np.asarray(system.ldos.energies)
    
    if reference == "vacuum":
        return _energy_axes(system)[1]

    raise ValueError("REFERENCE MUST BE 'fermi' OR 'vacuum'")
