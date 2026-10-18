import numpy as np

from config import *


def _vacuum_reference(system, window=VACUUM_REFERENCE_WINDOW, z_reference=Z_REFERENCE):
    z = system.potential.z
    potential = system.potential.averaged * RY_TO_EV

    mask = (z >= window[0]) & (z <= window[1])

    slope, intercept = np.polyfit(z[mask] - z_reference, potential[mask], 1)

    return slope, intercept


def energy_axes(system):
    _, reference = _vacuum_reference(system)
    
    dos_energy = system.dos.energy - reference

    # ldos energies centred around fermi energy in qe
    ldos_energy = (system.ldos.energies + system.fermi_energy - reference)

    return dos_energy, ldos_energy
