import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm
import numpy as np
from ase.spectrum.band_structure import get_band_structure, BandStructure
from ase.data import covalent_radii

from results import * 
from config import *
from alignment import *
import fft as fft


def _induced_charge(results: Results):
    """
    Delta n = coupled - bottom - top on the full 3D grid
    """
    coupled, bottom, top = results.coupled.charge_density, results.bottom.charge_density, results.top.charge_density
    return (coupled.data - bottom.data - top.data) / BOHR_TO_ANGSTROM**3


def atom_columns(bottom, top):
    """
    In-plane atom positions (cell fractions) -> which layer(s) have an atom there, e.g. 'bottom+top' for a dimer
    """
    columns = {}  # keys=positions, values=layer names
    for layer, atoms in (("bottom", bottom), ("top", top)):
        for frac in np.round(atoms.get_scaled_positions()[:, :2], 3) % 1:
            columns.setdefault(tuple(frac), []).append(layer)
    
    return {frac: "+".join(dict.fromkeys(layers)) for frac, layers in columns.items()}


def _atoms_on_line(atoms, p0, p1):
    """
    (s, z) of every atom, or periodic image, within COLUMN_TOL of the in-plane line p0 -> p1 (cell fractions)
    """
    cell = atoms.cell.array[:2, :2]
    start, u = p0 @ cell, (p1 - p0) @ cell  # convert (fractions) to real space

    hits = []
    for position in atoms.positions:
        for n1, n2 in np.ndindex(7, 7):
            xy = position[:2] + (n1 - 3) * cell[0] + (n2 - 3) * cell[1]
            t = (xy - start) @ u / (u @ u)
            if -1e-6 <= t <= 1 + 1e-6 and np.linalg.norm(xy - start - t * u) < COLUMN_TOL:
                hits.append((t * np.sqrt(u @ u), position[2]))
    return hits


def plot_band_structure(bandpath, energies, name):
    """
    Plot and save band structure
    """
    print("PLOTTING BAND STRUCTURE")
    
    band_structure = BandStructure(path=bandpath, energies=energies, reference=0.0)  # reference is now zero after shifting
    
    fig, ax = plt.subplots()

    ax = band_structure.plot()
    assert ax is not None  # ensures BandStructure doesn't return None to prevent an error
    ax.set_ylim(-10, 10)
    ax.set_ylabel("Energy - $E_F$ (eV)")
    ax.set_title(f"Graphene {name.capitalize()} Band Structure")

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name.capitalize()} Graphene Band Structure.png", dpi=300)
    plt.close(fig)


def _get_vacuum_level(V, z):
        
    assert V.shape == z.shape
    
    # old implementation
    # atomic_positions = structure.get_positions()[:, 2]
    # z_bottom = atomic_positions.min()
    # z_top = atomic_positions.max()
    # vacuum_mask = (z < z_bottom) | (z > z_top)
    # positions = V[vacuum_mask]
    
    dV_dz = np.gradient(V, z)
    tolerance = VACUUM_LEVEL_TOLERANCE
    vacuum_level = V[np.abs(dV_dz) < tolerance]

    # if tolerance is too low prevent code crash by increasing magnitude
    while vacuum_level.size == 0:
        tolerance *= 10
        vacuum_level = V[np.abs(dV_dz) < tolerance]
    
    vacuum_level = np.min(vacuum_level)
    
    return vacuum_level


def potential_subtracted(results: Results, field, vdw_scheme):
    print("PLOTTING SUBTRACTED POTENTIAL")
    
    coupled = results.coupled.potential
    bottom = results.bottom.potential
    top = results.top.potential

    np.testing.assert_allclose(coupled.z, bottom.z)
    np.testing.assert_allclose(coupled.z, top.z)
    
    z = coupled.z
    
    potential = (coupled.averaged - bottom.averaged - top.averaged) * RY_TO_EV

    # coordinates = coupled.potential_averaged.coordinates
    # potential = coupled.potential_averaged.planar - bottom.potential_averaged.planar - top.potential_averaged.planar
    
    fig, ax = plt.subplots()
    
    ax.plot(z, potential, color='black', label=r"$U_{coupled layers} - U_{bottom layer} - U_{top layer}$")
    
    ax.set_xlabel(r"z ($\AA$)")
    ax.set_ylabel("Potential Energy (eV)")
    ax.set_title(f"Subtracted Potential, E-field={field}au, Correction: {vdw_scheme}")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(POTENTIAL_FIG_DIR / f"Subtracted_Potential_{field}au_{vdw_scheme}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    

def potential_individual(results: Results, field, vdw_scheme):
    print("PLOTTING INDIVIDUAL POTENTIALS")
    
    # iterate through coupled, top, and bottom
    for system in results.__dict__.values():
        system:System

        z = system.potential.z
        potential = system.potential.averaged * RY_TO_EV
        
        print(f"PLOTTING {system.name} POTENTIAL")

        fig, ax = plt.subplots()

        ax.plot(z, potential, linewidth=0.75, color='black')
        ax.set_title(f"{system.name.capitalize()} Layer Potential V(z), E-field={field}au, Correction: {vdw_scheme}")
        ax.set_xlabel(r"z ($\AA$)")
        ax.set_ylabel("Potential Energy (eV)")
        
        if field == 0:
            vacuum_level = _get_vacuum_level(potential, z)
            ax.axhline(vacuum_level, color="red", linestyle="--", linewidth=0.8, label="Vacuum Level")

        plt.tight_layout()
        path = POTENTIAL_FIG_DIR / "Individual"
        path.mkdir(exist_ok=True)
        plt.savefig(path / f"Potential_{system.name.capitalize()}_Layer_{field}au_{vdw_scheme}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def charge_density_subtracted(results: Results, field, vdw_scheme):
    print("PLOTTING SUBTRACTED (AVERAGED) CHARGE DENSITY")

    coupled = results.coupled.charge_density
    bottom = results.bottom.charge_density
    top = results.top.charge_density
    
    z = coupled.z
    
    charge_density = (coupled.averaged - bottom.averaged - top.averaged) / (BOHR_TO_ANGSTROM)**3
    
    # print("\nc09 Coupled Atoms Positions: \n", c09.coupled.atoms.positions)
    positions_z = coupled.atoms.positions[:, 2]
    
    bottom_z = positions_z.min()
    top_z = positions_z.max()
    
    fig, ax = plt.subplots(figsize=FIGSIZE_CHARGE_DENSITY)
    
    ax.plot(charge_density, z, color="black")
    ax.axhline(bottom_z, color="blue", linestyle=":", linewidth=0.8, label="Bottom Layer")
    ax.axhline(top_z, color="red", linestyle=":", linewidth=0.8, label="Top Layer")
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)

    ax.set_ylim(bottom_z - 5, top_z + 5)
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    
    ax.tick_params(axis="both", direction="in")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    ax.set_xlabel(r"Charge Density ($e/{\AA}^3)$")  # averaged but not multiplied by area
    ax.set_ylabel(r"z ($\AA$)")
    ax.set_title(f"Subtracted Charge Density, E-field={field}au, Correction: {vdw_scheme}", pad=15)

    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARGE_DENSITY_FIG_DIR / f"Subtracted_Charge_Density_{field}au_{vdw_scheme}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_vdw_charge_difference(results, field, name):
    """
    Plot the plane averaged charge density difference between dn_c09 and dn_pbe, 
    where each is the subtracted charge density i.e. coupled - top - bottom
    """
    print(f"PLOTTING VDW Charge Density Difference")
    pbe: Results = results["d3"]
    c09: Results = results["c09"]
    
    np.testing.assert_allclose(pbe.coupled.charge_density.z, pbe.bottom.charge_density.z)
    np.testing.assert_allclose(pbe.coupled.charge_density.z, pbe.top.charge_density.z)
    np.testing.assert_allclose(c09.coupled.charge_density.z, c09.bottom.charge_density.z)
    np.testing.assert_allclose(c09.coupled.charge_density.z, c09.top.charge_density.z)
    np.testing.assert_allclose(pbe.coupled.charge_density.z, c09.coupled.charge_density.z)
    
    z = pbe.coupled.charge_density.z
    
    dn_pbe = pbe.coupled.charge_density.averaged - pbe.bottom.charge_density.averaged - pbe.top.charge_density.averaged
    dn_c09 = c09.coupled.charge_density.averaged - c09.bottom.charge_density.averaged - c09.top.charge_density.averaged
    
    cell = c09.coupled.atoms.cell.array
    area = np.linalg.norm(np.cross(cell[0], cell[1]))  # cross product of a1 and a2 (i.e. in-plane vecors)
    
    dn_vdw = ((dn_c09 - dn_pbe) / BOHR_TO_ANGSTROM**3) * area

    fig, ax = plt.subplots(figsize=FIGSIZE_CHARGE_DENSITY)
    
    ax.plot(dn_vdw, z, color="black")

    # print("\nc09 Coupled Atoms Positions: \n", c09.coupled.atoms.positions)
    positions_z = c09.coupled.atoms.positions[:, 2]
    
    bottom_z = positions_z.min()
    top_z = positions_z.max()
    
    ax.axhline(bottom_z, color="blue", linestyle=":", linewidth=0.8, label="Bottom Layer")
    ax.axhline(top_z, color="red", linestyle=":", linewidth=0.8, label="Top Layer")
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)

    ax.set_ylim(bottom_z - 5, top_z + 5)
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    
    ax.tick_params(axis="both", direction="in")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    ax.set_xlabel(r"$\Delta n_{vdW}(z)$ ($e/{\AA})$")
    ax.set_ylabel(r"z ($\AA$)")
    ax.set_title(rf"vdW Charge Density $\Delta n_{{c09}}(z)-\Delta n_{{PBE + d3}}(z)$, E-field={field}au", pad=15)

    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARGE_DENSITY_FIG_DIR / f"vdW_Charge_Density_{name}_{field}au.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def dos_subtracted(results: Results, field, vdw_scheme, delta_e=DELTA_E_DOS):
    print(f"PLOTTING FERMI-ALIGNED SUBTRACTED DOS")

    coupled = results.coupled
    bottom = results.bottom
    top = results.top
    
    energy_coupled = coupled.dos.energy - coupled.fermi_energy
    energy_bottom = bottom.dos.energy - bottom.fermi_energy
    energy_top = top.dos.energy - top.fermi_energy
    
    dos_coupled = coupled.dos.dos
    dos_bottom = bottom.dos.dos
    dos_top = top.dos.dos
    
    min_energy = max(energy_coupled.min(), energy_bottom.min(), energy_top.min())
    max_energy = min(energy_coupled.max(), energy_top.max(), energy_bottom.max())
    
    energy_grid = np.arange(min_energy, max_energy, delta_e)
    
    dos_coupled_grid = np.interp(energy_grid, energy_coupled, dos_coupled)
    dos_bottom_grid = np.interp(energy_grid, energy_bottom, dos_bottom)
    dos_top_grid = np.interp(energy_grid, energy_top, dos_top)
    
    dos_subtracted = dos_coupled_grid - dos_bottom_grid - dos_top_grid
    
    fig, ax = plt.subplots()

    # ax.plot(grid, dos_coupled_grid, label="$DOS_{coupled}$")
    ax.plot(energy_grid, dos_subtracted, color="black", label=r"$DOS_{coupled} - DOS_{top} - DOS_{bottom}$")

    ax.set_title(f"Fermi-Aligned DOS Subtracted, E-field={field}au, Correction: {vdw_scheme}")
    ax.set_xlabel(r"$Energy$ (eV)")
    ax.set_ylabel("DOS (states/eV/cell)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(DOS_FIG_DIR / f"DOS_Subtracted_{field}au_{vdw_scheme}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def dos_individual(results: Results, field, vdw_scheme, delta_e=LDOS_GRID_DELTA_E):
    print(f"PLOTTING FERMI-ALIGNED INDIVIDUAL DOS")

    coupled = results.coupled
    bottom = results.bottom
    top = results.top
    
    energy_coupled = coupled.dos.energy - coupled.fermi_energy
    energy_bottom = bottom.dos.energy - bottom.fermi_energy
    energy_top = top.dos.energy - top.fermi_energy
    
    dos_coupled = coupled.dos.dos
    dos_bottom = bottom.dos.dos
    dos_top = top.dos.dos
    
    min_energy = max(energy_coupled.min(), energy_bottom.min(), energy_top.min())
    max_energy = min(energy_coupled.max(), energy_top.max(), energy_bottom.max())
    
    energy_grid = np.arange(min_energy, max_energy, delta_e)

    dos_coupled_grid = np.interp(energy_grid, energy_coupled, dos_coupled)
    dos_bottom_grid = np.interp(energy_grid, energy_bottom, dos_bottom)
    dos_top_grid = np.interp(energy_grid, energy_top, dos_top)
    
    plots = [(dos_coupled_grid, "Coupled Layers", "black", "-"), (dos_bottom_grid, "Bottom Layer", "red", "--"), (dos_top_grid, "Top Layer", "green", ":")]
    
    fig, ax = plt.subplots()

    for plot, label, color, linestyle in plots:
        ax.plot(energy_grid, plot, color=color, label=f"{label}", linestyle=linestyle)

    ax.set_title(f"Fermi-Aligned DOS Subtracted, E-field={field}au, Correction: {vdw_scheme}")
    ax.set_xlabel(r"$Energy$ (eV)")
    ax.set_ylabel("DOS (states/eV/cell)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(DOS_FIG_DIR / f"DOS_Individual_{field}au_{vdw_scheme}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def ldos_individual(results: Results, field, vdw_scheme):
    print(f"PLOTTING FERMI-ALIGNED INDIDIVDUAL LDOS")
        
    coupled = results.coupled
    bottom = results.bottom
    top = results.top
    
    np.testing.assert_allclose(coupled.ldos.z, bottom.ldos.z)
    np.testing.assert_allclose(coupled.ldos.z, top.ldos.z)
    
    z = coupled.ldos.z
    
    plots = [(coupled, "coupled"), (bottom, "bottom"), (top, "top")]

    for layer, label in plots:
        fig, ax = plt.subplots()
        
        E = layer.ldos.energies
        V = coupled.potential.averaged * RY_TO_EV - coupled.fermi_energy

        limit = np.max(np.abs(layer.ldos.averaged))
        symmetric_norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        mesh = ax.pcolormesh(z, E, layer.ldos.averaged, cmap="seismic", shading="auto", norm=symmetric_norm)
        
        fig.colorbar(mesh, ax=ax, label=r"$\Delta LDOS$")

        ax.plot(layer.potential.z, V, linewidth=0.75, color='black')
        ax.set_title(f"Fermi-Aligned LDOS, E-field={field}au, Correction: {vdw_scheme}")
        ax.set_xlabel(r"$z$ ($\AA$)")
        ax.set_ylabel(r"$E-E_F$ (eV)")
        
        plt.tight_layout()
        path = LDOS_FIG_DIR / "Individual"
        path.mkdir(exist_ok=True)
        plt.savefig(path / f"LDOS_{label}_{field}au_{vdw_scheme}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def fermi_ldos_subtracted(results: Results, field, vdw_scheme):
    print(f"PLOTTING FERMI-ALIGNED SUBTRACTED LDOS")

    coupled = results.coupled.ldos
    bottom = results.bottom.ldos
    top = results.top.ldos
    
    z = coupled.z
    
    np.testing.assert_allclose(coupled.z, bottom.z)
    np.testing.assert_allclose(coupled.z, top.z)
    
    E = coupled.energies
    mask = (E >= PLOT_WINDOW_LDOS[0]) & (E <= PLOT_WINDOW_LDOS[1])
    E = E[mask]
    
    # averaged is a 2D array with shape (n_energies, n_z)
    ldos_subtracted = (coupled.averaged - bottom.averaged - top.averaged)[mask, :]
    
    fig, ax = plt.subplots()
    
    limit = np.max(np.abs(ldos_subtracted))
    symmetric_norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    mesh = ax.pcolormesh(z, E, ldos_subtracted * LDOS_WEIGHT, cmap="seismic", shading="auto", norm=symmetric_norm)
    
    fig.colorbar(mesh, ax=ax, label=r"$\Delta LDOS$")
    
    ax.set_title(f"Fermi-Aligned Subtracted LDOS, E-field={field}au, Correction: {vdw_scheme}")
    ax.set_xlabel(r"$z$ ($\AA$)")
    ax.set_ylabel(r"$E-E_F$ (eV)")
    
    #ax.set_ylim(*PLOT_WINDOW_LDOS)
    
    plt.tight_layout()
    plt.savefig(LDOS_FIG_DIR / f"LDOS_Subtracted_{field}au_{vdw_scheme}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def vacuum_ldos_subtracted(results: Results, field, vdw_scheme):
    print("PLOTTING VACUUM-ALIGNED SUBTRACTED LDOS")

    coupled = results.coupled.ldos
    bottom = results.bottom.ldos
    top = results.top.ldos

    z = coupled.z

    np.testing.assert_allclose(z, bottom.z)
    np.testing.assert_allclose(z, top.z)

    # Each system gets its own vacuum-referenced energy axis.
    E_coupled = ldos_energy_axis(results.coupled, "vacuum")
    E_bottom = ldos_energy_axis(results.bottom, "vacuum")
    E_top = ldos_energy_axis(results.top, "vacuum")

    minimum = max(E_coupled.min(), E_bottom.min(), E_top.min(), PLOT_WINDOW_LDOS[0])
    maximum = min(E_coupled.max(), E_bottom.max(), E_top.max(), PLOT_WINDOW_LDOS[1])

    E = np.arange(minimum, maximum, LDOS_GRID_DELTA_E)

    shape = (len(E), len(z))

    coupled_grid = np.zeros(shape)
    bottom_grid = np.zeros(shape)
    top_grid = np.zeros(shape)

    for j in range(len(z)):
        coupled_grid[:, j] = np.interp(E, E_coupled, coupled.averaged[:, j])
        bottom_grid[:, j] = np.interp(E, E_bottom, bottom.averaged[:, j])
        top_grid[:, j] = np.interp(E, E_top, top.averaged[:, j])

    ldos_subtracted = coupled_grid - bottom_grid - top_grid

    fig, ax = plt.subplots()

    limit = np.max(np.abs(ldos_subtracted))
    symmetric_norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    mesh = ax.pcolormesh(z, E, ldos_subtracted * LDOS_WEIGHT, cmap="seismic", shading="auto", norm=symmetric_norm)

    fig.colorbar(mesh, ax=ax, label=r"$\Delta LDOS$")

    ax.set_title(f"Vacuum-Aligned Subtracted LDOS, E-field={field}au, Correction: {vdw_scheme}")

    ax.set_xlabel(r"$z$ ($\AA$)")
    ax.set_ylabel(r"E-V_ref (eV)")

    plt.tight_layout()
    plt.savefig(LDOS_FIG_DIR / f"LDOS_Subtracted_Vacuum_{field}au_{vdw_scheme}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def ldos_subtracted(results: Results, field, vdw_scheme):
    print("PLOTTING FERMI- AND VACUUM-ALIGNED SUBTRACTED LDOS")

    coupled = results.coupled.ldos
    bottom = results.bottom.ldos
    top = results.top.ldos

    z = coupled.z

    np.testing.assert_allclose(coupled.z, bottom.z)
    np.testing.assert_allclose(coupled.z, top.z)

    # Fermi-alignment
    E_fermi = coupled.energies
    mask = (E_fermi >= PLOT_WINDOW_LDOS[0]) & (E_fermi <= PLOT_WINDOW_LDOS[1])
    E_fermi = E_fermi[mask]

    fermi_subtracted = (coupled.averaged - bottom.averaged - top.averaged)[mask, :]

    # Vacuum-alignment
    E_coupled = ldos_energy_axis(results.coupled, "vacuum")
    E_bottom = ldos_energy_axis(results.bottom, "vacuum")
    E_top = ldos_energy_axis(results.top, "vacuum")

    minimum = max(E_coupled.min(), E_bottom.min(), E_top.min(), PLOT_WINDOW_LDOS[0])
    maximum = min(E_coupled.max(), E_bottom.max(), E_top.max(), PLOT_WINDOW_LDOS[1])

    E = np.arange(minimum, maximum, LDOS_GRID_DELTA_E)

    shape = (len(E), len(z))

    coupled_grid = np.zeros(shape)
    bottom_grid = np.zeros(shape)
    top_grid = np.zeros(shape)

    for i in range(len(z)):
        coupled_grid[:, i] = np.interp(E, E_coupled, coupled.averaged[:, i])
        bottom_grid[:, i] = np.interp(E, E_bottom, bottom.averaged[:, i])
        top_grid[:, i] = np.interp(E, E_top, top.averaged[:, i])

    vacuum_subtracted = coupled_grid - bottom_grid - top_grid

    limit = max(np.max(np.abs(fermi_subtracted)), np.max(np.abs(vacuum_subtracted)))
    symmetric_norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    fig, (ax_fermi, ax_vacuum) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

    mesh = ax_fermi.pcolormesh(z, E_fermi, fermi_subtracted * LDOS_WEIGHT, cmap="seismic", shading="auto", norm=symmetric_norm)
    ax_fermi.set_title("Fermi-Aligned")
    ax_fermi.set_xlabel(r"$z$ ($\AA$)")
    ax_fermi.set_ylabel(r"$E-E_F$ (eV)")

    mesh = ax_vacuum.pcolormesh(z, E, vacuum_subtracted * LDOS_WEIGHT, cmap="seismic", shading="auto", norm=symmetric_norm)
    ax_vacuum.set_title("Vacuum-Aligned")
    ax_vacuum.set_xlabel(r"$z$ ($\AA$)")
    ax_vacuum.set_ylabel(r"$E-V_{ref}$ (eV)")

    fig.colorbar(mesh, ax=[ax_fermi, ax_vacuum], label=r"$\Delta LDOS$")
    fig.suptitle(f"Subtracted LDOS, E-field={field}au, Correction: {vdw_scheme}")

    plt.savefig(LDOS_FIG_DIR / f"LDOS_Subtracted_Comparison_{field}au_{vdw_scheme}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fft_plot(spectrum, data_path):
    print(f"PLOTTING FFT")
    
    data = np.load(data_path)
    # box_z = data["box_z"]
    F = data["coefficients"][:, :, 0] / BOHR_TO_ANGSTROM**3  # (n1, n2) complex
    G = fft.g_vectors(data["cell"], F.shape) # (n1, n2, 2)
    
    Gx = G[:, :, 0]
    Gy = G[:, :, 1]
    
    gap_map = np.fft.ifft2(F).real * F.size  # inverse fft and unnormalise

    U = gap_map.mean()  # average charge in gap
    sigma = gap_map.std()  # 
    
    fourier_peaks_data = fft.fourier_peaks(F, G)
    
    peaks = fourier_peaks_data["peaks"]
    
    print("Coefficients shape: ", data["coefficients"].shape)
    print("F shape: ", F.shape)
    print("Origin: ", data["origin"][:2])
    print("Gap Map at Dimer: ", gap_map[20, 10])
    
    plt.scatter(Gx, Gy, c=peaks, cmap="inferno", s=20)
    plt.colorbar(label="Intensity")
    plt.xlabel(r"$G_x$ ($\AA^{-1}$)")
    plt.ylabel(r"$G_y$ ($\AA^{-1}$)")
    plt.axis("equal")
    plt.show()
    
    # x, y, z = np.indices(spectrum.shape)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")

    # sc = ax.scatter(
    #     x.ravel(), y.ravel(), z.ravel(),
    #     c=spectrum.ravel(),
    #     cmap="viridis",
    #     s=10,
    #     alpha=0.5)
    
    # plt.tight_layout()
    # plt.show()


def _interface(results: Results):
    """
    Return top of the bottom layer, bottom of the top layer (z positions)
    """
    return results.bottom.atoms.positions[:, 2].max(), results.top.atoms.positions[:, 2].min()


def slice_plot(data, results: Results, system, title, path, label=r"$\Delta n$ ($e/{\AA}^3$)"):
    """
    Vertical cut of 3D grid data along system.slice_line (cell fractions), around the interface
    """
    print(f"PLOTTING {path.stem}")

    p0, p1 = (np.asarray(p, dtype=float) for p in system.slice_line)
    t = np.linspace(0, 1, SLICE_POINTS)
    values = fft.sample_columns(data, p0 + t[:, None] * (p1 - p0))  # (SLICE_POINTS, nz)
    s = t * np.linalg.norm((p1 - p0) @ results.coupled.atoms.cell.array[:2, :2])

    z = results.coupled.charge_density.z
    bottom_z, top_z = _interface(results)
    window = (z >= bottom_z - 1.5) & (z <= top_z + 1.5)
    limit = np.percentile(np.abs(values[:, window]), 99.5)  # ignore the largest values at the atoms

    fig, ax = plt.subplots(figsize=(5, 4))
    
    symmetric_norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    mesh = ax.pcolormesh(s, z[window], values[:, window].T, cmap="seismic", shading="gouraud",
                         norm=symmetric_norm)
    for s_atom, z_atom in _atoms_on_line(results.coupled.atoms, p0, p1):
        ax.plot(s_atom, z_atom, "o", color="black", markersize=4)

    fig.colorbar(mesh, ax=ax, label=label)
    ax.set_xlabel(r"Distance along slice ($\AA$)")
    ax.set_ylabel(r"z ($\AA$)")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)




def charge_columns(results: Results, field, vdw_scheme):
    """
    Left: Delta n(z) above every atomic column against the planar average.
    Right: lateral standard deviation sigma(z) of each plane, i.e. how much the planar average hides at each height
    """
    print("PLOTTING CHARGE DENSITY COLUMNS")

    dn = _induced_charge(results)
    z = results.coupled.charge_density.z
    bottom_z, top_z = _interface(results)
    window = (z >= bottom_z - 1) & (z <= top_z + 1)

    columns = atom_columns(results.bottom.atoms, results.top.atoms)
    values = fft.sample_columns(dn, list(columns))  # (n_columns, nz)

    fig, (ax, ax_sigma) = plt.subplots(1, 2, sharey=True, figsize=(8, 5))

    ax.plot(dn.mean(axis=(0, 1))[window], z[window], color="black", linewidth=2, label="Planar average")
    for (frac, layers), profile in zip(columns.items(), values):
        ax.plot(profile[window], z[window], label=f"({frac[0]:.2f}, {frac[1]:.2f}): {layers}")
    ax_sigma.plot(dn.std(axis=(0, 1))[window], z[window], color="black")

    for a in (ax, ax_sigma):
        a.axhline(bottom_z, color="blue", linestyle=":", linewidth=0.8)
        a.axhline(top_z, color="red", linestyle=":", linewidth=0.8)
        a.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)

    ax.set_xlabel(r"$\Delta n$ ($e/{\AA}^3$)")
    ax_sigma.set_xlabel(r"$\sigma(z)$ ($e/{\AA}^3$)")
    ax.set_ylabel(r"z ($\AA$)")
    ax.legend(fontsize=7)
    fig.suptitle(f"Charge Density Columns, E-field={field}au, Correction: {vdw_scheme}")

    plt.tight_layout()
    plt.savefig(CHARGE_DENSITY_FIG_DIR / f"Charge_Columns_{field}au_{vdw_scheme}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def charge_fourier(results: Results, field, vdw_scheme):
    """
    Fourier view of the induced charge: F(G, z) = 2D FFT of every plane - F(0, z) is the planar average
    Left: spectrum on the mid-gap plane, |F(G)|^2 against |G| for every G != 0, coloured by whose lattice G is on
    Right: ripple power of each lattice class against z; the classes add up to sigma(z)^2 of charge_columns (Parseval)
    """
    print("PLOTTING CHARGE FOURIER")

    dn = _induced_charge(results)
    n1, n2, _ = dn.shape
    power = np.abs(np.fft.fft2(dn, axes=(0, 1)) / (n1 * n2)) ** 2  # (n1, n2, nz)

    G = fft.g_vectors(results.coupled.atoms.cell.array, (n1, n2))
    norm = np.linalg.norm(G, axis=-1)
    classes = fft.g_origin(G, results.bottom.atoms.cell.array, results.top.atoms.cell.array)

    z = results.coupled.charge_density.z
    bottom_z, top_z = _interface(results)
    window = (z >= bottom_z - 1) & (z <= top_z + 1)
    k = np.argmin(np.abs(z - (bottom_z + top_z) / 2))  # mid-gap plane

    fig, (ax_spectrum, ax_z) = plt.subplots(1, 2, figsize=(10, 5))

    for name, mask in classes.items():
        mask = mask & (norm > 0)
        if mask.any():
            ax_spectrum.scatter(norm[mask], power[mask][:, k], s=12, label=name)
            ax_z.plot(power[mask].sum(axis=0)[window], z[window], label=name)
    
    largest = power[norm > 0][:, k].max()
    ax_spectrum.axhline(FFT_PEAK_THRESHOLD * largest, color="grey", linestyle=":", linewidth=0.8, label="peak threshold")
    ax_spectrum.set_yscale("log")
    ax_spectrum.set_ylim(largest * 1e-8, largest * 3)
    ax_spectrum.set_xlim(0, 1.05 * norm[power[:, :, k] > largest * 1e-8].max())
    ax_spectrum.set_xlabel(r"$|G|$ ($\AA^{-1}$)")
    ax_spectrum.set_ylabel(r"$|F(G)|^2$ at mid-gap ($e^2/{\AA}^6$)")
    ax_spectrum.legend()

    ax_z.axhline(bottom_z, color="blue", linestyle=":", linewidth=0.8)
    ax_z.axhline(top_z, color="red", linestyle=":", linewidth=0.8)
    ax_z.set_xscale("log")
    ax_z.set_xlabel(r"ripple power $\sum_{G \neq 0} |F(G, z)|^2$ ($e^2/{\AA}^6$)")
    ax_z.set_ylabel(r"z ($\AA$)")
    ax_z.legend()

    fig.suptitle(f"Charge Fourier, E-field={field}au, Correction: {vdw_scheme}")
    plt.tight_layout()
    plt.savefig(FFT_FIG_DIR / f"Charge_Fourier_{field}au_{vdw_scheme}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def ldos_columns(results: Results, field, vdw_scheme):
    """
    Fermi-aligned Delta LDOS(E) on the mid-gap plane above every atomic column, and its planar average
    """
    print("PLOTTING LDOS COLUMNS")

    coupled, bottom, top = results.coupled.ldos, results.bottom.ldos, results.top.ldos
    k = np.argmin(np.abs(coupled.box_z - np.mean(_interface(results))))  # mid-gap plane of the box

    # (n1, n2, n_energies): energy on the last axis, so sample_columns reads one Delta LDOS(E) per column
    plane = (coupled.box[..., k] - bottom.box[..., k] - top.box[..., k]).transpose(1, 2, 0) / (BOHR_TO_ANGSTROM**3 * RY_TO_EV)

    E = coupled.energies
    mask = (E >= PLOT_WINDOW_LDOS[0]) & (E <= PLOT_WINDOW_LDOS[1])
    columns = atom_columns(results.bottom.atoms, results.top.atoms)
    values = fft.sample_columns(plane, list(columns))  # (n_columns, n_energies)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(E[mask], plane.mean(axis=(0, 1))[mask], color="black", linewidth=2, label="Planar average")
    for (frac, layers), profile in zip(columns.items(), values):
        ax.plot(E[mask], profile[mask], label=f"({frac[0]:.2f}, {frac[1]:.2f}): {layers}")

    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$E-E_F$ (eV)")
    ax.set_ylabel(r"mid-gap $\Delta LDOS$ (states/${\AA}^3$/eV)")
    ax.set_title(f"Mid-Gap LDOS Columns, E-field={field}au, Correction: {vdw_scheme}")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(LDOS_FIG_DIR / f"LDOS_Columns_{field}au_{vdw_scheme}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def ldos_fourier(results: Results, field, vdw_scheme):
    """
    Fourier view of the mid-gap Delta LDOS, energy by energy (FT-LDOS): power of the strongest ripple groups
    (same |G|, same lattice class) against energy, with the planar average |F(0, E)|^2 for comparison
    """
    print("PLOTTING LDOS FOURIER")

    coupled, bottom, top = results.coupled.ldos, results.bottom.ldos, results.top.ldos
    k = np.argmin(np.abs(coupled.box_z - np.mean(_interface(results))))  # mid-gap plane of the box

    plane = (coupled.box[..., k] - bottom.box[..., k] - top.box[..., k]).astype(float) / (BOHR_TO_ANGSTROM**3 * RY_TO_EV)
    n1, n2 = plane.shape[1:]
    power = np.abs(np.fft.fft2(plane, axes=(1, 2)) / (n1 * n2)) ** 2  # (n_energies, n1, n2)

    G = fft.g_vectors(results.coupled.atoms.cell.array, (n1, n2))
    classes = fft.g_origin(G, results.bottom.atoms.cell.array, results.top.atoms.cell.array)
    groups, index = fft.ripple_groups(G, classes)

    E = coupled.energies
    mask = (E >= PLOT_WINDOW_LDOS[0]) & (E <= PLOT_WINDOW_LDOS[1])
    group_power = np.stack([power[:, index == g].sum(axis=1) for g in range(len(groups))], axis=1)  # (n_energies, n_groups)

    # strongest groups over the plotted window, G = 0 excluded
    order = np.argsort(group_power[mask].sum(axis=0))[::-1]
    strongest = [g for g in order if groups[g][0] > 0][:FFT_PLOT_GROUPS]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(E[mask], power[mask, 0, 0], color="black", linewidth=2, label="G = 0 (planar average)")
    for g in strongest:
        norm, name, count = groups[g]
        ax.plot(E[mask], group_power[mask, g], label=rf"$|G|$ = {norm:.2f} $\AA^{{-1}}$, {name}, {count} G")

    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(r"$E-E_F$ (eV)")
    ax.set_ylabel(r"mid-gap ripple power (states$^2$/${\AA}^6$/eV$^2$)")
    ax.set_title(f"Mid-Gap LDOS Fourier, E-field={field}au, Correction: {vdw_scheme}")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(FFT_FIG_DIR / f"LDOS_Fourier_{field}au_{vdw_scheme}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def field_response(field_results, label, name):
    """
    Planar induced density per unit field at fixed geometry (Pourfath & Grasser, Fig. 3A): coupled bilayer (solid)
    against the two isolated layers summed (dashed). Curves that coincide for every field mean a linear response
    """
    print("PLOTTING FIELD RESPONSE")

    fields = sorted(f for f in field_results if f != 0)
    zero: Results = field_results[0][label]

    z = zero.coupled.charge_density.z
    bottom_z, top_z = _interface(zero)
    window = (z >= bottom_z - 3) & (z <= top_z + 3)

    fig, ax = plt.subplots(figsize=FIGSIZE_CHARGE_DENSITY)

    for field in fields:
        result: Results = field_results[field][label]

        coupled = (result.coupled.charge_density.averaged - zero.coupled.charge_density.averaged) / BOHR_TO_ANGSTROM**3 / field
        isolated = sum(getattr(result, layer).charge_density.averaged - getattr(zero, layer).charge_density.averaged
                       for layer in ("bottom", "top")) / BOHR_TO_ANGSTROM**3 / field

        line, = ax.plot(coupled[window], z[window], label=f"{field} au")
        ax.plot(isolated[window], z[window], color=line.get_color(), linestyle="--")

    ax.axhline(bottom_z, color="blue", linestyle=":", linewidth=0.8)
    ax.axhline(top_z, color="red", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    ax.set_xlabel(r"$\delta n(z) / E$ ($e/{\AA}^3$ per au)")
    ax.set_ylabel(r"z ($\AA$)")
    ax.set_title(f"Field Response (dashed: isolated layers), Correction: {label}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(POTENTIAL_FIG_DIR / f"Field_Response_{name}_{label}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
