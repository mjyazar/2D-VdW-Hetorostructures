import qe.pw as pw
import qe.dos as dos
import numpy as np
from config import *
from systems import *

import matplotlib.pyplot as plt
import shutil


def _graph_convergence(values, energies, field, header, parameter, name):
    
    fig, ax = plt.subplots()
    
    ax.plot(values, energies, "o-", color="black")
    ax.set_title(f"{name} {parameter} Parameter Convergence at E-field={field}au")
    
    if parameter.startswith("kgrid"):
        ax.set_xlabel(f"{parameter} Parameter x --> (x, x, 1)")
    else:
        ax.set_xlabel(f"{parameter} Parameter")

    ax.set_ylabel(header)
    
    fig.savefig(CONV_FIG_DIR / f"{name}_{parameter}_Convergence_{field}au", bbox_inches="tight")
    plt.close(fig)


def test_kgrid(structure, system:SystemSettings, computation, field=0):
    """
    Convergence testing of kgrid by iterating through from values of x and y 
    ranging from 3 to upper bound, with z being kept constant at 1.
    Involves running the scf process and plotting the total energy against kgrid values.
    return: scf.pwo file read using ASE
    """
    
    # list to store values for plotting
    kgrid_values = []
    energies = []
    doses = []
    
    if computation == "scf":
        calc_path = CONV_DATA_DIR / system.name / "scf"
        l_bound, u_bound, step = system.conv_scf

    elif computation == "nscf":
        calc_path = CONV_DATA_DIR / system.name / "nscf"
        l_bound, u_bound, step = system.conv_nscf
        
        pw._calculate("scf", structure, calc_path, system.kgrid, efield=field)
    
    for i in range(l_bound, u_bound+1, step):
        kgrid = (i, i, 1)
                
        if computation == "scf":
            scf = pw._calculate("scf", structure, calc_path, kgrid, efield=field)
            
            total_energy = scf.get_potential_energy()  # extract total energy
            energies.append(total_energy)
            
            print(f"KGRID = ({i}, {i}, 1) --> TOTAL ENERGY: {total_energy}")
            
        elif computation == "nscf":
            nscf = pw._calculate("nscf", structure, calc_path, kgrid, efield=field)
            
            fermi_energy = nscf.calc.get_fermi_level()
            dos_ = dos.calculate(calc_path, fermi_energy)
            doses.append((dos_.energy - fermi_energy, dos_.dos))
            
            print(f"KGRID = ({i}, {i}, 1) --> DOS: {dos_}")
        
        kgrid_values.append(i)
    
    kgrid_values = np.array(kgrid_values)
    energies = np.array(energies)

    if computation == "scf":
        data = np.column_stack((kgrid_values, energies))
        np.savetxt(CONV_DATA_DIR / system.name / "scf kgrid Convergence.txt", data, header="kgrid total_energy_eV")
        _graph_convergence(kgrid_values, energies, field, "Total Energy (eV)", "kgrid_scf", system.name)

    # plot dos for each kgrid density compared against last density i.e. [-1]
    if computation == "nscf":
        e_reference, dos_reference = doses[-1]  # take densest grid as reference (relative to E_F)
        error = []
        
        for energy, dos in doses:
            d_interp = np.interp(e_reference, energy, dos)  # common energy grid
            error.append(np.abs(d_interp - dos_reference).sum() / dos_reference.sum())  # relative DOS error

        data = np.column_stack((kgrid_values, error))
        np.savetxt(CONV_DATA_DIR / system.name / "nscf kgrid Convergence.txt", data, header="kgrid relative_dos_error")
        _graph_convergence(kgrid_values, error, field, "Relative DOS error", "kgrid_nscf_dos", system.name)
    
    # delete large metadata
    shutil.rmtree(calc_path / "data", ignore_errors=True)  # don't crash if folder not found
    return (kgrid_values, energies)


def test_ecutwfc(structure, system:SystemSettings, field=0):
    """
    Convergence testing of ecutwfc by iterating through values 30 to upper bound in increments of 10.
    Involves running the scf process and plotting the total energy against ecutwfc values.
    return: scf.pwo file read using ASE
    """
    
    # list to store values for plotting
    ecutwfc_values = []
    energy_values = []
    
    for ecutwfc in range(30, ECUTWFC_BOUND+1, 10):
        
        calc_path = CONV_DATA_DIR / system.name / "ecutwfc" / str(ecutwfc)
        
        scf = pw._calculate("scf", structure, calc_path, system.kgrid, ecutwfc=ecutwfc, ecutrho=4*ecutwfc, efield=field)
        
        total_energy = scf.get_potential_energy()  # extract total energy
        
        ecutwfc_values.append(ecutwfc)
        energy_values.append(total_energy)
        
        # delete large metadata
        shutil.rmtree(calc_path / "data", ignore_errors=True)  # don't crash if folder not found
    
        print(f"ECUTWFC = {ecutwfc} --> TOTAL ENERGY: {total_energy}")

    ecutwfc_values = np.array(ecutwfc_values)
    energy_values = np.array(energy_values)
    
    # save the results as a .txt file for reference
    header="ecutwfc total_energy_eV"
    np.savetxt(CONV_DATA_DIR / system.name / "ecutwfc Convergence.txt", np.column_stack((ecutwfc_values, energy_values)), header=header)
    
    _graph_convergence(ecutwfc_values, energy_values, field, header, "ecutwfc", system.name)

    return (ecutwfc_values, energy_values)
