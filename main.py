from config import *
from structures import *
from systems import *
from results import *
import qe.pw as pw
import qe.dos as dos
import qe.pp as pp
import qe.average as average
import convergence
import plotter as plot


def run_calculations(structure, system:SystemSettings, path, eamp, xc, label):
    
    print(f"\n{label} | {path.name.upper()} LAYER(S) COMPUTATIONS")
    print("-" * 35)

    if RUN_QE:
        scf = pw.scf(structure, path, eamp, system.kgrid, xc)
        nscf = pw.nscf(structure, path, eamp, system.kgrid_dense, xc)
    else: 
        scf = pw._read_output(path / "scf.pwo")
        nscf = pw._read_output(path / "nscf.pwo")
    
    fermi_energy = nscf.calc.get_fermi_level()
    
    if RUN_POTENTIAL:
        potential, potential_pp_path = pp.potential(path)
        # potential_averaged = average.potential(path, potential_pp_path)
    else:
        # potential, potential_pp_path = pp._read_output(path / "potential.cube", 6), path / "data" / f"potential.pp.dat"        
        data, atoms, xy_averaged, z = pp._read_output(path / "potential.xsf", 5)
        potential = Potential(data=data, atoms=atoms, averaged=xy_averaged, z=z)


    if RUN_CHARGE_DENSITY:
        charge, charge_pp_path = pp.charge_density(path)
    else:
        # charge, charge_pp_path = pp._read_output(path / "charge.cube", 6), path / "data" / f"charge.pp.dat"
        data, atoms, xy_averaged, z = pp._read_output(path / "charge.xsf", 5)
        charge = ChargeDensity(data=data, atoms=atoms, averaged=xy_averaged, z=z)
    
    if RUN_DOS:
        dos_ = dos.calculate(path, fermi_energy)
    else:
        dos_ = dos._read_output(path / "dos.out")
    
    if RUN_LDOS:
        ldos, ldos_pp_path, energies = pp.ldos(path, fermi_energy)
        for f in (path / "data").glob("*.save/wfc*.dat"):
            f.unlink()
        # ldos_averaged = average.ldos(path, ldos_pp_path, energies)
    else:
        ldos = pp._load_ldos(path)
        
    # ldos_averaged = average.ldos(path, ldos_pp_path, energies)
        

    results = System(name=f"{path.name}",
                            atoms=structure,
                            path=path,
                            fermi_energy=fermi_energy,

                            potential=potential,
                            charge_density=charge,
                            dos = dos_,
                            ldos = ldos)
    
    return results


def main():

    energies = [0]
    for eamp in energies:
        for system in SYSTEMS:
            system:SystemSettings  
            structure = system.structure
            
            path = OUT_DIR / f"{system.name}" / f"field_{str(eamp)}"

            print()
            print("*" * 35)
            print(f"E-field = {str(eamp)}au")
            print("*" * 35)

            print("\nCREATING GRAPHENE BILAYERS")
            bilayer = structure.bilayer()
            
            if RUN_CONVERGENCE and eamp==0:  # only run for efield=0au
                convergence.test_kgrid(bilayer, system, "scf")
                convergence.test_kgrid(bilayer, system, "nscf")
                convergence.test_ecutwfc(bilayer, system)
            
            if RUN_QE:
                print("RELAXING COUPLED BILAYER")
                relaxed_coupled = pw.relax(bilayer, path, eamp, system.kgrid)
            
            else:
                relaxed_coupled = pw._read_output(path / "relax.pwo")
            
            print("\nEXTRACTING FROZEN LAYERS")
            isolated_bottom, isolated_top = structure.isolate_bilayer(relaxed_coupled)
            
            # store pbe+d3 results as d3 and vdw results as c09
            results = {}
            for xc in XC:
                if xc is None:
                    label = "d3"
                else:
                    label = xc.split("-")[-1]
                
                tag = f"{system.name}_{label}"
                
                PATH_COUPLED = path / (f"PBE-{label}" if xc is None else xc) / f"coupled" 
                PATH_BOTTOM = path / (f"PBE-{label}" if xc is None else xc) / f"bottom"
                PATH_TOP = path / (f"PBE-{label}" if xc is None else xc) / f"top"
                
                coupled = run_calculations(relaxed_coupled, system, PATH_COUPLED, eamp, xc, label)
                bottom = run_calculations(isolated_bottom, system, PATH_BOTTOM, eamp, xc, label)
                top = run_calculations(isolated_top, system, PATH_TOP, eamp, xc, label)
                
                result = Results(coupled, bottom, top)
                results[label] = result

                plot.potential_individual(result, eamp, tag)
                plot.potential_subtracted(result, eamp, tag)         
                plot.charge_density_subtracted(result, eamp, tag)
                plot.dos_subtracted(result, eamp, tag)
                plot.dos_individual(result, eamp, tag)
                #plot.fermi_ldos_subtracted(result, eamp, tag)
                #plot.vacuum_ldos_subtracted(result, eamp, tag)
                plot.ldos_individual(result, eamp, tag)
                plot.ldos_subtracted(result, eamp, tag)
            
            pp.vdw_charge_difference(path, eamp)  # creates corresponding xsf file
            plot.plot_vdw_charge_difference(results, eamp, system.name)


if __name__ == "__main__":
    main()
