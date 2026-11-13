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
import fft as fft


def run_calculations(structure, system:SystemSettings, path, eamp, xc, label, z_mid):
    
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
        ldos, ldos_pp_path, energies = pp.ldos(path, fermi_energy, z_mid)
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
    for system in SYSTEMS:
        system:SystemSettings  
        structure = system.structure
        system_path = OUT_DIR / system.name

        print("\nCREATING BILAYER")
        bilayer = structure.bilayer()
        
        if RUN_CONVERGENCE:  # only run for efield=0au
            convergence.test_kgrid(bilayer, system, "scf")
            convergence.test_kgrid(bilayer, system, "nscf")
            convergence.test_ecutwfc(bilayer, system)
        
        if RUN_QE:
            print("RELAXING COUPLED BILAYER")
            relaxed_coupled = pw.relax(bilayer, system_path, 0, system.kgrid)
        else:
            relaxed_coupled = pw._read_output(system_path / "relax.pwo")
        
        print("\nEXTRACTING FROZEN LAYERS")
        isolated_bottom, isolated_top = structure.isolate_bilayer(relaxed_coupled)
        
        z_bottom = isolated_bottom.positions[:, 2].max()
        z_top = isolated_top.positions[:, 2].min()
        z_mid = (z_bottom + z_top) / 2
        
        field_results = {}
        
        for eamp in FIELDS:
            path = system_path / f"field_{eamp}"

            print()
            print("*" * 35)
            print(f"E-field = {eamp} au")
            print("*" * 35)
            
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
                
                coupled = run_calculations(relaxed_coupled, system, PATH_COUPLED, eamp, xc, label, z_mid)
                bottom = run_calculations(isolated_bottom, system, PATH_BOTTOM, eamp, xc, label, z_mid)
                top = run_calculations(isolated_top, system, PATH_TOP, eamp, xc, label, z_mid)
                
                result = Results(coupled, bottom, top)
                results[label] = result
                
                # fft_charge_output_path = path / "fft" / f"charge_fft_{label}.npz"
                # grid, spectrum = fft.save_induced_charge_fft(PATH_COUPLED / "charge.xsf",
                #                                              PATH_BOTTOM / "charge.xsf",
                #                                              PATH_TOP / "charge.xsf",
                #                                              fft_charge_output_path, z_mid)
                # F, G = fft.inplane_spectrum(spectrum, grid.cell, grid.origin)
                # box_mask = fft.interface_mask(grid.z, z_mid)
                # box_mean = (grid.data[:, :, box_mask].mean() / BOHR_TO_ANGSTROM**3)
                # np.testing.assert_allclose(F[0, 0], box_mean, rtol=1e-10, atol=1e-12)

                plot.potential_individual(result, eamp, tag)
                if eamp == 0:
                    plot.potential_subtracted(result, eamp, tag)         
                plot.charge_density_subtracted(result, eamp, tag)
                plot.dos_subtracted(result, eamp, tag)
                plot.dos_individual(result, eamp, tag)                
                plot.ldos_individual(result, eamp, tag)
                # plot.ldos_subtracted(result, eamp, tag)
                plot.fermi_ldos_subtracted(result, eamp, tag)
                #plot.vacuum_ldos_subtracted(result, eamp, tag)
                plot.slice_plot(plot._induced_charge(result), result, system, f"Induced Charge, E-field={eamp}au, Correction: {label}",
                                CHARGE_DENSITY_FIG_DIR / f"Charge_Slice_{eamp}au_{tag}.png")
                plot.charge_columns(result, eamp, tag)
                plot.charge_fourier(result, eamp, tag)
                plot.ldos_columns(result, eamp, tag)
                plot.ldos_fourier(result, eamp, tag)
            
            pp.vdw_charge_difference(path, eamp)  # creates corresponding xsf file
            plot.plot_vdw_charge_difference(results, eamp, system.name)
            
            for label in results:
                if label != "d3":
                    plot.slice_plot(plot._induced_charge(results[label]) - plot._induced_charge(results["d3"]), results[label], system,
                                    f"{label} - d3, E-field={eamp}au",
                                    CHARGE_DENSITY_FIG_DIR / f"vdW_Charge_Slice_{system.name}_{label}_{eamp}au.png")
            
            field_results[eamp] = results
        
        if 0 in field_results and len(field_results) > 1:
            for label in field_results[0]:
                plot.field_response(field_results, label, system.name)

                field = max(field_results)
                induced_per_field = (field_results[field][label].coupled.charge_density.data 
                                     - field_results[0][label].coupled.charge_density.data) / BOHR_TO_ANGSTROM**3 / field
                plot.slice_plot(induced_per_field, field_results[0][label], system,
                                f"Field-Induced Charge per au, {field}au, Correction: {label}",
                                POTENTIAL_FIG_DIR / f"Field_Slice_{system.name}_{label}.png",
                                label=r"$\delta n / E$ ($e/{\AA}^3$ per au)")   

if __name__ == "__main__":
    main()
