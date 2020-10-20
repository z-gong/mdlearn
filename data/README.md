### Experimental data
 
- `nist-All-tc.txt`

  Experimental critical temperatures from NIST ThermoData Engine.
 
- `nist-All-tvap.txt`

  Experimental normal boiling temperatures from NIST ThermoData Engine.
  
### Simulation data

- `alkanes-npt-2018v3.txt`

  Calculated density, cohesive energy and isobaric heat capacity for 876 alkanes at various temperatures and pressures.  
  The details of molecules, force field and simulation conditions are described in [this article](https://doi.org/10.1021/acs.jcim.8b00407).

- `All-npt_rand.txt`

  Calculated density, cohesive energy and isobaric heat capacity for over 9000 molecules at various temperatures and pressures.  
  The molecules are made of C, H, O and N elements.

 
### Data for graph convolutional neural network 

- `msdfiles.zip`

  Topologies in MSD format for molecules in `All-npt_rand.txt`.  
  Atom types and coordinates have already been assigned.  
  This dataset is used for building graphs and assigning node features for graph convolutional model.
  
- `distfiles.zip`

  1-2, 1-3 and 1-4 pair distance distributions extracted from vacuum simulations at 300 K for molecules in `All-npt_rand.txt`.  
  This dataset is used as edge features for edge attention graph convolutional model.

