### This folder stores pre-trained ML model for several properties

### Molecules are separated into two categories
* `CH` means hydrocarbons
* `All` means all molecules

### Six properties
* `tvap` is experimental boiling point from NIST TDE
* `tc` is experimental critical point from NIST TDE
* `density-l` is simulated liquid density
* `einter-l` is simulated cohesive energy
* `cp-l` is simulated isobaric heat capacity
* `hvap-lg` is simulated heat of vaporization

### Special case
* `wyz` contains the results for alkanes reported in JCIM 2018.
* `out-all-npt.tgz` contains the results of FFGCN graph model on cohesive energy
