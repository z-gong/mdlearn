# mdlearn
Machine learning of the thermodynamic properties of molecular liquids with graph neural network.

The first part of this work (Feedforward NN) has been published in the following article  
[Predicting Thermodynamic Properties of Alkanes by High-throughput Force Field Simulation and Machine Learning](https://doi.org/10.1021/acs.jcim.8b00407)

### Install required packages
  ```
  conda install matplotlib scikit-learn
  conda install -c pytorch pytorch
  conda install -c dglteam dgl-cuda10.1
  conda install -c openbabel openbabel
  conda install -c rdkit rdkit
  ```
 
### General steps
A machine learning workflow can be separated into four steps: fingerprint calculation, data splitting, model training and prediction.
(All script used here are located at `run` directory)
1. Calculate fingerprints
   ```
   ./gen-fp.py -i ../data/nist-CH-tc.txt -e morgan1,simple -o out
   ```

   Several encoders are available, which are suitable for different purposes.
   Multiple fingerprints can be (or should be) combined for better performance.
   * [wyz](https://doi.org/10.1021/acs.jcim.8b00407) - Handcrafted substructure fingerprint for predicting the properties of alkanes.
   * [morgan1](https://pubs.acs.org/doi/10.1021/ci100050t) - Extended connectivity substructure count with radius equal to one. In order to suppress overfitting, substructures occurred in less than 200 molecules are dropped.
   * simple - Four features describing the global structure of a molecule: number of atoms, molecular weight, maximum of shortest paths, number of rotatable bonds.

2. Split data to training and validation sets using 5-Fold cross-validation
   ```
   ./split-data.py -i ../data/nist-CH-tc.txt -o out
   ```
   
3. Train the model (see the following examples)

4. Predict property for new molecules (see the following examples)

### Case 1. Feedforward neural network with structural features

Here is an example of learning isotropic heat capacity of alkanes using handcrafted `xyz` fingerprint, as described in [this article](https://doi.org/10.1021/acs.jcim.8b00407).
   ```
   ./gen-fp.py -i ../data/alkanes-npt-2018v3.txt -e wyz -o out
   ./split-data.py -i ../data/alkanes-npt-2018v3.txt -o out
   ./train.py -i ../data/alkanes-npt-2018v3.txt -t Cp -f out/fp_wyz -p out/part-1.txt -o out/result
   ./predict.py -d out/result -e wyz -i CCCCCC,300,1
   ```
Here is an example of learning critical temperature of hydrocarbons using `morgan1` and `simple` fingerprints.
   ```
   ./gen-fp.py -i ../data/nist-CH-tc.txt -e morgan1,simple -o out
   ./split-data.py -i ../data/nist-CH-tc.txt -o out
   ./train.py -i ../data/nist-CH-tc.txt -t tc -f out/fp_morgan1,out/fp_simple -p out/part-1.txt -o out/result
   ./predict.py -d out/result -e predefinedmorgan1,simple -i CCCCCC
   ```
- Note that the length of `morgan1` fingerprint depends on the molecular structures in the training set.
  For prediction, `predefinedmorgan1` encoder should be used for calculating the `morgan1` fingerprint.

### Case 2. Graph convolutional network with force field features
Here is an example of learning cohesive energy using molecular graph and force field atom types.  
- The node feature is a one-hot vector representing the atom type of each atom in the force field.
- The molecular graphs and node features are loaded from `data/msdfiles.zip`.
   ```
   ./gen-fp.py -i ../data/All-npt_rand.txt -e simple -o out
   ./split-data.py -i ../data/All-npt_rand.txt -o out
   ./train-gcn.py -i ../data/All-npt_rand.txt -t einter -f out/fp_simple -p out/part-1.txt -o out/result
   ```

### Case 3. Edge attention graph convolutional network with force field features
Here is an example of learning cohesive energy using molecular graph, force field parameters and vacuum simulation results.  
- The node feature is the LJ parameters and charge of each atom in the force field.  
- Each 1-2, 1-3 and 1-4 pair is considered as an edge. The edge feature is the pair distance distribution extracted from vacuum simulation of a single molecule at 300 K.
- The molecular graphs are loaded from `data/msdfiles.zip`.
- The edge features are loaded from `data/distfiles.zip`.
- The node features are calculated from a force field file named `dump-MGI.ppf`, which should be put into the `data` directory.
   ```
   ./gen-fp.py -i ../data/All-npt_rand.txt -e simple -o out
   ./split-data.py -i ../data/All-npt_rand.txt -o out
   ./train-ffgcn.py -i ../data/All-npt_rand.txt -t einter -f out/fp_simple -p out/part-1.txt -o out/result
   ```
