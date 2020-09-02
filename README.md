# mdlearn
Machine learning of the thermodynamic properties of molecular liquids with graph neural network.

This is the enhanced version of our previously published work  
[Predicting Thermodynamic Properties of Alkanes by High-throughput Force Field Simulation and Machine Learning](https://doi.org/10.1021/acs.jcim.8b00407)

### Required Python packages
* Create and activate a separate conda environment  
`conda create -n ml`  
`source activate ml`

* Add tsinghua pytorch mirror (optional)  
`conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/`  
`conda config --set show_channel_urls yes`

* Install packages  
`conda install matplotlib scikit-learn pytorch`  
`conda install -c dglteam dgl-cuda10.1`  
`conda install -c openbabel openbabel`  
`conda install -c rdkit rdkit`

### Steps (Feedforward neural network)
*Following is an example of learning critical temperature of hydrocarbon using two simple fingerprints*

These scripts are located at directory `run`

* Activate ml environment
```
source activate ml
```

* Calculate fingerprints
```
./gen-fp.py -i ../data/nist-CH-tc.txt -e morgan1,simple -o fp
```
* Split data to train/validate datasets using 5-Fold cross-validation
```
./split-data.py -i ../data/nist-CH-tc.txt -o fp
```
* Train the model  
```
./train.py -i ../data/nist-CH-tc.txt -t tc -f fp/fp_morgan1,fp/fp_simple -p fp/part-1.txt -o out
```
* Predict property for new molecules
```
./predict.py -d out -e predefinedmorgan1,simple -i CCCCCC
```
*Note that if you train the model with **morgan1** or **morgan** fingerprint, you should use **predefinedmorgan1** or **predefinedmorgan** to predict*

### Steps (Graph convolutional network)
*Following is an example of learning critical temperature of hydrocarbon using molecular graph and atom types defined in force field*
