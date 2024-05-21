The source code developed in Python 3.8 using PyTorch 1.13.0.The required python dependencies are given below.There is no additional non-standard hardware requirements.
```
torch>=1.7.1
dgl>=0.7.1
dgllife>=0.2.8
numpy>=1.20.2
scikit-learn>=0.24.2
pandas>=1.2.4
prettytable>=2.2.1
rdkit~=2021.03.2
yacs~=0.1.8
comet-ml~=3.23.1 # optional
```
The datasets folder contains all experimental data used in SiamDTI: BindingDB, BioSNAP and Human. In datasets/bindingdb,datasets/biosnap and datasets/human folder.S1 fold is for known drugs and targets,S2 is for novel drugs and targets

Run DrugBAN on Our Data to Reproduce Results
${dataset} could either be bindingdb, biosnap and human.
For the known drugs and targets experiments with SiamDTI, you can directly run the following command.
```
python main.py --cfg "configs/SiamDTI.yaml" --data ${dataset} --split S1
```
