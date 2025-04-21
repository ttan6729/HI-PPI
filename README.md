# Protein--protein interaction prediction with interaction-specific learning and hierarchical structure information
This repository contains an official implementation of HI-PPI and datasets used for evaluating PPI prediction model.
----
## Setup
1. Download the code in https://github.com/ttan6729/HI-PPI .
2. Download and unzip the pre-generated structure features of SHS27K and SHS148K in the same folder from [SHS27K data](https://drive.google.com/file/d/1SEplMBH36521XsG0yIDLY7X5xRaN7Ekb/view?usp=sharing) and [SHS148K data ](https://drive.google.com/file/d/1SEplMBH36521XsG0yIDLY7X5xRaN7Ekb/view?usp=sharing). Or generate the features based on other PPI data: ```python3 main.py -m data -i1 [sequence file]  -i2 [interaction file] -sf [folder that contains pdb file of each protein] -o [output name]```
3. See usage for the command of HI-PPI, sample command for the prvoided data:
```
python3 main.py -m bfs -t HI-PPI -i 27K.txt -i4 27K -o test -e 100 -mainfold Hyperboloid
```

### Usage

```
usage: PPIM [-h] [-m M] [-o O] [-t T] [-i I] [-i1 I1] [-i2 I2] [-i3 I3] [-i4 I4] [-s1 S1] [-e E] [-b B] [-ln LN] [-L L]
            [-Loss LOSS] [-ff FF] [-hl HL] [-sv SV] [-cuda CUDA] [-force FORCE] [-mainfold MAINFOLD] [-pr PR]

options:
  -h, --help          show this help message and exit
  -m M                mode, optinal value: bfs,dfs,rand,read,data
  -o O
  -t T                for test distintct models
  -i I                path for sequnce and relation file
  -i1 I1              sequence file
  -i2 I2              relation file
  -i3 I3              file path of test set indices (for read mode)
  -i4 I4              prefix for the path of embedding
  -s1 S1              file path for structure file
  -e E                epochs
  -b B                batch size
  -ln LN              graph layer num
  -L L                length for sequence padding
  -Loss LOSS          loss function
  -ff FF              feature fusion option, default mul
  -hl HL              hidden layer
  -sv SV              if save dataset path
  -cuda CUDA          if use cuda
  -force FORCE        if write to existed output file
  -mainfold MAINFOLD  any of the following: Euclidean, Hyperboloid, PoincareBall
  -pr PR              perturbation ratio

```
### Sample command for training and testing
```
python3 main.py -m bfs -t HI-PPI -i data/27K.txt -i4 features/27K -o test -e 100 -mainfold Hyperboloid
```


