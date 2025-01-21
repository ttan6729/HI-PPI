# Protein--protein interaction prediction with interaction-specific learning and hierarchical structure information
This repository contains an official implementation of HI-PPI and datasets used for evaluating PPI prediction model.
----
## Setup
```
Download the pre-generated structure features of SHS27K and SHS148K from [SHS27K data]
<a href='https://drive.google.com/file/d/1SEplMBH36521XsG0yIDLY7X5xRaN7Ekb/view?usp=sharing and https://drive.google.com/file/d/1Lqyg05aTbXYTb-uXpl3F36TfY7VTmk-B/view?usp=sharing'>SHS27Kdata<\a>
```
Unzip the file in the folder features
----
### Usage

```
usage: HIPPI [-h] [-m M] [-o O] [-i I] [-i1 I1] [-i2 I2] [-i3 I3] [-e E] [-b B] [-ln LN] [-L L]
            [-Loss LOSS] [-jk JK] [-ff FF] [-hl HL] [-sv SV] [-cuda CUDA] [-force FORCE]
            [-PSSM PSSM]

options:
  -h, --help    show this help message and exit
  -m M          mode, optinal value: read,bfs,dfs,rand,
  -o O
  -i I
  -i1 I1        sequence file
  -i2 I2        relation file
  -i3 I3        file path of test set indices (for read mode)
  -e E          epochs
  -b B          batch size
  -ln LN        graph layer num
  -L L          length for sequence padding
  -Loss LOSS    loss function
  -jk JK        use jump knowledege to fuse pair or not
  -ff FF        option for protein pair representaion
  -hl HL        hidden layer
  -sv SV        if save dataset path
  -cuda CUDA    if use cuda
  -force FORCE  if write to existed output file
```
### Sample command for training and testing
```
python3 main.py -m bfs -t HI-PPI -i data/27K.txt -i4 features/27K -o test -e 100 -mainfold Hyperboloid
```


