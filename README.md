# Atom mapping using a genetic algorithm

## Setup

- All dependencies can be installed using either pip or conda.<br>
`pip install -r requirements.txt` <br>
`conda install --file requirements.txt` <br>

## How to use

Compute the best fit of atom indices of two isomeric molecule structures. This can for example be used to keep track of atoms in indepentently indexed educt or product states. <br>
The full GA is found in GA.py and I recomend to use with a jupyter notebook (see [example](example.ipynb)). The components of the GA are found [here](source/GA_util.py) and can be independently tweaked if needed (In the current version, crossover is not used in GA.py since it is not ready yet).  <br>
The principle GA is basedon [this](https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/) aricle from machinelearningmastery.com with the turnament selection implemented based on: https://en.wikipedia.org/wiki/Tournament_selection