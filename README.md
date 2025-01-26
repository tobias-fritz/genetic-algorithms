# Atom Mapping Using a Genetic Algorithm

## Setup

- All dependencies can be installed using either pip or conda.<br>
`pip install -r requirements.txt` <br>
`conda install --file requirements.txt` <br>

## How to Use

Compute the best fit of atom indices of two isomeric molecule structures. This can, for example, be used to keep track of atoms in independently indexed educt or product states. <br>
The full genetic algorithm (GA) is found in `main.py` and can be run directly from the command line. The components of the GA are found [here](source/GA_util.py) and can be independently tweaked if needed. The principle GA is based on [this](https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/) article from machinelearningmastery.com with the tournament selection implemented based on: https://en.wikipedia.org/wiki/Tournament_selection

## Example Usage

To run the genetic algorithm, you can use the following command in your terminal:

```bash
python main.py <educt_path> <product_path> <n_generations> <pop_size> <cross_rate> <mut_rate>
```

For example:

```bash
python main.py educt.xyz product.xyz 100 100 0.9 0.1
```

For a detailed example, refer to the [example Jupyter notebook](example.ipynb).

## Repository Structure

- `main.py`: Main file containing the genetic algorithm implementation.
- `src/GA_util.py`: Utility functions for the genetic algorithm.
- `example.ipynb`: Jupyter notebook demonstrating how to use the genetic algorithm.
- `requirements.txt`: List of dependencies required to run the project.