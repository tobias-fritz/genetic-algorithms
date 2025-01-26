#!/usr/bin/env python3
#===============================================================================================
# Script for scoring the atom indexing in xyz files of two at the very least isomeric structures
# using a genetic algorithm to reduce computational cost compared to brute force approach
# Date: 30.05.2022
# Author: Tobias Fritz
# Summary:
# Based on a previous atom mapping appraoch using permutation Reads two xyz files that have to 
# at the very least have the same molecular formulae. E.g. Structures of product and educt. 
# Based on the pairwise distance matrix the indexing of the two structures is compared to map 
# atoms of e.g. the educt to the corresponding atom in the product structure. 
#===============================================================================================

from src.pairwise_distance_matrix import pairwise_distance_matrix
from src.GA_util import init_pop, objective, tournament_selection, mutation, crossover, gen_reference
from src.xyz_parser import XYZ_reader, XYZ_writer
import argparse
from scipy.spatial import cKDTree


def genetic_atom_mapping(educt_path: str,
                         product_path: str,
                         n_generations: int = 100,
                         pop_size: int = 100,
                         cross_rate: float = 0.9,
                         mut_rate: float = 0.1):
    '''Genetic algorithm for atom mapping of two isomeric structures
    
    Args:
        educt_path: (str) Path to the xyz file of the educt (Default: None)
        product_path: (str) Path to the xyz file of the product (Default: None)
        n_generations: (int) Number of generations (Default: 100)
        pop_size: (int) Population size (Default: 100)
        cross_rate: (float) Crossover rate (Default: 0.9)
        mut_rate: (float) Mutation rate (Default: 0.1)
    Returns:
        best: (list) Best candidate
        best_eval: (float) Best evaluation score
    '''

    product_xyz = XYZ_reader(product_path)
    educt_xyz = XYZ_reader(educt_path)

    # Sort product and educt xyz based on element
    product_xyz = product_xyz.sort_values(by='Element').reset_index(drop=True)
    educt_xyz = educt_xyz.sort_values(by='Element').reset_index(drop=True)
    
    educt = pairwise_distance_matrix(educt_xyz, inverse=True, unit=True, exp=None)
    product = pairwise_distance_matrix(product_xyz, inverse=True, unit=True, exp=None)

    # Get the reference which atoms can get which places
    reference, options = gen_reference(product_xyz)

    # Initialize the population
    pop = [j for i, j in enumerate(init_pop(educt, product, reference,n=5)) if i < (pop_size - 1)]
    
    # Set initial guess
    best, best_eval = pop[0], objective(pop[0], product, educt)
    theoretical_best = objective(list(educt.index), educt, educt)
    print("Initial best f(%s) = %.3f (Theoretical best score = %.3f)" % (pop[0], best_eval, theoretical_best))

   
    for gen in range(n_generations):
        
        # Evaluate all candidates in the population
        scores = [objective(i, product, educt) for i in pop]

        # Check if there's a new best in this generation
        for i in range(len(scores)):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen, best, best_eval))
        
        # Select parents in tournament selection
        parents = [tournament_selection(pop, scores) for _ in range(pop_size)]
        
        # Create the next generation    
        children = []
        for i in range(0, pop_size, 2):
            # Get pair of parents
            p1, p2 = parents[i], parents[i + 1]

            for child in [p1,p2]: #crossover(p1, p2, cross_rate):
                child = mutation(child, options, mut_rate)
                children.append(child)

        pop = children

    mapped_educt = educt_xyz.iloc[list(best)].reset_index(drop=True)

    return {"mapping": best, "score": best_eval, "mapped_educt": mapped_educt, 
            "product_xyz": product_xyz, "educt_xyz": educt_xyz}

def map_similarity_molecule_element(file1, file2):
    """Map the atom indices of two molecules

    cKDTree is used to map the atom indices of two molecules based on the element type.
    This works by finding the nearest atom in the product molecule for each atom in the educt molecule.
    The mapping is then enforced to have the same element type.

    Args:
        file1: (str) Path to the xyz file of the first molecule
        file2: (str) Path to the xyz file of the second molecule
    Returns:
        mapping: (tuple) Mapping of the atom indices
        mapped_educt: (DataFrame) Mapped educt molecule
        product_xyz: (DataFrame) Product molecule
        educt_xyz: (DataFrame) Educt molecule
    """

    product_xyz = XYZ_reader(file1)
    educt_xyz = XYZ_reader(file2)

    product_xyz = product_xyz.sort_values(by='Element').reset_index(drop=True)
    educt_xyz = educt_xyz.sort_values(by='Element').reset_index(drop=True)

    tree = cKDTree(product_xyz[['x', 'y', 'z']].values.tolist())
    _, idx = tree.query(educt_xyz[['x', 'y', 'z']].values.tolist())
   
    # Get the mapping, and enforce the element to be the same
    educt_map = [j for i, j in enumerate(idx) if educt_xyz['Element'][i] == product_xyz['Element'][j]]
    product_map = [i for i, j in enumerate(idx) if educt_xyz['Element'][i] == product_xyz['Element'][j]]
    mapping = (educt_map, product_map)

    # Remap educt_xyz based on idx
    mapped_educt = educt_xyz.iloc[mapping[0]].reset_index(drop=True)

    return {"mapping": mapping, "mapped_educt": mapped_educt, "product_xyz": product_xyz, "educt_xyz": educt_xyz}



def main():
    """Main function for the atom mapping task"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Program for atom mapping of two isomeric structures')
    parser.add_argument('Model', type=str, help='Model to use for atom mapping (Options: genetic, similarity_tree)')
    parser.add_argument('educt', type=str, help='Path to the xyz file of the educt')
    parser.add_argument('product', type=str, help='Path to the xyz file of the product')
    parser.add_argument('--n_generations', type=int, help='Number of generations', default=100)
    parser.add_argument('--pop_size', type=int, help='Population size', default=100)
    parser.add_argument('--cross_rate', type=float, help='Crossover rate', default=0.9)
    parser.add_argument('--mut_rate', type=float, help='Mutation rate', default=0.1)
    args = parser.parse_args()

    initargs = {'educt': args.educt,
                'product': args.product}
    geneticargs = {'n_generations': args.n_generations,
                   'pop_size': args.pop_size,
                   'cross_rate': args.cross_rate,
                   'mut_rate': args.mut_rate}
    
    print(f"Running {args.Model} with the following arguments:")
    for key, value in initargs.items():
        print(f'{key}: {value}')
    if args.Model == 'genetic':
        for key, value in geneticargs.items():
            print(f'{key}: {value}')

    # Check if arguments are valid
    if args.cross_rate < 0 or args.cross_rate > 1:
        raise ValueError('Crossover rate must be between 0 and 1')
    if args.mut_rate < 0 or args.mut_rate > 1:
        raise ValueError('Mutation rate must be between 0 and 1')
    try: 
        XYZ_reader(args.educt)
    except FileNotFoundError:
        raise FileNotFoundError('Educt file not found')
    try:
        XYZ_reader(args.product)
    except FileNotFoundError:
        raise FileNotFoundError('Product file not found')
    if len(XYZ_reader(args.educt)) != len(XYZ_reader(args.product)):
        raise ValueError('Educt and product must have the same number of atoms')

    # Run genetic algorithm
    if args.Model == 'genetic':
        result = genetic_atom_mapping(args.educt, args.product, args.n_generations, args.pop_size, args.cross_rate, args.mut_rate)
    if args.Model == 'similarity_tree':
        result = map_similarity_molecule_element(args.educt, args.product)

    print(result['mapping'])

if __name__ == '__main__':
    main()