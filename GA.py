
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

from source.pairwise_distance_matrix import pairwise_distance_matrix
from source.GA_util import init_pop, objective, tournament_selection, mutation, crossover
from source.xyz_parser import XYZ_reader

#===============================================================================================

def genetic_atom_mapping(educt_path,product_path,n_generations,pop_size,cross_rate,mut_rate):
    '''
    
    
    
    '''

    product_xyz = XYZ_reader(product_path)

    educt   = pairwise_distance_matrix(XYZ_reader(educt_path)  , inverse = True, unit = True, exp = None)
    product = pairwise_distance_matrix(XYZ_reader(product_path), inverse = True, unit = True, exp = None)

    # get the reference which atoms can get which places
    reference = [product_xyz.loc[product_xyz['Element'] == element].index.to_list() for element in product_xyz['Element'].unique()]
    
    options = list()
    for i in reference:
        for column in i :
            options.append(i)
     
    # initial population based on estimated best candidate 
    pop = [ j for i,j in enumerate(init_pop(educt,product,reference)) if i < (pop_size - 1)]

    # set initial guess
    best, best_eval = 0, objective(pop[0],educt,product)

    # enumerate generations
    for gen in range(n_generations):
        
        # evaluate all candidates in the population
        scores = [objective(i,educt,product) for i in pop]

        #print(scores)
        #check if there's a new best in this generation
        #best,best_eval = update_best(gen,pop,scores,best,best_eval)
        for i in range(len(scores)):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
                
        # select parents in turnament selection
        parents = [tournament_selection(pop,scores) for _ in range(pop_size)]

        # create the next generation    
        children = list()
        for i in range(0,pop_size,2):

            # get pair of parents
            p1, p2 = parents[i], parents[i+1]

            for child in [p1,p2]:#crossover(p1,p2,cross_rate):

                child = mutation(child,options,mut_rate)

                children.append(child)

        pop = children

    return [best,best_eval]