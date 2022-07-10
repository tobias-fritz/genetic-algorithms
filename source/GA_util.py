
import itertools
import pandas as pd
import numpy as np
import random


def init_pop(educt,product,reference, n = 2):
    ''' Obtain an initial population generator, based on random permutatation of
        the n best options for every position

        param:
            educt:      pairwise distance matrix of the educt molecule
            product:    pairwise distance matrix of the product molecule
            reference:  list of possible atom indices at each positions
            n:          number of options per position (default = 2)
        '''

    # compute similarity scores between educt and product
    scores = pd.DataFrame(np.dot(educt, product.transpose()))

    #print(n ** product.shape[0])

    # initiate options vector
    options = list(range(product.shape[0]))
    for i in reference:
        for column in i :
            # fill the options vector at each position i with the nlargest possibilities for that
            options[column] = scores[column].iloc[i].nlargest(n).index.to_list() 
    
    # generator object of all possible permutations
    permutations = (p for p in itertools.product(* options) if len(set(p)) == len(p))

    return permutations


def objective(p,educt,product):
    ''' scoring function for the GA, criterion is the maximum dot product of a given 
        product permutation with the educt.
        
        param:
            p:          permutation
            educt:      pairwise distance matrix of the educt molecule
            product:    pairwise distance matrix of the product molecule  
        '''
    
    prod = product.to_numpy()

    return np.dot(educt, prod[p,:][:,p].transpose()).diagonal().sum()


def update_best(gen,pop,scores,best,best_eval):
    ''' Check each generation and update current best option
    
        param:
            gen:        the current generation (int)
            pop:        the current population
            scores:     the scores of the current population
            best:       the best individual (permuatation)
            best_eval:  score of the best individual
    '''
    
    # get the set of scores and sort by highest score
    sort_scores = frozenset(scores)
    sort_scores = sorted(sort_scores, reverse=True)

    # if a solution is better than the current best, update current best
    if scores[0] > best_eval:
        best, best_eval = scores[0], pop[scores.index(sort_scores[0])]
    
    print(">%d, new best f(%s) = %.3f" % (gen,  best, best_eval))

    return best,best_eval

def mutation(p,reference,mutation_rate):
    ''' Mutate a permutation vector at a number of positions by randomly selecting
        from the reference at each position 
        param:
            p:              permuattion vector to permute
            reference:      list with possible indices at each position
            mutation_rate:  float between 0 and one giving the percentage of mutations
        '''

    # translate mutation rate to a number of positions to mutate
    n_pos = int(len(p) * mutation_rate)

    # randomly determine mutation positions
    mut_positions =  random.sample(range(len(p)), n_pos)

    # iterate over positions to be mutated
    for i in mut_positions:
        
        # randomly select one of the options in the refernce list for position i
        mut = random.choice(reference[i])
        # change the prior position of atom mut to the one at pos i and vice versa
        p[p.index(mut)] = p[i]
        p[i] = mut

    return p


def tournament_selection(pop,scores,t_size=3):

    # randomly select t_size chromosomes/ individuals from the pop for the tunament
    sel_idx = random.choices(range(len(pop)),k = t_size)
    
    # get turnament population and their scores
    t_pop = list(np.array(pop)[sel_idx])
    t_scores = list(np.array(scores)[sel_idx])

    # get the set of t_scores and sort by highest score in the turnament selection
    sorted_scores = frozenset(t_scores)
    sorted_scores = sorted(sorted_scores, reverse=True)

    return list(t_pop[t_scores.index(sorted_scores[0])])

