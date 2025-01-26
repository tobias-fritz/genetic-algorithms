import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import pandas as pd
import numpy as np
import random


def init_pop(educt, product, reference, n=5):
    ''' Obtain an initial population generator, based on random permutation of
        the n best options for every position

        param:
            educt:      pairwise distance matrix of the educt molecule
            product:    pairwise distance matrix of the product molecule
            reference:  list of possible atom indices at each positions
            n:          number of options per position (default = 2)
    '''

    # compute similarity scores between educt and product
    scores = pd.DataFrame(np.dot(educt, product.transpose()))

    # initiate options vector
    options = list(range(product.shape[0]))
    for i in reference:
        for column in i:
            # fill the options vector at each position i with the nlargest possibilities for that
            options[column] = scores[column].iloc[i].nlargest(n).index.to_list()
    
    # generator object of all possible permutations
    permutations = (p for p in itertools.product(*options) if len(set(p)) == len(p))

    return permutations


def objective(p, educt, product):
    ''' scoring function for the GA, criterion is the maximum dot product of a given 
        product permutation with the educt.
        
        param:
            p:          permutation
            educt:      pairwise distance matrix of the educt molecule
            product:    pairwise distance matrix of the product molecule  
        '''
    
    prod = product.to_numpy()

    return np.dot(educt, prod[p,:][:,p].transpose()).diagonal().sum()


def update_best(gen, pop, scores, best, best_eval):
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

def mutation(p, reference, mutation_rate):
    '''Mutation function for genetic algorithm
    
    Args:
        p: (list) A candidate solution
        reference: (list) Reference list of possible mutations
        mutation_rate: (float) Mutation rate
    Returns:
        p: (list) Mutated candidate solution
    '''

    for i in range(len(p)):
        if random.random() < mutation_rate:
            mut = random.choice(reference[i])
            if mut in p:
                # Change the prior position of atom mut to the one at pos i and vice versa
                p[p.index(mut)] = p[i]
                p[i] = mut
            else:
                # Handle case where mut is not in p
                p[i] = mut
        # Ensure uniqueness
        while len(set(p)) != len(p):

            p = list(set(p))
            missing_indices = [i for i in range(len(reference)) if i not in p]
            for idx in missing_indices:
                p.append(random.choice(reference[idx]))
    return p


def tournament_selection(pop,scores,t_size=3):
    ''' turnament selection algorithm based on https://en.wikipedia.org/wiki/Tournament_selection
        param: 
            pop:        population vector
            scores:     scores vector
            t_size:     turnamnament size
    '''

    # randomly select t_size chromosomes/ individuals from the pop for the tunament
    sel_idx = random.choices(range(len(pop)),k = t_size)
    
    # get turnament population and their scores
    t_pop = list(np.array(pop)[sel_idx])
    t_scores = list(np.array(scores)[sel_idx])

    # get the set of t_scores and sort by highest score in the turnament selection
    sorted_scores = frozenset(t_scores)
    sorted_scores = sorted(sorted_scores, reverse=True)

    return list(t_pop[t_scores.index(sorted_scores[0])])

def crossover(p1, p2, r_cross):
    ''' Crossover implementation
        param: 
            p1:         parent 1
            p2:         parent 2
            r_cross:    crossover rate
    '''

    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    print(c1, c2)
    print("Crossover")
    # check for recombination
    if np.random.rand() < r_cross:

        # select crossover point that is not on the end of the string
        pt = np.random.randint(1, len(p1)-2)

        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]

    return [c1, c2]

def gen_reference(product_xyz: pd.DataFrame) -> list:
    '''Generates a reference list of possible mutations
    
    Args:
        n: (int) Number of atoms
    Returns:
        reference: (list) Reference list of possible mutations
    '''

    reference = {element: product_xyz.loc[product_xyz['Element'] == element].index.to_list() 
                for element in product_xyz['Element'].unique()}

    reference_map = []
    for element in product_xyz['Element']:
        indices = product_xyz.loc[product_xyz['Element'] == element].index
        if len(indices) > 0:
            reference_map.append(reference[element])
        else:
            reference_map.append([])  # or handle this case as needed
    
    reference = [value for key, value in reference.items()]

    return reference, reference_map

def visualize(result_dict):
    '''Visualize the result of the genetic algorithm
    
    Args:
        result_dict: (dict) Result of the genetic algorithm
    '''
    
    plt.rcParams.update({'font.size': 14})

    product_xyz = result_dict['product_xyz']
    mapped_educt = result_dict['mapped_educt']

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Product
    ax.scatter(product_xyz['x'], product_xyz['y'], product_xyz['z'], c=product_xyz['Element'].apply(lambda x: ord(x)), s=100)
    for i in range(len(product_xyz)):
        ax.text(product_xyz['x'][i], product_xyz['y'][i], product_xyz['z'][i], '%s' % (str(i)), size=20, zorder=1, color='k')
    # Educt
    ax.scatter(mapped_educt['x'], mapped_educt['y'], mapped_educt['z'], c=mapped_educt['Element'].apply(lambda x: ord(x)), s=100)
    for i in range(len(mapped_educt)):
        ax.text(mapped_educt['x'][i], mapped_educt['y'][i], mapped_educt['z'][i], '%s' % (str(i)), size=20, zorder=1, color='red', horizontalalignment='right')
    
    # Add a legend based on the text color
    ax.scatter([], [], [], c='k', label='Product')
    ax.scatter([], [], [], c='red', label='Mapped Educt')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Visualization of the Genetic Algorithm Result')
    plt.tight_layout()
    plt.show()

    