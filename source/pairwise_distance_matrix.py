
from scipy.spatial.distance import cdist
import pandas as pd
from numpy import inf

#===============================================================================================

def pairwise_distance_matrix(molecule,inverse = False, unit = False, normalize = False, exp = None):
    '''Compute the pairwise distance matrix of a molecule (dataframe object) with x,y,z columns
    param:
        molecule:   dataframe containing at the least 3 col wit 'x', 'y', 'z' 
        inverse:    compute inverse of pairwise distance matrix (default: False) 
        exp:        compute inverse with exponent for faster decay (default: None)  
        unit:       normalize columns to unit vectors (default: False)
        normalize:  mean normalize matrix (default: False)
    '''
    
    # compute pairwise distance matrix
    df_dist = pd.DataFrame(cdist(molecule[['x','y','z']],molecule[['x','y','z']]))

    # if requested compute inverse matrix with or without exponennt
    if inverse == True and not exp:
        df_dist = (1 / df_dist).replace(inf, 0)
        
    elif inverse == True and exp:
        df_dist = (1 / df_dist).replace(inf, 0) ** exp

    # if requested normalize columns to unit vector (v / sqrt( sum(v_i ** 2) ) 
    if unit == True :
        df_dist = (df_dist / ((df_dist**2).sum().pow(1./2)))
    
    # if requested mean normalize columns 
    if normalize == True :
        df_dist =  (df_dist - df_dist.mean()) / df_dist.std()
    
    return df_dist