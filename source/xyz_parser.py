
import pandas as pd
#===============================================================================================

def XYZ_reader(fname):
    ''' A XYZ parser
    param:
        fname :     full or relative file path of the xyz to be read
    '''

    # open the psf file
    with open(fname, 'r') as ff:
        lines = ff.readlines()
    
    # initiate psf file as list
    xyz = []

    #iterate over all lines
    for line in lines[2:]: 

        # read only lines actually containing all information specified in this parser
        if len(line.split()) == 4:
            
            xyz.append({'Element': str(line[:4].strip()),
                        'x': float(line[4:26].strip()),
                        'y': float(line[26:46].strip()),
                        'z': float(line[46:67].strip())})


    # transfer psf to pd Dataframe with atom number as index
    # this conveniently gives the Dict keys as column names 
    xyz = pd.DataFrame(xyz)
    
    return xyz