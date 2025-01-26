import pandas as pd
#===============================================================================================

def XYZ_reader(fname: str) -> pd.DataFrame:
    ''' A XYZ parser
    Args:
        fname :   (str)  full or relative file path of the xyz to be read
    Returns:
        xyz :     (pd.DataFrame)  DataFrame containing the xyz file data
    '''

    with open(fname, 'r') as ff:
        lines = ff.readlines()
    
    xyz = []
    for line in lines[2:]:
        if len(line.split()) == 4:
            xyz.append({
                'Element': str(line[:4].strip()),
                'x': float(line[4:20].strip()),
                'y': float(line[26:36].strip()),
                'z': float(line[40:67].strip())
            })
    xyz = pd.DataFrame(xyz)

    return xyz

def XYZ_writer(xyz: pd.DataFrame, filename: str) -> None:
    '''Write a DataFrame to an xyz file
    
    Args:
        xyz: (DataFrame) DataFrame to write to an xyz file
        filename: (str) Name of the file to write
    '''
    
    with open(filename, 'w') as f:
        f.write(f"{len(xyz)}\n")
        f.write("\n")
        for i in range(len(xyz)):
            f.write(f"{xyz['Element'][i]} {xyz['x'][i]} {xyz['y'][i]} {xyz['z'][i]}\n")
