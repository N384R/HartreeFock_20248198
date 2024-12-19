import os
import numpy as np


def load_basis_set(basis_set_name):
    '''Load basis set from a file and store it in a dictionary.'''
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, basis_set_name)
    basis_set_dict = {}
    with open(file_path, 'r', encoding="UTF-8") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip().startswith('#') or line.strip() == '':
                continue
            data = line.split()

            if data[0] == '!':
                element, charge = data[1], int(data[2])
                basis_set_dict[element] = {'charge': charge, 'basis': {}}
            elif data[0].isalpha():
                orbital_type = data[0]
                if orbital_type not in basis_set_dict[element]:
                    basis_set_dict[element]['basis'][orbital_type] = []
            else:
                data = np.array(data, float)
                basis_set_dict[element]['basis'][orbital_type].append(data)

    for element, data in basis_set_dict.items():
        for orbital_type, values in data['basis'].items():
            data['basis'][orbital_type] = np.vstack(values)
    return basis_set_dict
