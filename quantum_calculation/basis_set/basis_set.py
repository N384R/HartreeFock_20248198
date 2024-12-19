'''
Basis set module to load basis set from a file and store it in a dictionary.
'''

import importlib
from . import get_module_path


class BasisSet:
    '''
    Class to load basis set from a file and store it in a dictionary.

    Attributes:
    basis_set_name (str): Name of the basis set.
    basis_set_file (str): File containing the basis set.
    basis_set (dict): Dictionary containing the basis set.
    '''

    def __init__(self, basis_set_name):
        self.basis_set_name = basis_set_name.upper()
        self.basis_set = self._load_basis_set()

    def __getitem__(self, element):
        return self.basis_set[element]

    def show_basis_set(self):
        '''Print the basis set.'''
        print(f"Basis set: {self.basis_set_name}")
        for element, info in self.basis_set.items():
            print(f"{element} (Charge: {info['charge']}): {info['basis']}")

    def _load_basis_set(self):
        '''Load the basis set from the file.'''
        module_path = get_module_path(self.basis_set_name)
        basis_set_module = importlib.import_module(module_path)
        return basis_set_module.load_basis_set(self.basis_set_name)

    def __str__(self):
        return f"Basis set: {self.basis_set_name}"


if __name__ == "__main__":
    basis_set = BasisSet("6-31G")
    basis_set.show_basis_set()
    print(basis_set['H'])
