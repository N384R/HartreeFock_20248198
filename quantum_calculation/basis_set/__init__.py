import os


BASIS_SET_DATABASE = {
    "slater_type": [
        "STO-3G",
    ],
    "split_valence": [
        "6-31G",
    ],
}


def get_module_path(name):
    '''Return the path to the load_basis_set module.'''
    package = "quantum_calculation.basis_set"
    name = name.upper()
    for category, sets in BASIS_SET_DATABASE.items():
        if name in sets:
            module_path = f"{package}.{category}.load_basis_set"
            return module_path
    raise ValueError(f"Basis set {name} not found.")
