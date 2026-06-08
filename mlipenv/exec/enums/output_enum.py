from enum import Enum

def _output_file_registry():
    return {
        "atoms": OutputFilesEnum.ATOMS,
        "coordinates": OutputFilesEnum.COORDINATES,
        "gradients": OutputFilesEnum.GRADIENTS,
        "energies": OutputFilesEnum.ENERGIES,
    }

class OutputFilesEnum(Enum):
    ATOMS = "atoms.npz"
    COORDINATES = "coordinates.npz"
    GRADIENTS = "gradients.npz"
    ENERGIES = "energies.npz"