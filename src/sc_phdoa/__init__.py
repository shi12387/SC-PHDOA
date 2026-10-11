"""SC-PHDOA model and deterministic inference implementation.

Heavy Torch modules are imported lazily so protocol and data-validation tools
remain usable before the CUDA runtime is initialized.
"""

from importlib import import_module

__all__ = [
    "CheckpointPaths",
    "SCPHDOAModel",
    "SessionCalibrationNetwork",
    "SourceCountEstimator",
    "load_components",
]


def __getattr__(name: str):
    locations = {
        "CheckpointPaths": (".checkpoints", "CheckpointPaths"),
        "SCPHDOAModel": (".model", "SCPHDOAModel"),
        "SessionCalibrationNetwork": (
            ".calibration", "SessionCalibrationNetwork"
        ),
        "SourceCountEstimator": (".source_count", "SourceCountEstimator"),
        "load_components": (".checkpoints", "load_components"),
    }
    if name not in locations:
        raise AttributeError(name)
    module_name, attribute = locations[name]
    return getattr(import_module(module_name, __name__), attribute)
