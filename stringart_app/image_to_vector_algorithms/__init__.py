# stringart_app/image_to_vector_algorithms/__init__.py

import pkgutil
import importlib
import inspect
from typing import Dict, Type, Callable, Optional
import numpy as np

from .base import StringArtAlgorithm

ALGORITHMS: Dict[str, StringArtAlgorithm] = {}

# --- Auto-discover all modules in this package ---
package_name = __name__  # "stringart_app.image_to_vector_algorithms"
package_path = __path__  # filesystem path to this directory

for finder, module_name, is_pkg in pkgutil.iter_modules(package_path):
    if module_name in ("base", "__init__"):
        continue  # skip base and this init
    full_name = f"{package_name}.{module_name}"
    module = importlib.import_module(full_name)

    # Find all concrete subclasses of StringArtAlgorithm in the module
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(cls, StringArtAlgorithm)
            and cls is not StringArtAlgorithm
            and not inspect.isabstract(cls)
        ):
            # Derive the registry key from module_name.
            # e.g. module coverage → key "coverage"
            key = module_name.replace("_", "-")
            ALGORITHMS[key] = cls()

def generate_string_vectors(
    pixels: np.ndarray,
    n_anchors: int = 180,
    n_strings: int = 200,
    line_thickness: int = 1,
    sample_pairs: int = 1000,
    algorithm: str = "greedy",
    *,
    vector_callback: Optional[Callable[[int, int], None]] = None,
) -> list[dict[str, int]]:
    """
    Dispatch to whichever StringArtAlgorithm you’ve registered.
    Optionally stream each vector via vector_callback(from_idx, to_idx).
    """
    algo = ALGORITHMS.get(algorithm)
    if algo is None:
        valid = ", ".join(ALGORITHMS.keys())
        raise ValueError(f"Unknown algorithm '{algorithm}'. Valid options: {valid}")
    return algo.generate(
        pixels,
        n_anchors=n_anchors,
        n_strings=n_strings,
        line_thickness=line_thickness,
        sample_pairs=sample_pairs,
        vector_callback=vector_callback,
    )
