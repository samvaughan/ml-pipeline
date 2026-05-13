import importlib
import pkgutil
from types import ModuleType
from typing import Iterable


def _iter_modules_recursive(pkg: ModuleType) -> Iterable[str]:
    """Yield fully-qualified module names under a package recursively."""
    for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        yield mod.name


def autodiscover_plugins():
    """Import all modules under ltv.plugins (transformers, models, evaluators, etc.)."""
    import ltv.plugins as base_pkg

    for modname in _iter_modules_recursive(base_pkg):
        importlib.import_module(modname)


def autodiscover_steps():
    """Import all steps under ltv.steps."""
    import ltv.steps as base_pkg

    for modname in _iter_modules_recursive(base_pkg):
        importlib.import_module(modname)
