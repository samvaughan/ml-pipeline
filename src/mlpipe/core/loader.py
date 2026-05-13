import importlib
import pkgutil
from types import ModuleType
from typing import Iterable, Union


def _iter_modules_recursive(pkg: ModuleType) -> Iterable[str]:
    """Yield fully-qualified module names under a package recursively."""
    for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        yield mod.name


def autodiscover(package: Union[str, ModuleType]) -> None:
    """Import all modules under `package`, triggering any @register decorators."""
    if isinstance(package, str):
        package = importlib.import_module(package)
    for modname in _iter_modules_recursive(package):
        importlib.import_module(modname)
