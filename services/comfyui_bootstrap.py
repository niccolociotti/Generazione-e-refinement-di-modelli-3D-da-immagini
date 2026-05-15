import builtins
import os
import sys
from contextlib import contextmanager
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_CANDIDATES = (
    "ComfyUI",
    "comfyui",
    "ComfyUi",
    "comfui",
)


def find_comfyui_path():
    configured_path = os.getenv("COMFYUI_PATH")
    if configured_path:
        path = Path(configured_path).expanduser().resolve()
        if path.exists():
            return path

    for folder_name in COMFYUI_CANDIDATES:
        path = PROJECT_ROOT / folder_name
        if (path / "nodes.py").exists():
            return path

    return None


def add_comfyui_to_path():
    comfyui_path = find_comfyui_path()
    if not comfyui_path:
        return None

    path_value = str(comfyui_path)
    if path_value in sys.path:
        sys.path.remove(path_value)
    sys.path.insert(1, path_value)
    return comfyui_path


def patch_torch_for_comfyui(torch_module):
    if not hasattr(torch_module.serialization, "add_safe_globals"):
        torch_module.serialization.add_safe_globals = lambda safe_globals: None

    unsigned_aliases = {
        "uint16": "int16",
        "uint32": "int32",
        "uint64": "int64",
    }
    for missing_name, fallback_name in unsigned_aliases.items():
        if not hasattr(torch_module, missing_name):
            setattr(torch_module, missing_name, getattr(torch_module, fallback_name))

    mps_available = (
        hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    )
    cuda_available = torch_module.cuda.is_available()
    if not cuda_available and not mps_available and "--cpu" not in sys.argv:
        sys.argv.append("--cpu")

    try:
        from comfy import options

        options.enable_args_parsing(True)
    except ImportError:
        pass


def comfyui_not_found_message():
    candidates = ", ".join(COMFYUI_CANDIDATES)
    return (
        "ComfyUI non trovato. Metti la cartella ComfyUI nella root del progetto "
        f"({PROJECT_ROOT}) con uno di questi nomi: {candidates}; oppure imposta "
        "COMFYUI_PATH con il path della cartella ComfyUI."
    )


@contextmanager
def block_optional_imports(*module_names):
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        for module_name in module_names:
            if name == module_name or name.startswith(f"{module_name}."):
                raise ImportError(
                    f"{module_name} disabilitato per compatibilita con questo ambiente"
                )
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    try:
        yield
    finally:
        builtins.__import__ = original_import
