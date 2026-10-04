"""
Config loading utilities.

Design rule (per ARCHITECTURE.md): no hardcoded absolute paths anywhere in
src/. All paths resolve from an environment config file + environment
variables, selected explicitly by the caller (never auto-detected by
guessing the OS), so behavior is predictable on Windows, Colab, or CI.

Usage:
    from src.utils.config import load_config
    cfg = load_config(env="windows")   # or env="colab"
    cfg["data_root"]  # fully resolved, env-vars expanded
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


_DEFAULT_FALLBACKS = {
    "FOOD_PROJECT_DATA_ROOT": lambda: str(Path(__file__).resolve().parents[2] / "data"),
    "FOOD_PROJECT_MODELS_ROOT": lambda: str(Path(__file__).resolve().parents[2] / "models"),
    "FOOD_PROJECT_OUTPUTS_ROOT": lambda: str(Path(__file__).resolve().parents[2] / "outputs"),
}


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ${VAR_NAME} placeholders using os.environ with local defaults."""
    if isinstance(value, str):
        def _sub(match: re.Match) -> str:
            var_name = match.group(1)
            if var_name in os.environ:
                return os.environ[var_name]
            if var_name in _DEFAULT_FALLBACKS:
                return _DEFAULT_FALLBACKS[var_name]()
            raise EnvironmentError(
                f"Config references ${{{var_name}}} but that environment "
                f"variable is not set. Set it before loading this config."
            )
        return _ENV_VAR_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(env: str, config_dir: Path | None = None) -> dict:
    """
    Load configs/base.yaml merged with configs/env.<env>.yaml, expanding
    ${ENV_VAR} placeholders. Raises EnvironmentError with a clear message if
    a required environment variable isn't set, rather than silently
    resolving to a default path.
    """
    config_dir = config_dir or _CONFIG_DIR
    base_path = config_dir / "base.yaml"
    env_path = config_dir / f"env.{env}.yaml"

    if not base_path.exists():
        raise FileNotFoundError(f"Missing base config: {base_path}")
    if not env_path.exists():
        raise FileNotFoundError(
            f"Missing env config: {env_path} "
            f"(expected one of the configs/env.*.yaml files)"
        )

    with open(base_path, "r") as f:
        base_cfg = yaml.safe_load(f) or {}
    with open(env_path, "r") as f:
        env_cfg = yaml.safe_load(f) or {}

    merged = _deep_merge(base_cfg, env_cfg)
    return _expand_env_vars(merged)
