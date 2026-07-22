#!/usr/bin/env python3
"""Run the published global-probing entry point with fold-isolated callbacks.

``main_global_probing.py`` creates one callback list and passes it to every
cross-validation Trainer. Lightning callbacks are stateful, so this launcher
gives each Trainer a deep copy of that pristine list. The published source file
is imported and executed unchanged.
"""

from __future__ import annotations

import argparse
import copy
import runpy
import sys
from pathlib import Path
from typing import Any, Type


def fold_isolated_trainer(base_trainer: Type[Any]) -> Type[Any]:
    """Return a Trainer subclass that copies callbacks before attaching them.

    The original callback instances remain untouched. Consequently, every new
    Trainer starts with fresh EarlyStopping, ModelCheckpoint, and learning-rate
    monitor state even when the caller reuses the same callback list.
    """

    class FoldIsolatedTrainer(base_trainer):
        trainer_count = 0

        def __init__(self, *args: Any, callbacks: Any = None, **kwargs: Any) -> None:
            type(self).trainer_count += 1
            isolated_callbacks = copy.deepcopy(callbacks)
            print(
                "[gLocal wrapper] Starting Trainer "
                f"{type(self).trainer_count} with fresh callback state.",
                flush=True,
            )
            super().__init__(
                *args,
                callbacks=isolated_callbacks,
                **kwargs,
            )

    FoldIsolatedTrainer.__name__ = "FoldIsolatedTrainer"
    return FoldIsolatedTrainer


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run main_global_probing.py without sharing callback state"
    )
    parser.add_argument("--repo-root", required=True, type=Path)
    return parser.parse_known_args()


def main() -> int:
    args, probing_args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    entrypoint = repo_root / "main_global_probing.py"
    if not entrypoint.is_file():
        raise SystemExit(f"Published entry point not found: {entrypoint}")

    sys.path.insert(0, str(repo_root))
    import pytorch_lightning  # Imported after the repository path is configured.

    pytorch_lightning.Trainer = fold_isolated_trainer(
        pytorch_lightning.Trainer
    )
    sys.argv = [str(entrypoint), *probing_args]
    runpy.run_path(str(entrypoint), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
