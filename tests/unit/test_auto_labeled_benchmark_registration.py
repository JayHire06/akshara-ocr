from __future__ import annotations

import importlib


def test_auto_labeled_suite_is_registered():
    mod = importlib.import_module("model.benchmark_eval")
    found = False
    for name in dir(mod):
        val = getattr(mod, name)
        if isinstance(val, (list, dict, tuple)):
            if "auto_labeled_verified_printed" in str(val):
                found = True
                break
    assert found, "auto_labeled_verified_printed suite not registered in model/benchmark_eval.py"


def test_auto_labeled_stage_is_registered_in_v9_curriculum():
    mod = importlib.import_module("scripts.training.train_v9")
    found = False
    for name in dir(mod):
        val = getattr(mod, name)
        if isinstance(val, (list, dict, tuple)):
            if "stage_auto_labeled_printed" in str(val):
                found = True
                break
    assert found, "stage_auto_labeled_printed not registered in scripts/training/train_v9.py"
