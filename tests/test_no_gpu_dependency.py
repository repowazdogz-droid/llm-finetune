"""Floor B0 gate, part 3: this repo cannot load a model or touch a GPU.

B0's job is to hand the PC phase a dataset and a harness. It is explicitly not
allowed to load model weights or require CUDA, and "we didn't do that" is a
promise. These tests make it a property of the source tree.

The check is an AST walk over imports rather than a text grep, for the same
reason Project A's test-set-leak detector is: prose that *mentions* torch is not
an import of torch, and a detector that cannot tell the difference will fire on
its own documentation.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Packages that would mean this floor had started doing the PC phase's job.
FORBIDDEN_IMPORTS = {
    "torch", "torchvision", "torchaudio",
    "transformers", "accelerate", "peft", "trl", "bitsandbytes",
    "vllm", "llama_cpp", "ctransformers", "onnxruntime",
    "deepspeed", "xformers", "flash_attn",
    "cupy", "pynvml", "numba",
}

SKIP_PREFIXES = (".venv/", "data/raw/", "data/processed/", "runs/")


def _python_files() -> list[Path]:
    return [
        p for p in REPO.rglob("*.py")
        if not p.relative_to(REPO).as_posix().startswith(SKIP_PREFIXES)
    ]


def _imported_roots(source: str) -> set[str]:
    """Root package name of every import in a module."""
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
    return roots


def test_no_module_imports_a_gpu_or_model_library():
    offenders = {}
    for path in _python_files():
        hits = _imported_roots(path.read_text()) & FORBIDDEN_IMPORTS
        if hits:
            offenders[path.relative_to(REPO).as_posix()] = sorted(hits)
    assert not offenders, (
        f"Floor B0 must not depend on model or GPU libraries, but: {offenders}"
    )


def test_the_import_detector_is_not_vacuous():
    """Negative control: the scan must actually catch a forbidden import."""
    assert _imported_roots("import torch") & FORBIDDEN_IMPORTS == {"torch"}
    assert _imported_roots("from transformers import AutoModel") & FORBIDDEN_IMPORTS == {
        "transformers"
    }
    assert _imported_roots("import torch.nn as nn") & FORBIDDEN_IMPORTS == {"torch"}
    # Prose mentioning torch is not an import of torch.
    assert not _imported_roots('"""We deliberately do not import torch."""') & FORBIDDEN_IMPORTS


def test_requirements_pins_nothing_gpu_shaped():
    text = (REPO / "requirements.txt").read_text()
    pinned = {
        line.split("==")[0].strip().lower()
        for line in text.splitlines()
        if "==" in line and not line.strip().startswith("#")
    }
    assert not (pinned & FORBIDDEN_IMPORTS), f"GPU packages pinned: {pinned & FORBIDDEN_IMPORTS}"


def test_every_requirement_is_pinned_exactly():
    """No floating pins — the same hygiene rule as Project A."""
    unpinned = []
    for line in (REPO / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            unpinned.append(line)
    assert not unpinned, f"unpinned requirements: {unpinned}"


def test_harness_contract_needs_no_inference_library():
    """The harness must be satisfiable by a plain callable — the whole design claim."""
    from eval.harness import run_eval

    def toy_generate(prompts):
        return ["#### 7"] * len(prompts)

    r = run_eval(
        toy_generate,
        [{"question": "q", "answer": "7", "n_steps": 1}],
        run_name="contract_check",
    )
    assert r["accuracy"] == 1.0
