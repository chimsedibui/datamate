#!/usr/bin/env python3
"""Collect validated training-split loaders into src/datamate/semantic/loader_examples.json (few-shot examples for builds).

Usage:
    python scripts/release/build_loader_library.py --work work/train --split data/train

For every business whose cached build is ok, stores its loader code, industry, raw column signature and data hash.
Provenance for the organizers (API-generated artifacts, guide section 8) is copied to
training_code/loader_examples/<biz>/ (the build's prompts/responses and log) and listed with SHA-256 checksums in
training_code/loader_examples/manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datamate.semantic.loader_examples import LIBRARY, column_signature, data_hash  # noqa: E402
from datamate.core.rawio import RawStore  # noqa: E402


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work/train")
    ap.add_argument("--split", default="data/train")
    ap.add_argument("--max-code-chars", type=int, default=12000, help="skip very long loaders (prompt budget)")
    a = ap.parse_args()

    work, split = Path(a.work), Path(a.split)
    prov = ROOT / "training_code" / "loader_examples"
    if prov.exists():
        shutil.rmtree(prov)
    prov.mkdir(parents=True)
    lib, files = [], []
    for d in sorted(p for p in work.iterdir() if (p / "meta.json").exists()):
        meta = json.loads((d / "meta.json").read_text())
        if not meta.get("ok") or not (d / "loader.py").exists():
            continue
        code = (d / "loader.py").read_text()
        if len(code) > a.max_code_chars:
            print(f"{d.name}: skipped ({len(code)} chars)")
            continue
        biz_dir = split / "data" / d.name
        lib.append({"source": d.name, "industry": meta.get("industry") or "", "tables": meta.get("tables", []),
                    "columns": column_signature(RawStore(str(biz_dir))), "data_hash": data_hash(biz_dir),
                    "build_model": meta.get("build_model", "gpt-4o or gpt-5.4-mini (see messages.json)"),
                    "code": code})
        out = prov / d.name
        out.mkdir()
        for name in ("loader.py", "messages.json", "build_log.json", "report.txt"):
            if (d / name).exists():
                shutil.copy2(d / name, out / name)
                files.append({"path": f"training_code/loader_examples/{d.name}/{name}",
                              "bytes": (out / name).stat().st_size, "sha256": sha256(out / name)})
        print(f"{d.name}: {meta.get('industry')} {len(code)} chars, {len(lib[-1]['columns'])} columns")
    LIBRARY.write_text(json.dumps(lib, ensure_ascii=False, indent=1))
    (prov / "manifest.json").write_text(json.dumps({
        "description": "Validated semantic-layer loaders generated during development for the provided training "
                       "businesses; shipped as few-shot examples in src/datamate/semantic/loader_examples.json",
        "source_split": "train (provided training data only; no public/private test data)",
        "library": {"path": "src/datamate/semantic/loader_examples.json", "sha256": sha256(LIBRARY), "examples": len(lib)},
        "files": files}, ensure_ascii=False, indent=1))
    print(f"{len(lib)} examples -> {LIBRARY.relative_to(ROOT)}; provenance -> {prov.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
