"""Build the semantic layer (canonical tables) for every business of a split.

Usage:
    python src/build_semantic.py --split data/train --work work/train            # all businesses
    python src/build_semantic.py --split data/train --work work/train --biz biz_111 --force
    python src/build_semantic.py --split data/train --work work/train --retry-failed   # after improving the mapper
    python src/build_semantic.py --split data/train --work work/train --refresh        # re-run cached loaders, no LLM
    python src/build_semantic.py --split data/train --work work/train --biz biz_111 --loader my_loader.py
        (--loader runs a hand-written loader instead of calling the LLM; useful for testing —
         see src/datamate/semantic/examples/loader_biz_111.py)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datamate.core.rawio import RawStore  # noqa: E402
from datamate.semantic.validate import finalize, validate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--biz", nargs="*")
    ap.add_argument("--force", action="store_true", help="rebuild even if the fingerprint matches")
    ap.add_argument("--refresh", action="store_true",
                    help="re-run cached loaders with current parsers/validation, no LLM calls")
    ap.add_argument("--retry-failed", action="store_true", help="rebuild businesses whose cached build failed")
    ap.add_argument("--loader")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()

    split = Path(a.split)
    bizs = a.biz or sorted(os.listdir(split / "data"))
    rulebook = (split / "rulebook.md").read_text()

    if a.refresh:
        from datamate.semantic.mapper import refresh
        for b in bizs:
            lp = Path(a.work) / b / "loader.py"
            if not lp.exists():
                continue
            ok, errs = refresh(a.work, b, str(split / "data" / b))
            print(f"{b}: {'ok' if ok else 'INVALID -> raw fallback'}" + "".join(f"\n   {e[:200]}" for e in errs))
        return

    if a.loader:
        from datamate.semantic.mapper import run_loader
        for b in bizs:
            tables, meta = run_loader(Path(a.loader).read_text(), RawStore(str(split / "data" / b)))
            tables = finalize(tables, meta)
            errs, rep = validate(tables, meta)
            print(rep)
            print("ERRORS:", errs or "none")
        return

    from datamate.core.llm import LLM
    from datamate.semantic.mapper import Mapper
    mapper = Mapper(LLM.for_build(), rulebook, a.work)

    def one(b):
        t = time.time()
        try:
            m = mapper.build(b, str(split / "data" / b), force=a.force, retry_failed=a.retry_failed)
        except Exception as e:  # noqa: BLE001
            m = {"ok": False, "error": repr(e)}
        print(f"{b}: ok={m.get('ok')} industry={m.get('industry')} tables={m.get('tables')} "
              f"{time.time() - t:.0f}s", flush=True)
        return b, m

    with ThreadPoolExecutor(a.jobs) as ex:
        results = dict(ex.map(one, bizs))
    Path(a.work).mkdir(parents=True, exist_ok=True)
    (Path(a.work) / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str))
    from datamate.core.llm import spent_usd
    print(f"{sum(m.get('ok', False) for m in results.values())}/{len(results)} businesses mapped, "
          f"est. spend ${spent_usd():.2f}")


if __name__ == "__main__":
    main()
