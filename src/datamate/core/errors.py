"""Tracebacks of generated code, shaped for sending back to the LLM."""
from __future__ import annotations

import os
import sys
import traceback


def short_traceback(tag: str, extra: int = 2, max_chars: int = 2500) -> str:
    """Traceback of the exception being handled, starting at the generated code.

    `tag` is the file name the generated code was compiled with ("<solve>", "<loader>"). Its frames are kept with
    their line numbers; of the frames below it (our helpers, pandas) the first `extra` are kept as
    `File "name.py", in function` plus the source line. Frames above the generated code, directories and library
    line numbers are left out: the text must not depend on where the package is installed or on unrelated edits,
    because it becomes part of a prompt (and of the LLM cache key).
    """
    et, ev, tb = sys.exc_info()
    frames = traceback.extract_tb(tb)
    start = next((i for i, f in enumerate(frames) if f.filename == tag), len(frames))
    lines, left = [], extra
    for f in frames[start:]:
        if f.filename == tag:
            lines.append(f'  File "{tag}", line {f.lineno}, in {f.name}\n')
        elif left > 0 and f.name != "<lambda>":
            left -= 1
            lines.append(f'  File "{os.path.basename(f.filename)}", in {f.name}\n'
                         + (f"    {f.line}\n" if f.line else ""))
    head = ["Traceback (most recent call last):\n"] if lines else []
    return "".join(head + lines + traceback.format_exception_only(et, ev))[-max_chars:]
