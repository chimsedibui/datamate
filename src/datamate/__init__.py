"""DataMate: offline agent answering Vietnamese business questions over shop data."""
from .core import pdcompat as _pdcompat

_pdcompat.install()  # pandas 1.x/2.x calls in LLM-written code keep working under pandas 3
