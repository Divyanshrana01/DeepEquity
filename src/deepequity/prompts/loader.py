from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from deepequity.core.logging import get_logger

logger = get_logger("deepequity.prompts")

_LIBRARY = Path(__file__).parent / "library"


#a prompt plus which version of it we used.
#
#the version travelling with the text is the whole point. phase 6 is a series of
#experiments where one prompt changes and the eval suite re-runs, and a score is
#meaningless unless you know which wording produced it. carrying the version into the
#trace means a run can always be traced back to the exact prompt behind it.
@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    text: str

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"


#loads a prompt from the library by name and version.
#
#these live as files rather than string literals on purpose. a prompt sitting in the
#middle of a function gets tweaked casually, and then nobody can say what changed
#between two eval runs. as files they diff, they review, and changing one is a visible
#act rather than an edit buried in a code change.
@lru_cache
def load(name: str, version: str = "v1") -> Prompt:
    path = _LIBRARY / f"{name}_{version}.md"
    if not path.exists():
        available = sorted(p.stem for p in _LIBRARY.glob("*.md"))
        raise FileNotFoundError(
            f"no prompt {name}_{version}.md in the library. available: {available}"
        )
    return Prompt(name=name, version=version, text=path.read_text().strip())


#every prompt we have, used by a test that checks they all load and none has been left
#half written
def list_prompts() -> list[str]:
    return sorted(path.stem for path in _LIBRARY.glob("*.md"))
