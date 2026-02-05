from __future__ import annotations

from scripts.budget import ResourceBudget
from scripts.methods_registry import get_method


def main() -> int:
    m = get_method("uniform", {"fps": 1.0}, ResourceBudget())
    assert m.name == "uniform"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
