"""Plain-text report rendering.

Every artefact this release renders goes to ``.txt``: the release's own hygiene
rule keeps a single ``.md`` in the tree and a report written to markdown would
break it the moment the driver ran.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from saber.runtime.atomic import write_text


def render_table(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
    title: str | None = None,
) -> str:
    """A fixed-width table whose column widths come from the data.

    Widths are derived rather than fixed: a fixed width silently truncates the
    long values, and a truncated value in a report is indistinguishable from a
    short one.
    """
    if not rows:
        return f"{title or 'table'}: no rows\n"
    widths = {
        column: max(
            len(column),
            max(len(_render(row.get(column))) for row in rows),
        )
        for column in columns
    }
    header = "  ".join(_pad(column, widths[column]) for column in columns)
    rule = "  ".join("-" * widths[column] for column in columns)
    lines = [f"{title}:" if title else "", header, rule]
    for row in rows:
        lines.append(
            "  ".join(_pad(_render(row.get(column)), widths[column]) for column in columns)
        )
    return "\n".join(line for line in lines if line != "") + "\n"


def render_key_values(values: dict[str, object]) -> str:
    width = max((len(key) for key in values), default=0)
    return (
        "\n".join(f"{key.ljust(width)} : {_render(value)}" for key, value in values.items()) + "\n"
    )


def write_report(path: Path, text: str) -> None:
    write_text(path, text if text.endswith("\n") else text + "\n")


def _render(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_render(item) for item in value) + "]"
    return str(value)


def _pad(text: str, width: int) -> str:
    return text.ljust(width)
