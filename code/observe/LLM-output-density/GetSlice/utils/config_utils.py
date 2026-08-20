import json
from typing import Iterable


def _strip_json_line_comments(text: str) -> str:
    """
    Remove // comments outside of JSON strings.
    Supports full-line and inline comments.
    """
    output_lines = []

    for line in text.splitlines():
        new_line_chars = []
        in_string = False
        escape = False
        i = 0

        while i < len(line):
            ch = line[i]

            if in_string:
                new_line_chars.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue

            if ch == '"':
                in_string = True
                new_line_chars.append(ch)
                i += 1
                continue

            if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                break

            new_line_chars.append(ch)
            i += 1

        output_lines.append("".join(new_line_chars))

    return "\n".join(output_lines)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        text = f.read()

    stripped = _strip_json_line_comments(text)
    return json.loads(stripped)


def validate_config_keys(
    cfg: dict,
    required_keys: Iterable[str],
    optional_keys: Iterable[str],
):
    required = set(required_keys)
    optional = set(optional_keys)

    missing = sorted([k for k in required if k not in cfg])
    unknown = sorted(set(cfg.keys()) - required - optional)
    return missing, unknown
