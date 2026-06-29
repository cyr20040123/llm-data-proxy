#!/usr/bin/env python3
"""Strip system messages and non-essential fields from a ChatML JSON file."""

import argparse
import json
import sys


KEEP_FIELDS = {"role", "content", "tool_calls", "tool_call_id", "name",
               "reasoning_content", "reasoning"}


def strip_chatml(data: dict, keep_system: bool = False) -> list:
    """Strip non-essential fields and optionally system messages from ChatML data.

    Args:
        data: Parsed ChatML JSON object (with a "messages" key).
        keep_system: If True, retain system messages.

    Returns:
        List of cleaned message dicts.
    """
    out = []
    for m in data.get("messages", []):
        if m.get("role") == "system" and not keep_system:
            continue
        out.append({k: v for k, v in m.items()
                    if k in KEEP_FIELDS and v is not None})
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Strip system messages and non-essential fields from ChatML JSON"
    )
    parser.add_argument("input", nargs="?", default=None,
                        help="Input ChatML JSON file (default: stdin)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file (default: stdout)")
    parser.add_argument("--keep-system", action="store_true",
                        help="Retain system messages")
    args = parser.parse_args()

    if args.input:
        with open(args.input) as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    cleaned = strip_chatml(data, keep_system=args.keep_system)
    output = json.dumps(cleaned, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
            f.write("\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
