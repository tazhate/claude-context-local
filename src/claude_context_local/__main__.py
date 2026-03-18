"""Entry point for `python -m claude_context_local` and `uvx claude-context-local`."""

import json
import os
import sys
import threading


def _patch_stdin():
    """Patch stdin to strip empty "params":{} from JSON-RPC messages.

    mcp SDK 1.x model_validate() rejects "params":{} for methods that
    don't declare parameters (tools/list, notifications/initialized, etc).
    This is technically valid JSON-RPC 2.0, but the SDK's Pydantic models
    treat it as a validation error (-32602). Strip it before the SDK sees it.
    """
    real_stdin = sys.stdin.buffer
    read_fd, write_fd = os.pipe()

    def _filter():
        for line in real_stdin:
            try:
                msg = json.loads(line)
                if isinstance(msg.get("params"), dict) and len(msg["params"]) == 0:
                    del msg["params"]
                os.write(write_fd, json.dumps(msg).encode() + b"\n")
            except (json.JSONDecodeError, TypeError):
                os.write(write_fd, line)
        os.close(write_fd)

    t = threading.Thread(target=_filter, daemon=True)
    t.start()
    sys.stdin = open(read_fd, "r")


def main():
    _patch_stdin()
    from claude_context_local.server import mcp
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
