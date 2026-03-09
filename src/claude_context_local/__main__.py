"""Entry point for `python -m claude_context_local` and `uvx claude-context-local`."""

from claude_context_local.server import mcp


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
