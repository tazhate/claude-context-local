# claude-context-local

Local semantic code search MCP server for [Claude Code](https://claude.ai/claude-code). Zero external APIs — runs entirely on your machine.

A lightweight alternative to [zilliztech/claude-context](https://github.com/zilliztech/claude-context) that uses local embeddings instead of OpenAI + Zilliz Cloud.

## How it works

- **[sentence-transformers](https://www.sbert.net/)** (`all-MiniLM-L6-v2`, 88 MB) for embeddings — no API keys needed
- **[ChromaDB](https://www.trychroma.com/)** for persistent vector storage — per-project isolation
- **Incremental indexing** — only re-indexes changed files (MD5 hash check)
- **MCP protocol** over stdio — plug into Claude Code or any MCP client

## Install

```bash
git clone https://github.com/tazhate/claude-context-local.git
cd claude-context-local
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Add to Claude Code

```bash
claude mcp add claude-context-local -- /path/to/claude-context-local/.venv/bin/python /path/to/claude-context-local/server.py
```

Or manually in `~/.claude.json`:

```json
{
  "mcpServers": {
    "claude-context-local": {
      "command": "/path/to/claude-context-local/.venv/bin/python",
      "args": ["/path/to/claude-context-local/server.py"]
    }
  }
}
```

Restart Claude Code after adding.

## MCP Tools

| Tool | Description |
|---|---|
| `index_project(project_path)` | Index a codebase. Incremental by default, `force=True` to rebuild. |
| `search_code(query, project_path)` | Semantic search. Optional `n_results` and `file_filter` (e.g. `"*.py"`). |
| `index_status(project_path)` | Show indexed files/chunks count. |
| `drop_index(project_path)` | Remove project index. |

## Usage

Once connected, Claude Code will automatically use these tools. You can also ask directly:

- *"Index this project"* — triggers `index_project` with current working directory
- *"Search for authentication logic"* — triggers `search_code`
- *"How many files are indexed?"* — triggers `index_status`

## Configuration

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `CCL_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model name |
| `CCL_CHUNK_LINES` | `50` | Max lines per chunk |
| `CCL_CHUNK_OVERLAP` | `10` | Overlap lines between chunks |
| `CCL_DATA_DIR` | `~/.cache/claude-context-local` | Index storage directory |

## Resource usage

| Resource | Value |
|---|---|
| RAM | ~780 MB (PyTorch + model in memory) |
| Model on disk | 88 MB |
| Index size | ~27 MB per 500 files |
| CPU | Near zero at idle |

## Per-project isolation

Each project gets its own ChromaDB database under `~/.cache/claude-context-local/<hash>/`, where `<hash>` is derived from the absolute project path. Projects never mix.

## Supported file types

Code: `.py` `.go` `.js` `.ts` `.tsx` `.jsx` `.rs` `.java` `.kt` `.c` `.cpp` `.h` `.hpp` `.cs` `.rb` `.php` `.swift` `.scala` `.sh` `.bash` `.lua` `.zig` `.nim` `.ex` `.exs` `.erl` `.nix`

Config: `.yaml` `.yml` `.toml` `.json` `.hcl` `.tf` `.sql` `.graphql` `.proto`

Docs: `.md` `.txt` `.rst`

Web: `.html` `.css` `.scss` `.less`

Plus `Dockerfile` and `Makefile`.

## License

MIT
