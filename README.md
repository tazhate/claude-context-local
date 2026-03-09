# claude-context-local

[![PyPI](https://img.shields.io/pypi/v/claude-context-local)](https://pypi.org/project/claude-context-local/)
[![Python](https://img.shields.io/pypi/pyversions/claude-context-local)](https://pypi.org/project/claude-context-local/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Local semantic code search [MCP](https://modelcontextprotocol.io/) server for [Claude Code](https://docs.anthropic.com/en/docs/claude-code). Zero external APIs — runs entirely on your machine.

A lightweight alternative to [zilliztech/claude-context](https://github.com/zilliztech/claude-context) that uses local embeddings instead of OpenAI + Zilliz Cloud.

## Features

- **100% local** — no API keys, no cloud, no data leaves your machine
- **[sentence-transformers](https://www.sbert.net/)** (`all-MiniLM-L6-v2`, 88 MB) for embeddings
- **[ChromaDB](https://www.trychroma.com/)** for persistent vector storage
- **Per-project isolation** — each project gets its own index
- **Incremental indexing** — only re-indexes changed files (MD5 hash)
- **40+ file types** supported out of the box

## Quick start

```bash
claude mcp add claude-context-local -- uvx claude-context-local
```

That's it. Restart Claude Code and the tools are available.

### Alternative: pip

```bash
pip install claude-context-local
claude mcp add claude-context-local -- claude-context-local
```

### Alternative: from source

```bash
git clone https://github.com/tazhate/claude-context-local.git
cd claude-context-local
pip install -e .
claude mcp add claude-context-local -- claude-context-local
```

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

## How it works

```
┌─────────────┐     ┌──────────────────┐     ┌──────────┐
│ Claude Code  │────▸│ claude-context-   │────▸│ ChromaDB │
│  (MCP client)│◂────│ local (MCP server)│◂────│ (vectors)│
└─────────────┘     └──────────────────┘     └──────────┘
                            │
                    ┌───────┴────────┐
                    │ sentence-      │
                    │ transformers   │
                    │ (embeddings)   │
                    └────────────────┘
```

1. **Index**: Walk project files → split into overlapping chunks (50 lines, 10 overlap) → embed with sentence-transformers → store in ChromaDB
2. **Search**: Embed query → cosine similarity search in ChromaDB → return ranked code snippets with file paths and line numbers
3. **Incremental updates**: MD5 hash per file — only changed files are re-embedded

### Per-project isolation

Each project gets its own ChromaDB database under `~/.cache/claude-context-local/<hash>/`, where `<hash>` is derived from the absolute project path. Projects never mix.

## Configuration

Environment variables (pass via `claude mcp add -e KEY=VALUE`):

| Variable | Default | Description |
|---|---|---|
| `CCL_MODEL` | `all-MiniLM-L6-v2` | [sentence-transformers model](https://www.sbert.net/docs/pretrained_models.html) |
| `CCL_CHUNK_LINES` | `50` | Max lines per chunk |
| `CCL_CHUNK_OVERLAP` | `10` | Overlap lines between chunks |
| `CCL_DATA_DIR` | `~/.cache/claude-context-local` | Index storage directory |

### Custom model example

```bash
claude mcp add claude-context-local \
  -e CCL_MODEL=jinaai/jina-embeddings-v2-base-code \
  -- uvx claude-context-local
```

## Resource usage

| Resource | Value |
|---|---|
| RAM | ~780 MB (PyTorch + model in memory) |
| Model on disk | 88 MB (downloaded once) |
| Index size | ~27 MB per 500 files |
| CPU | Near zero at idle |
| First index | ~2 min for 500 files (CPU) |
| Incremental | Seconds (only changed files) |

## Supported file types

**Code:** `.py` `.go` `.js` `.ts` `.tsx` `.jsx` `.rs` `.java` `.kt` `.c` `.cpp` `.h` `.hpp` `.cs` `.rb` `.php` `.swift` `.scala` `.sh` `.bash` `.lua` `.zig` `.nim` `.ex` `.exs` `.erl` `.nix`

**Config:** `.yaml` `.yml` `.toml` `.json` `.hcl` `.tf` `.sql` `.graphql` `.proto`

**Docs:** `.md` `.txt` `.rst`

**Web:** `.html` `.css` `.scss` `.less`

**Other:** `Dockerfile`, `Makefile`

## Comparison with zilliztech/claude-context

| | claude-context-local | zilliztech/claude-context |
|---|---|---|
| Embeddings | Local (sentence-transformers) | OpenAI API |
| Vector DB | Local (ChromaDB) | Zilliz Cloud |
| API keys needed | None | OpenAI + Zilliz |
| Data privacy | 100% local | Cloud |
| Setup | One command | Multiple API keys |
| Cost | Free | Pay per use |
| Search quality | Good | Better (larger models) |
| RAM usage | ~780 MB | ~50 MB (Node.js) |

## Development

```bash
git clone https://github.com/tazhate/claude-context-local.git
cd claude-context-local
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

[MIT](LICENSE)
