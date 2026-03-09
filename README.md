# claude-context-local

[![CI](https://github.com/tazhate/claude-context-local/actions/workflows/ci.yml/badge.svg)](https://github.com/tazhate/claude-context-local/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/claude-context-local)](https://pypi.org/project/claude-context-local/)
[![Python](https://img.shields.io/pypi/pyversions/claude-context-local)](https://pypi.org/project/claude-context-local/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Your entire codebase as context.** A local [MCP](https://modelcontextprotocol.io/) server that gives [Claude Code](https://docs.anthropic.com/en/docs/claude-code) deep semantic understanding of your codebase — without sending a single byte to the cloud.

claude-context-local uses AST-aware chunking and hybrid semantic+keyword search to find all relevant code from your entire codebase. No multi-round file discovery needed. It brings results straight into Claude's context.

**Cost-effective for large codebases:** Instead of loading entire directories into Claude for every request (which can be very expensive), claude-context-local efficiently stores your codebase in a local vector database and only retrieves the code that's actually relevant — keeping your token usage manageable.

A lightweight alternative to [zilliztech/claude-context](https://github.com/zilliztech/claude-context) that uses local embeddings instead of OpenAI + Zilliz Cloud.

## Features

- **100% local** — no API keys, no cloud, no data leaves your machine
- **AST-aware chunking** — splits code at function/class boundaries using tree-sitter (9+ languages), not arbitrary line counts
- **Hybrid search** — BM25 keyword + semantic embedding for best-of-both-worlds results
- **Language-aware metadata** — search results include language, symbol name, and symbol type
- **Symbol dependency graph** — "who calls this function?" / "what does this function call?"
- **Auto-reindex** — file watcher detects changes and re-indexes in the background
- **Multi-project search** — search across all your indexed projects at once
- **Context-aware results** — see surrounding code lines for better understanding
- **Diff-aware search** — search only code that changed since a git ref
- **Lightweight** — ONNX embeddings (~200 MB RAM, no PyTorch required)
- **`.gitignore`-aware** — respects your project's gitignore patterns
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
| `index_project(project_path)` | Index a codebase with AST-aware chunking. Incremental by default, `force=True` to rebuild. `watch=True` to auto-reindex on file changes. |
| `search_code(query, project_path)` | Hybrid semantic+keyword search. Supports `file_filter`, `symbol_type`, and `context_lines`. |
| `search_all(query)` | Search across ALL indexed projects at once. |
| `search_diff(project_path, query, ref)` | Search only code changed since a git ref (commit/branch/tag). |
| `find_symbol(symbol_name, project_path)` | Find who calls a function (`callers`) or what it calls (`callees`). |
| `index_status(project_path)` | Show index stats: files, chunks, languages, symbols, watcher status. |
| `drop_index(project_path)` | Remove project index and stop watcher. |

## Usage examples

Once connected, Claude Code will automatically use these tools. You can also ask directly:

- *"Index this project"* — triggers `index_project`
- *"Search for authentication logic"* — semantic search across your codebase
- *"Find all Python functions related to caching"* — `search_code` with `file_filter="*.py"` and `symbol_type="function"`
- *"Who calls the validate_email function?"* — triggers `find_symbol`
- *"What changed since yesterday?"* — triggers `search_diff`
- *"Search for error handling across all my projects"* — triggers `search_all`

## How it works

```
                           ┌───────────────┐
                           │  tree-sitter  │
                           │  AST parser   │
                           └───────┬───────┘
                                   │
┌─────────────┐     ┌──────────────┴───────┐     ┌──────────┐
│ Claude Code  │────>│ claude-context-local │────>│ ChromaDB │
│  (MCP client)│<────│    (MCP server)      │<────│ (vectors)│
└─────────────┘     └──────────────┬───────┘     └──────────┘
                         │    │    │
                    ┌────┴┐ ┌┴───┐ ┌┴────────┐
                    │ ONNX│ │BM25│ │ Symbol  │
                    │embed│ │keys│ │  Graph  │
                    └─────┘ └────┘ └─────────┘
```

1. **Index**: Walk project files → parse AST with tree-sitter → split at function/class boundaries → embed with ONNX model + build BM25 index + build symbol call graph → store in ChromaDB
2. **Search**: Hybrid — cosine similarity (semantic) + BM25 (keyword) merged with configurable alpha → ranked results with file paths, line numbers, language, symbol info
3. **Incremental**: MD5 hash per file — only changed files are re-processed
4. **Watch**: `watchfiles` monitors your project directory and triggers incremental re-index on save

### AST-aware chunking

Traditional tools split files at arbitrary line boundaries, cutting functions in half. claude-context-local uses [tree-sitter](https://tree-sitter.github.io/tree-sitter/) to parse code into AST and split at natural boundaries:

| Language | Supported symbols |
|---|---|
| Python | functions, classes, decorated definitions |
| Go | functions, methods, types |
| JavaScript/TypeScript | functions, classes, exports, interfaces |
| Rust | functions, structs, impls, enums, traits |
| Java | methods, classes, interfaces |
| C/C++ | functions, structs, classes, namespaces |
| Ruby | methods, classes, modules |
| PHP | functions, classes, methods |
| Bash | functions |

Files without tree-sitter support fall back to overlapping line-based chunking.

### Per-project isolation

Each project gets its own ChromaDB database under `~/.cache/claude-context-local/<hash>/`, where `<hash>` is derived from the absolute project path. Projects never mix.

## Configuration

Environment variables (pass via `claude mcp add -e KEY=VALUE`):

| Variable | Default | Description |
|---|---|---|
| `CCL_MODEL` | `all-MiniLM-L6-v2` | Embedding model (default uses built-in ONNX, no PyTorch) |
| `CCL_HYBRID_ALPHA` | `0.7` | Search blend: 0=BM25 only, 1=semantic only |
| `CCL_CHUNK_LINES` | `50` | Max lines per chunk |
| `CCL_CHUNK_OVERLAP` | `10` | Overlap lines between chunks |
| `CCL_CONTEXT_LINES` | `5` | Default surrounding context lines in results |
| `CCL_DATA_DIR` | `~/.cache/claude-context-local` | Index storage directory |

### Custom model example

```bash
# Use a code-specific model (requires: pip install claude-context-local[gpu])
claude mcp add claude-context-local \
  -e CCL_MODEL=jinaai/jina-embeddings-v2-base-code \
  -- uvx claude-context-local

# More keyword-heavy search
claude mcp add claude-context-local \
  -e CCL_HYBRID_ALPHA=0.4 \
  -- uvx claude-context-local
```

## Resource usage

| Resource | Default (ONNX) | With `[gpu]` (PyTorch) |
|---|---|---|
| RAM | ~200 MB | ~780 MB |
| Model on disk | 80 MB | 88 MB |
| Install size | ~310 MB | ~2 GB |
| Index size | ~27 MB per 500 files | same |
| CPU | Near zero at idle | same |
| First index | ~2 min for 500 files | same |

## Supported file types

**Code:** `.py` `.go` `.js` `.ts` `.tsx` `.jsx` `.rs` `.java` `.kt` `.c` `.cpp` `.h` `.hpp` `.cs` `.rb` `.php` `.swift` `.scala` `.sh` `.bash` `.lua` `.zig` `.nim` `.ex` `.exs` `.erl` `.nix`

**Config:** `.yaml` `.yml` `.toml` `.json` `.hcl` `.tf` `.sql` `.graphql` `.proto`

**Docs:** `.md` `.txt` `.rst`

**Web:** `.html` `.css` `.scss` `.less`

**Other:** `Dockerfile`, `Makefile`

## Comparison with zilliztech/claude-context

| | claude-context-local | zilliztech/claude-context |
|---|---|---|
| Embeddings | Local (ONNX, no PyTorch) | OpenAI API |
| Vector DB | Local (ChromaDB) | Zilliz Cloud |
| Hybrid search | BM25 + semantic | BM25 + semantic |
| AST chunking | tree-sitter (9+ languages) | No |
| Symbol graph | Yes (who calls / what calls) | No |
| Auto-reindex | Yes (file watcher) | No |
| Multi-project | Yes | No |
| Diff search | Yes (git-aware) | No |
| Context lines | Yes | No |
| API keys needed | None | OpenAI + Zilliz |
| Data privacy | 100% local | Cloud |
| Setup | One command | Multiple API keys |
| Cost | Free | Pay per use |
| Search quality | Good | Better (larger models) |
| .gitignore | Yes | No |
| RAM usage | ~200 MB | ~50 MB (Node.js) |

## Security

- All data stays local — no network calls, no telemetry, no cloud
- Index files stored under `~/.cache/` with user-only permissions
- No secrets or credentials are ever indexed (lock files, `.env` excluded)
- CI runs [pip-audit](https://github.com/pypa/pip-audit) and [bandit](https://bandit.readthedocs.io/) on every push

## Development

```bash
git clone https://github.com/tazhate/claude-context-local.git
cd claude-context-local
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```

## License

[MIT](LICENSE)
