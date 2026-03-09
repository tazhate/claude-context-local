#!/usr/bin/env python3
"""Local semantic code search MCP server.

v0.3.0 — AST-aware chunking, language metadata, auto-reindex,
multi-project search, context-aware results, symbol graph, diff-aware search.

Uses ChromaDB built-in ONNX embeddings and BM25 for hybrid search.
Per-project indexing with persistent storage.
"""

import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings
from collections import Counter
from pathlib import Path

import chromadb
from mcp.server.fastmcp import FastMCP

# Suppress tree-sitter FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")

# --- Config ---

MODEL_NAME = os.environ.get("CCL_MODEL", "all-MiniLM-L6-v2")
CHUNK_MAX_LINES = int(os.environ.get("CCL_CHUNK_LINES", "50"))
CHUNK_OVERLAP_LINES = int(os.environ.get("CCL_CHUNK_OVERLAP", "10"))
DATA_DIR = Path(
    os.environ.get("CCL_DATA_DIR", Path.home() / ".cache" / "claude-context-local")
)
HYBRID_ALPHA = float(
    os.environ.get("CCL_HYBRID_ALPHA", "0.7")
)  # 0=BM25 only, 1=semantic only
CONTEXT_LINES = int(
    os.environ.get("CCL_CONTEXT_LINES", "5")
)  # surrounding lines in results

# File extensions to index
CODE_EXTENSIONS = {
    ".py",
    ".go",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".rs",
    ".java",
    ".kt",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".scala",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".hcl",
    ".tf",
    ".sql",
    ".graphql",
    ".proto",
    ".md",
    ".txt",
    ".rst",
    ".dockerfile",
    ".containerfile",
    ".html",
    ".css",
    ".scss",
    ".less",
    ".lua",
    ".zig",
    ".nim",
    ".ex",
    ".exs",
    ".erl",
    ".nix",
    ".dhall",
}

# Directories to always skip
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "venv",
    "env",
    ".env",
    "vendor",
    "third_party",
    ".tox",
    ".eggs",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".output",
    "target",  # rust/java
    ".terraform",
    ".claude",
}

SKIP_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "go.sum",
    "Cargo.lock",
    "poetry.lock",
    "uv.lock",
    "Pipfile.lock",
    "composer.lock",
    "Gemfile.lock",
}

MAX_FILE_SIZE = 512 * 1024  # 512KB

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("claude-context-local")

mcp = FastMCP("claude-context-local")


# --- AST parsing (Feature: tq1 + 7r8) ---

# Extension -> tree-sitter language name
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".go": "go",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "javascript",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "c_sharp",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".scala": "scala",
    ".kt": "kotlin",
    ".swift": "swift",
    ".lua": "lua",
    ".ex": "elixir",
    ".exs": "elixir",
    ".zig": "zig",
    ".hcl": "hcl",
    ".tf": "hcl",
}

# Node types that represent top-level symbols per language
_SYMBOL_NODES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "javascript": {
        "function_declaration",
        "class_declaration",
        "lexical_declaration",
        "export_statement",
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "lexical_declaration",
        "export_statement",
        "interface_declaration",
        "type_alias_declaration",
    },
    "tsx": {
        "function_declaration",
        "class_declaration",
        "lexical_declaration",
        "export_statement",
    },
    "rust": {"function_item", "struct_item", "impl_item", "enum_item", "trait_item"},
    "java": {"method_declaration", "class_declaration", "interface_declaration"},
    "c": {"function_definition", "struct_specifier", "declaration"},
    "cpp": {
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "namespace_definition",
    },
    "ruby": {"method", "class", "module"},
    "bash": {"function_definition"},
    "php": {"function_definition", "class_declaration", "method_declaration"},
}

_PARSER_CACHE: dict[str, object] = {}
_TREE_SITTER_AVAILABLE = None


def _check_tree_sitter() -> bool:
    global _TREE_SITTER_AVAILABLE
    if _TREE_SITTER_AVAILABLE is None:
        try:
            from tree_sitter_languages import get_parser  # noqa: F401

            _TREE_SITTER_AVAILABLE = True
        except ImportError:
            _TREE_SITTER_AVAILABLE = False
            log.info("tree-sitter-languages not installed, using line-based chunking")
    return _TREE_SITTER_AVAILABLE


def _get_parser(lang: str):
    if lang not in _PARSER_CACHE:
        try:
            from tree_sitter_languages import get_parser

            _PARSER_CACHE[lang] = get_parser(lang)
        except Exception:
            _PARSER_CACHE[lang] = None
    return _PARSER_CACHE[lang]


def _detect_language(filepath: Path) -> str | None:
    return _EXT_TO_LANG.get(filepath.suffix.lower())


def _extract_symbol_name(node) -> str:
    """Extract the symbol name from an AST node."""
    # For decorated definitions (Python), look at the inner definition
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return _extract_symbol_name(child)

    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8", errors="replace")

    # For export statements, try to get the declaration name
    if node.type == "export_statement":
        for child in node.children:
            name = _extract_symbol_name(child)
            if name:
                return name

    return ""


def _symbol_type_from_node(node_type: str) -> str:
    """Map AST node type to a simple symbol type."""
    if "function" in node_type or "method" in node_type:
        return "function"
    if "class" in node_type:
        return "class"
    if "struct" in node_type:
        return "struct"
    if "interface" in node_type:
        return "interface"
    if "enum" in node_type:
        return "enum"
    if "trait" in node_type:
        return "trait"
    if "impl" in node_type:
        return "impl"
    if "type" in node_type:
        return "type"
    if "module" in node_type or "namespace" in node_type:
        return "module"
    if "declaration" in node_type or "lexical" in node_type:
        return "declaration"
    return "other"


def _ast_chunk_file(filepath: Path, project_root: str) -> list[dict] | None:
    """Try to chunk a file using AST boundaries. Returns None if AST unavailable."""
    if not _check_tree_sitter():
        return None

    lang = _detect_language(filepath)
    if not lang:
        return None

    parser = _get_parser(lang)
    if not parser:
        return None

    try:
        content = filepath.read_text(errors="replace")
    except (OSError, UnicodeDecodeError):
        return None

    lines = content.splitlines()
    if not lines:
        return None

    try:
        tree = parser.parse(content.encode("utf-8", errors="replace"))
    except Exception:
        return None

    root = tree.root_node
    rel_path = str(filepath.relative_to(project_root))
    symbol_nodes = _SYMBOL_NODES.get(lang, set())

    # Collect top-level symbol boundaries
    symbols = []
    for child in root.children:
        if child.type in symbol_nodes:
            symbols.append(
                {
                    "start_line": child.start_point[0],
                    "end_line": child.end_point[0],
                    "name": _extract_symbol_name(child),
                    "type": _symbol_type_from_node(child.type),
                    "node_type": child.type,
                }
            )

    if not symbols:
        # No recognized symbols — fall back to line-based
        return None

    chunks = []
    covered = set()

    for sym in symbols:
        start = sym["start_line"]
        end = sym["end_line"] + 1  # inclusive -> exclusive

        # If symbol is too large, split it with overlap
        if (end - start) > CHUNK_MAX_LINES:
            pos = start
            while pos < end:
                chunk_end = min(pos + CHUNK_MAX_LINES, end)
                chunk_text = "\n".join(lines[pos:chunk_end])
                chunks.append(
                    {
                        "id": f"{rel_path}:{pos}",
                        "document": chunk_text,
                        "metadata": {
                            "file": rel_path,
                            "start_line": pos + 1,
                            "end_line": chunk_end,
                            "total_lines": len(lines),
                            "language": lang,
                            "symbol_name": sym["name"],
                            "symbol_type": sym["type"],
                        },
                    }
                )
                for ln in range(pos, chunk_end):
                    covered.add(ln)
                if chunk_end >= end:
                    break
                pos += CHUNK_MAX_LINES - CHUNK_OVERLAP_LINES
        else:
            chunk_text = "\n".join(lines[start:end])
            chunks.append(
                {
                    "id": f"{rel_path}:{start}",
                    "document": chunk_text,
                    "metadata": {
                        "file": rel_path,
                        "start_line": start + 1,
                        "end_line": end,
                        "total_lines": len(lines),
                        "language": lang,
                        "symbol_name": sym["name"],
                        "symbol_type": sym["type"],
                    },
                }
            )
            for ln in range(start, end):
                covered.add(ln)

    # Collect uncovered lines into "gap" chunks (imports, comments, etc.)
    uncovered_ranges = []
    gap_start = None
    for i in range(len(lines)):
        if i not in covered:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                uncovered_ranges.append((gap_start, i))
                gap_start = None
    if gap_start is not None:
        uncovered_ranges.append((gap_start, len(lines)))

    for start, end in uncovered_ranges:
        if (end - start) < 2:
            continue  # skip trivial gaps (blank lines)
        chunk_text = "\n".join(lines[start:end])
        if chunk_text.strip():
            chunks.append(
                {
                    "id": f"{rel_path}:{start}",
                    "document": chunk_text,
                    "metadata": {
                        "file": rel_path,
                        "start_line": start + 1,
                        "end_line": end,
                        "total_lines": len(lines),
                        "language": lang,
                        "symbol_name": "",
                        "symbol_type": "preamble",
                    },
                }
            )

    # Sort by start line
    chunks.sort(key=lambda c: c["metadata"]["start_line"])
    return chunks if chunks else None


# --- Symbol graph (Feature: 7ub) ---


def _extract_calls(filepath: Path, lang: str) -> dict[str, list[str]] | None:
    """Extract function call relationships from a file using tree-sitter."""
    if not _check_tree_sitter():
        return None

    parser = _get_parser(lang)
    if not parser:
        return None

    try:
        content = filepath.read_bytes()
        tree = parser.parse(content)
    except Exception:
        return None

    symbol_nodes = _SYMBOL_NODES.get(lang, set())
    calls: dict[str, list[str]] = {}

    def _walk_calls(node, current_func: str | None = None):
        """Walk AST to find function definitions and their call expressions."""
        if node.type in symbol_nodes:
            name = _extract_symbol_name(node)
            if name:
                current_func = name
                if current_func not in calls:
                    calls[current_func] = []

        if node.type == "call" or node.type == "call_expression":
            callee = node.child_by_field_name("function") or node.child_by_field_name(
                "name"
            )
            if not callee:
                # Try first child for simple calls
                for child in node.children:
                    if (
                        child.type == "identifier"
                        or child.type == "member_expression"
                        or child.type == "selector_expression"
                    ):
                        callee = child
                        break

            if callee and current_func:
                callee_name = callee.text.decode("utf-8", errors="replace")
                # Simplify: take last part of dotted name
                if "." in callee_name:
                    callee_name = callee_name.split(".")[-1]
                calls[current_func].append(callee_name)

        for child in node.children:
            _walk_calls(child, current_func)

    _walk_calls(tree.root_node)
    return calls


class SymbolGraph:
    """Stores function call relationships as adjacency list."""

    def __init__(self, path: Path):
        self.path = path / "symbol_graph.json"
        self.graph: dict[str, dict[str, list[str]]] = {}  # file -> {func -> [callees]}
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        if self.path.exists():
            try:
                self.graph = json.loads(self.path.read_text())
            except Exception:
                self.graph = {}
        self._loaded = True

    def save(self):
        self.path.write_text(json.dumps(self.graph))

    def update_file(self, rel_path: str, calls: dict[str, list[str]]):
        self._load()
        self.graph[rel_path] = calls

    def remove_file(self, rel_path: str):
        self._load()
        self.graph.pop(rel_path, None)

    def clear(self):
        self.graph = {}
        self._loaded = True
        if self.path.exists():
            self.path.unlink()

    def who_calls(self, symbol_name: str) -> list[tuple[str, str]]:
        """Find all functions that call the given symbol. Returns [(file, caller)]."""
        self._load()
        results = []
        for file, funcs in self.graph.items():
            for func, callees in funcs.items():
                if symbol_name in callees:
                    results.append((file, func))
        return results

    def what_calls(self, symbol_name: str) -> list[str]:
        """Find all functions called by the given symbol."""
        self._load()
        for funcs in self.graph.values():
            if symbol_name in funcs:
                return funcs[symbol_name]
        return []


def get_symbol_graph(proj_path: str) -> SymbolGraph:
    pid = project_id(proj_path)
    db_path = DATA_DIR / pid
    db_path.mkdir(parents=True, exist_ok=True)
    return SymbolGraph(db_path)


# --- Embedding function ---


def get_embedding_function():
    """Get the embedding function based on config.

    Uses ChromaDB's built-in ONNX embedder by default (no PyTorch needed).
    Falls back to sentence-transformers if a custom model is specified.
    """
    if MODEL_NAME == "all-MiniLM-L6-v2":
        return chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
    else:
        try:
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )

            return SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
        except ImportError:
            log.warning(
                "sentence-transformers not installed, falling back to default ONNX model. "
                "Install with: pip install sentence-transformers"
            )
            return chromadb.utils.embedding_functions.DefaultEmbeddingFunction()


# --- BM25 ---

_TOKENIZE_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for code: split on word boundaries, lowercase."""
    return [t.lower() for t in _TOKENIZE_RE.findall(text)]


class BM25Index:
    """Simple BM25 index stored as JSON alongside ChromaDB."""

    def __init__(self, path: Path):
        self.path = path / "bm25.json"
        self.docs: dict[str, list[str]] = {}  # id -> tokens
        self.df: Counter = Counter()  # token -> doc frequency
        self.avgdl: float = 0.0
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.docs = data.get("docs", {})
                self.df = Counter(data.get("df", {}))
                self.avgdl = data.get("avgdl", 0.0)
            except Exception:
                self.docs, self.df, self.avgdl = {}, Counter(), 0.0
        self._loaded = True

    def save(self):
        self.path.write_text(
            json.dumps(
                {
                    "docs": self.docs,
                    "df": dict(self.df),
                    "avgdl": self.avgdl,
                }
            )
        )

    def add(self, doc_id: str, text: str):
        self._load()
        tokens = tokenize(text)
        if doc_id in self.docs:
            self.remove(doc_id)
        self.docs[doc_id] = tokens
        for t in set(tokens):
            self.df[t] += 1
        self._recompute_avgdl()

    def remove(self, doc_id: str):
        self._load()
        if doc_id not in self.docs:
            return
        old_tokens = self.docs.pop(doc_id)
        for t in set(old_tokens):
            self.df[t] -= 1
            if self.df[t] <= 0:
                del self.df[t]
        self._recompute_avgdl()

    def remove_by_prefix(self, prefix: str):
        self._load()
        to_remove = [did for did in self.docs if did.startswith(prefix)]
        for did in to_remove:
            self.remove(did)

    def clear(self):
        self.docs, self.df, self.avgdl = {}, Counter(), 0.0
        self._loaded = True
        if self.path.exists():
            self.path.unlink()

    def _recompute_avgdl(self):
        if self.docs:
            self.avgdl = sum(len(t) for t in self.docs.values()) / len(self.docs)
        else:
            self.avgdl = 0.0

    def search(self, query: str, top_k: int = 20) -> dict[str, float]:
        """Return {doc_id: score} for top-k BM25 matches."""
        self._load()
        if not self.docs:
            return {}

        query_tokens = tokenize(query)
        if not query_tokens:
            return {}

        k1, b = 1.5, 0.75
        n = len(self.docs)
        scores: dict[str, float] = {}

        for doc_id, doc_tokens in self.docs.items():
            tf_map = Counter(doc_tokens)
            dl = len(doc_tokens)
            score = 0.0
            for qt in query_tokens:
                if qt not in tf_map:
                    continue
                tf = tf_map[qt]
                df = self.df.get(qt, 0)
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                tf_norm = (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * dl / max(self.avgdl, 1))
                )
                score += idf * tf_norm
            if score > 0:
                scores[doc_id] = score

        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return {did: scores[did] for did in sorted_ids}


# --- Gitignore ---


def load_gitignore_patterns(project_path: str) -> list[str]:
    """Load .gitignore patterns from project root."""
    gitignore = Path(project_path) / ".gitignore"
    if not gitignore.exists():
        return []
    try:
        lines = gitignore.read_text().splitlines()
        return [
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]
    except OSError:
        return []


def matches_gitignore(rel_path: str, patterns: list[str]) -> bool:
    """Check if a relative path matches any gitignore pattern."""
    parts = rel_path.split("/")
    filename = parts[-1]

    for pattern in patterns:
        if pattern.startswith("!"):
            continue

        is_dir_pattern = pattern.endswith("/")
        p = pattern.rstrip("/")

        if p.startswith("**/"):
            p = p[3:]
            if _simple_match(p, rel_path) or _simple_match(p, filename):
                return True
            for i in range(len(parts)):
                if _simple_match(p, "/".join(parts[i:])):
                    return True
        elif "/" in p:
            if _simple_match(p, rel_path):
                return True
            if rel_path.startswith(p + "/"):
                return True
        elif is_dir_pattern:
            if p in parts[:-1]:
                return True
        else:
            if _simple_match(p, filename):
                return True
            if p in parts:
                return True

    return False


def _simple_match(pattern: str, text: str) -> bool:
    """Simple glob match supporting * and **."""
    regex = "^"
    i = 0
    while i < len(pattern):
        if pattern[i] == "*":
            if i + 1 < len(pattern) and pattern[i + 1] == "*":
                regex += ".*"
                i += 2
                if i < len(pattern) and pattern[i] == "/":
                    i += 1
                continue
            else:
                regex += "[^/]*"
        elif pattern[i] == "?":
            regex += "[^/]"
        elif pattern[i] in ".+^${}()|[]":
            regex += "\\" + pattern[i]
        else:
            regex += pattern[i]
        i += 1
    regex += "$"
    return bool(re.match(regex, text))


# --- Helpers ---


def project_id(project_path: str) -> str:
    """Stable ID for a project path."""
    return hashlib.sha256(project_path.encode()).hexdigest()[:16]


def get_collection(proj_path: str):
    """Get or create a ChromaDB collection for a project."""
    pid = project_id(proj_path)
    db_path = DATA_DIR / pid
    db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path))
    ef = get_embedding_function()
    return client.get_or_create_collection(
        name="code",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def get_bm25(proj_path: str) -> BM25Index:
    """Get BM25 index for a project."""
    pid = project_id(proj_path)
    db_path = DATA_DIR / pid
    db_path.mkdir(parents=True, exist_ok=True)
    return BM25Index(db_path)


def should_index(path: Path) -> bool:
    """Check if a file should be indexed."""
    if path.name in SKIP_FILES:
        return False
    if path.name == "Dockerfile" or path.name == "Makefile":
        return True
    if path.suffix.lower() in CODE_EXTENSIONS:
        return True
    return False


def walk_project(project_path: str, gitignore_patterns: list[str] | None = None):
    """Yield files to index."""
    root = Path(project_path)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            fpath = Path(dirpath) / fname
            if not should_index(fpath):
                continue

            if gitignore_patterns:
                rel = str(fpath.relative_to(root))
                if matches_gitignore(rel, gitignore_patterns):
                    continue

            try:
                size = fpath.stat().st_size
                if size <= MAX_FILE_SIZE and size > 0:
                    yield fpath
            except OSError:
                continue


def chunk_file(filepath: Path, project_root: str) -> list[dict]:
    """Split a file into chunks — AST-aware if possible, line-based fallback."""
    # Try AST-aware chunking first
    ast_chunks = _ast_chunk_file(filepath, project_root)
    if ast_chunks:
        return ast_chunks

    # Fallback: line-based chunking
    try:
        content = filepath.read_text(errors="replace")
    except (OSError, UnicodeDecodeError):
        return []

    lines = content.splitlines()
    if not lines:
        return []

    rel_path = str(filepath.relative_to(project_root))
    lang = _detect_language(filepath) or ""
    chunks = []

    if len(lines) <= CHUNK_MAX_LINES:
        chunks.append(
            {
                "id": f"{rel_path}:0",
                "document": content,
                "metadata": {
                    "file": rel_path,
                    "start_line": 1,
                    "end_line": len(lines),
                    "total_lines": len(lines),
                    "language": lang,
                    "symbol_name": "",
                    "symbol_type": "",
                },
            }
        )
    else:
        start = 0
        while start < len(lines):
            end = min(start + CHUNK_MAX_LINES, len(lines))
            chunk_lines = lines[start:end]
            chunk_text = "\n".join(chunk_lines)

            chunks.append(
                {
                    "id": f"{rel_path}:{start}",
                    "document": chunk_text,
                    "metadata": {
                        "file": rel_path,
                        "start_line": start + 1,
                        "end_line": end,
                        "total_lines": len(lines),
                        "language": lang,
                        "symbol_name": "",
                        "symbol_type": "",
                    },
                }
            )

            if end >= len(lines):
                break
            start += CHUNK_MAX_LINES - CHUNK_OVERLAP_LINES

    return chunks


def file_hash(filepath: Path) -> str:
    """Fast hash of file content for change detection."""
    h = hashlib.md5()
    try:
        h.update(filepath.read_bytes())
    except OSError:
        return ""
    return h.hexdigest()


def _read_file_lines(filepath: Path) -> list[str] | None:
    """Read file lines for context display."""
    try:
        return filepath.read_text(errors="replace").splitlines()
    except (OSError, UnicodeDecodeError):
        return None


def _get_context_lines(
    project_path: str, file_rel: str, start_line: int, end_line: int, context: int
) -> str:
    """Get chunk content with surrounding context lines."""
    if context <= 0:
        return ""

    filepath = Path(project_path) / file_rel
    lines = _read_file_lines(filepath)
    if not lines:
        return ""

    ctx_start = max(0, start_line - 1 - context)
    ctx_end = min(len(lines), end_line + context)

    result = []
    for i in range(ctx_start, ctx_end):
        marker = "  " if (start_line - 1) <= i < end_line else "| "
        result.append(f"{i + 1:4d}{marker}{lines[i]}")

    return "\n".join(result)


# --- Auto-reindex watcher (Feature: 1p5) ---

_watchers: dict[str, threading.Event] = {}


def _start_watcher(project_path: str):
    """Start a background file watcher for auto-reindexing."""
    if project_path in _watchers:
        return  # already watching

    try:
        import watchfiles
    except ImportError:
        log.info("watchfiles not installed, auto-reindex disabled")
        return

    stop_event = threading.Event()
    _watchers[project_path] = stop_event

    gitignore_patterns = load_gitignore_patterns(project_path)

    def _watch_thread():
        root = Path(project_path)
        log.info("Watcher started for %s", project_path)
        try:
            for changes in watchfiles.watch(
                project_path,
                stop_event=stop_event,
                debounce=2000,  # 2s debounce
                recursive=True,
            ):
                changed_files = set()
                for change_type, path_str in changes:
                    fpath = Path(path_str)
                    if not should_index(fpath):
                        continue
                    # Check skip dirs
                    try:
                        rel = str(fpath.relative_to(root))
                    except ValueError:
                        continue
                    parts = rel.split("/")
                    if any(p in SKIP_DIRS for p in parts[:-1]):
                        continue
                    if gitignore_patterns and matches_gitignore(
                        rel, gitignore_patterns
                    ):
                        continue
                    changed_files.add(rel)

                if changed_files:
                    log.info("Auto-reindex: %d files changed", len(changed_files))
                    try:
                        _incremental_reindex(project_path, changed_files)
                    except Exception as e:
                        log.error("Auto-reindex error: %s", e)
        except Exception as e:
            if not stop_event.is_set():
                log.error("Watcher error: %s", e)
        finally:
            _watchers.pop(project_path, None)
            log.info("Watcher stopped for %s", project_path)

    thread = threading.Thread(
        target=_watch_thread,
        daemon=True,
        name=f"watcher-{project_id(project_path)[:8]}",
    )
    thread.start()


def _stop_watcher(project_path: str):
    stop_event = _watchers.pop(project_path, None)
    if stop_event:
        stop_event.set()


def _incremental_reindex(project_path: str, changed_files: set[str]):
    """Reindex only the specified changed files."""
    collection = get_collection(project_path)
    bm25 = get_bm25(project_path)
    sg = get_symbol_graph(project_path)
    root = Path(project_path)

    for rel in changed_files:
        fpath = root / rel

        # Remove old chunks
        try:
            old_ids = collection.get(where={"file": rel}, include=[])
            if old_ids and old_ids["ids"]:
                collection.delete(ids=old_ids["ids"])
        except Exception:
            pass
        bm25.remove_by_prefix(f"{rel}:")
        sg.remove_file(rel)

        if not fpath.exists():
            continue  # file was deleted

        fh = file_hash(fpath)
        chunks = chunk_file(fpath, project_path)

        batch_ids, batch_docs, batch_metas = [], [], []
        for chunk in chunks:
            chunk["metadata"]["hash"] = fh
            batch_ids.append(chunk["id"])
            batch_docs.append(chunk["document"])
            batch_metas.append(chunk["metadata"])
            bm25.add(chunk["id"], chunk["document"])

        if batch_ids:
            collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

        # Update symbol graph
        lang = _detect_language(fpath)
        if lang:
            calls = _extract_calls(fpath, lang)
            if calls:
                sg.update_file(rel, calls)

    bm25.save()
    sg.save()


# --- Git diff helpers (Feature: bx0) ---


def _git_changed_files(project_path: str, ref: str = "HEAD") -> list[str] | None:
    """Get list of files changed since a git ref."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", ref],
            capture_output=True,
            text=True,
            cwd=project_path,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        files = [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]
        return files
    except Exception:
        return None


def _git_diff_content(project_path: str, ref: str = "HEAD") -> str | None:
    """Get unified diff since a git ref."""
    try:
        result = subprocess.run(
            ["git", "diff", ref],
            capture_output=True,
            text=True,
            cwd=project_path,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except Exception:
        return None


# --- Multi-project registry (Feature: 0py) ---


def _get_project_registry() -> dict[str, str]:
    """Load project registry: {project_path: project_id}."""
    registry_path = DATA_DIR / "projects.json"
    if registry_path.exists():
        try:
            return json.loads(registry_path.read_text())
        except Exception:
            pass
    return {}


def _save_project_registry(registry: dict[str, str]):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "projects.json").write_text(json.dumps(registry))


def _register_project(project_path: str):
    registry = _get_project_registry()
    registry[project_path] = project_id(project_path)
    _save_project_registry(registry)


# --- MCP Tools ---


@mcp.tool()
def index_project(project_path: str, force: bool = False, watch: bool = False) -> str:
    """Index or re-index a project for semantic code search.

    Scans all code files (respecting .gitignore), splits into AST-aware chunks
    (function/class boundaries), and stores embeddings + BM25 index for hybrid search.
    Incremental by default — only re-indexes changed files.

    Args:
        project_path: Absolute path to project root directory.
        force: If True, re-index everything from scratch.
        watch: If True, start background file watcher for auto-reindex on changes.
    """
    project_path = os.path.expanduser(project_path)
    if not os.path.isdir(project_path):
        return f"Error: {project_path} is not a directory"

    collection = get_collection(project_path)
    bm25 = get_bm25(project_path)
    sg = get_symbol_graph(project_path)

    if force:
        pid = project_id(project_path)
        db_path = DATA_DIR / pid
        client = chromadb.PersistentClient(path=str(db_path))
        try:
            client.delete_collection("code")
        except Exception:
            pass
        collection = get_collection(project_path)
        bm25.clear()
        sg.clear()

    gitignore_patterns = load_gitignore_patterns(project_path)

    # Load existing file hashes
    existing_hashes = {}
    try:
        existing = collection.get(include=["metadatas"])
        if existing and existing["metadatas"]:
            for meta in existing["metadatas"]:
                if meta and "file" in meta and "hash" in meta:
                    existing_hashes[meta["file"]] = meta["hash"]
    except Exception:
        pass

    files = list(walk_project(project_path, gitignore_patterns))
    total_files = len(files)
    indexed_files = set()
    new_count = 0
    skip_count = 0
    ast_count = 0
    batch_ids = []
    batch_docs = []
    batch_metas = []

    t0 = time.time()
    last_progress = 0

    for i, fpath in enumerate(files):
        rel = str(fpath.relative_to(project_path))
        fh = file_hash(fpath)

        if not force and rel in existing_hashes and existing_hashes[rel] == fh:
            skip_count += 1
            indexed_files.add(rel)
            continue

        # Remove old chunks
        try:
            old_ids = collection.get(where={"file": rel}, include=[])
            if old_ids and old_ids["ids"]:
                collection.delete(ids=old_ids["ids"])
        except Exception:
            pass
        bm25.remove_by_prefix(f"{rel}:")

        chunks = chunk_file(fpath, project_path)
        has_ast = any(c["metadata"].get("symbol_name") for c in chunks)
        if has_ast:
            ast_count += 1

        for chunk in chunks:
            chunk["metadata"]["hash"] = fh
            batch_ids.append(chunk["id"])
            batch_docs.append(chunk["document"])
            batch_metas.append(chunk["metadata"])
            bm25.add(chunk["id"], chunk["document"])

        # Build symbol graph
        lang = _detect_language(fpath)
        if lang:
            calls = _extract_calls(fpath, lang)
            if calls:
                sg.update_file(rel, calls)

        indexed_files.add(rel)
        new_count += 1

        if len(batch_ids) >= 100:
            collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
            batch_ids, batch_docs, batch_metas = [], [], []

        progress = int((i + 1) / total_files * 100) if total_files else 100
        if progress >= last_progress + 10:
            log.info("Indexing: %d%% (%d/%d files)", progress, i + 1, total_files)
            last_progress = progress

    if batch_ids:
        collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

    bm25.save()
    sg.save()

    # Register project for multi-project search
    _register_project(project_path)

    elapsed = time.time() - t0
    total = collection.count()
    gitignore_info = (
        f"\n.gitignore: {len(gitignore_patterns)} patterns applied"
        if gitignore_patterns
        else ""
    )
    ast_info = (
        f"\nAST-chunked files: {ast_count}"
        if _check_tree_sitter()
        else "\nAST chunking: tree-sitter not installed (using line-based)"
    )

    # Start watcher if requested
    watch_info = ""
    if watch:
        _start_watcher(project_path)
        watch_info = "\nFile watcher: started (auto-reindex on changes)"

    return (
        f"Indexed {new_count} files ({skip_count} unchanged) in {elapsed:.1f}s\n"
        f"Total chunks in index: {total}\n"
        f"Project: {project_path}"
        f"{gitignore_info}"
        f"{ast_info}"
        f"{watch_info}"
    )


@mcp.tool()
def search_code(
    query: str,
    project_path: str,
    n_results: int = 10,
    file_filter: str = "",
    symbol_type: str = "",
    context_lines: int = 0,
) -> str:
    """Search code using hybrid semantic + keyword (BM25) search.

    Combines embedding similarity with BM25 keyword matching for better results.
    Supports filtering by file path, symbol type, and returning surrounding context.

    Args:
        query: Natural language or keyword search query.
        project_path: Absolute path to the indexed project.
        n_results: Number of results to return (default 10).
        file_filter: Optional filter for file paths (e.g. "*.py", "src/", "test").
        symbol_type: Optional filter by symbol type (e.g. "function", "class", "struct").
        context_lines: Number of surrounding context lines to include (0 = chunk only).
    """
    project_path = os.path.expanduser(project_path)
    pid = project_id(project_path)
    db_path = DATA_DIR / pid
    if not db_path.exists():
        return "Project not indexed yet. Run index_project first."

    collection = get_collection(project_path)

    if collection.count() == 0:
        return "Project not indexed yet. Run index_project first."

    # --- Semantic search ---
    fetch_n = min(max(n_results * 3, 20), 50)
    try:
        sem_results = collection.query(
            query_texts=[query],
            n_results=fetch_n,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        return f"Search error: {e}"

    sem_scores: dict[str, float] = {}
    sem_data: dict[str, tuple[str, dict]] = {}
    if sem_results["documents"] and sem_results["documents"][0]:
        for doc, meta, dist in zip(
            sem_results["documents"][0],
            sem_results["metadatas"][0],
            sem_results["distances"][0],
        ):
            doc_id = f"{meta['file']}:{meta['start_line']}"
            similarity = max(0.0, 1.0 - dist)
            sem_scores[doc_id] = similarity
            sem_data[doc_id] = (doc, meta)

    # --- BM25 search ---
    bm25 = get_bm25(project_path)
    bm25_raw = bm25.search(query, top_k=fetch_n)

    bm25_scores: dict[str, float] = {}
    if bm25_raw:
        max_bm25 = max(bm25_raw.values())
        if max_bm25 > 0:
            bm25_scores = {k: v / max_bm25 for k, v in bm25_raw.items()}

    bm25_mapped: dict[str, float] = {}
    for chunk_id, score in bm25_scores.items():
        parts = chunk_id.rsplit(":", 1)
        if len(parts) == 2:
            file_path, offset = parts[0], int(parts[1])
            doc_id = f"{file_path}:{offset + 1}"
            bm25_mapped[doc_id] = score
            if doc_id not in sem_data:
                try:
                    r = collection.get(
                        ids=[chunk_id], include=["documents", "metadatas"]
                    )
                    if r["documents"]:
                        sem_data[doc_id] = (r["documents"][0], r["metadatas"][0])
                except Exception:
                    pass

    # --- Hybrid merge ---
    alpha = HYBRID_ALPHA
    all_ids = set(sem_scores) | set(bm25_mapped)
    hybrid_scores: dict[str, float] = {}
    for doc_id in all_ids:
        s_sem = sem_scores.get(doc_id, 0.0)
        s_bm25 = bm25_mapped.get(doc_id, 0.0)
        hybrid_scores[doc_id] = alpha * s_sem + (1 - alpha) * s_bm25

    ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

    def matches_filter(filepath: str) -> bool:
        if not file_filter:
            return True
        if file_filter.startswith("*."):
            return filepath.endswith(file_filter[1:])
        return file_filter in filepath

    def matches_symbol_type(meta: dict) -> bool:
        if not symbol_type:
            return True
        return meta.get("symbol_type", "") == symbol_type

    output = []
    for doc_id, score in ranked:
        if doc_id not in sem_data:
            continue
        doc, meta = sem_data[doc_id]
        if not matches_filter(meta["file"]):
            continue
        if not matches_symbol_type(meta):
            continue
        if len(output) >= n_results:
            break

        # Build header with metadata
        sym_info = ""
        if meta.get("symbol_name"):
            sym_info = f" [{meta.get('symbol_type', '')} {meta['symbol_name']}]"
        lang_info = f" ({meta['language']})" if meta.get("language") else ""

        header = f"### {meta['file']}:{meta['start_line']}-{meta['end_line']}{lang_info}{sym_info} (score: {score:.3f})"

        # Context-aware results
        ctx = (
            context_lines
            if context_lines > 0
            else CONTEXT_LINES
            if os.environ.get("CCL_CONTEXT_LINES")
            else 0
        )
        if ctx > 0:
            context_text = _get_context_lines(
                project_path, meta["file"], meta["start_line"], meta["end_line"], ctx
            )
            if context_text:
                output.append(f"{header}\n```\n{context_text}\n```\n")
                continue

        output.append(f"{header}\n```\n{doc}\n```\n")

    if not output:
        return "No results found."

    return f"Found {len(output)} results for: {query}\n\n" + "\n".join(output)


@mcp.tool()
def search_all(query: str, n_results: int = 10, file_filter: str = "") -> str:
    """Search across ALL indexed projects at once.

    Useful for finding patterns, shared utilities, or similar code across codebases.

    Args:
        query: Natural language or keyword search query.
        n_results: Total number of results to return across all projects.
        file_filter: Optional filter for file paths (e.g. "*.py", "src/").
    """
    registry = _get_project_registry()
    if not registry:
        return "No projects indexed yet. Run index_project on at least one project."

    all_results = []

    for proj_path, pid in registry.items():
        db_path = DATA_DIR / pid
        if not db_path.exists():
            continue

        try:
            collection = get_collection(proj_path)
            if collection.count() == 0:
                continue
        except Exception:
            continue

        # Semantic search per project
        fetch_n = min(max(n_results * 2, 10), 30)
        try:
            sem_results = collection.query(
                query_texts=[query],
                n_results=fetch_n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            continue

        if not sem_results["documents"] or not sem_results["documents"][0]:
            continue

        proj_name = Path(proj_path).name

        for doc, meta, dist in zip(
            sem_results["documents"][0],
            sem_results["metadatas"][0],
            sem_results["distances"][0],
        ):
            score = max(0.0, 1.0 - dist)

            def matches_filter_inner(filepath: str) -> bool:
                if not file_filter:
                    return True
                if file_filter.startswith("*."):
                    return filepath.endswith(file_filter[1:])
                return file_filter in filepath

            if not matches_filter_inner(meta["file"]):
                continue

            sym_info = ""
            if meta.get("symbol_name"):
                sym_info = f" [{meta.get('symbol_type', '')} {meta['symbol_name']}]"

            all_results.append((score, proj_name, proj_path, doc, meta, sym_info))

    if not all_results:
        return "No results found across any indexed project."

    all_results.sort(key=lambda x: x[0], reverse=True)
    all_results = all_results[:n_results]

    output = []
    for score, proj_name, proj_path, doc, meta, sym_info in all_results:
        header = f"### [{proj_name}] {meta['file']}:{meta['start_line']}-{meta['end_line']}{sym_info} (score: {score:.3f})"
        output.append(f"{header}\n```\n{doc}\n```\n")

    return (
        f"Found {len(output)} results across {len(registry)} projects for: {query}\n\n"
        + "\n".join(output)
    )


@mcp.tool()
def search_diff(
    project_path: str, query: str = "", ref: str = "HEAD", n_results: int = 10
) -> str:
    """Search only code that changed since a git ref (commit, branch, tag).

    If query is empty, returns all changed chunks. If query is provided,
    searches only within changed files.

    Args:
        project_path: Absolute path to the project (must be a git repo).
        query: Optional search query to filter changed code.
        ref: Git ref to diff against (default: HEAD = uncommitted changes).
        n_results: Max results to return.
    """
    project_path = os.path.expanduser(project_path)
    if not os.path.isdir(project_path):
        return f"Error: {project_path} is not a directory"

    changed_files = _git_changed_files(project_path, ref)
    if changed_files is None:
        return "Error: not a git repository or git not available"
    if not changed_files:
        return f"No files changed since {ref}"

    # Filter to indexable files
    root = Path(project_path)
    gitignore_patterns = load_gitignore_patterns(project_path)
    indexable = []
    for f in changed_files:
        fpath = root / f
        if not fpath.exists() or not should_index(fpath):
            continue
        if gitignore_patterns and matches_gitignore(f, gitignore_patterns):
            continue
        indexable.append(f)

    if not indexable:
        return f"No indexable code files changed since {ref}"

    if not query:
        # No query — just show all changed chunks
        output = [f"Changed files since {ref}: {len(indexable)}\n"]
        count = 0
        for rel in indexable:
            if count >= n_results:
                break
            fpath = root / rel
            chunks = chunk_file(fpath, project_path)
            for chunk in chunks:
                if count >= n_results:
                    break
                meta = chunk["metadata"]
                sym_info = ""
                if meta.get("symbol_name"):
                    sym_info = f" [{meta.get('symbol_type', '')} {meta['symbol_name']}]"
                header = f"### {meta['file']}:{meta['start_line']}-{meta['end_line']}{sym_info}"
                output.append(f"{header}\n```\n{chunk['document']}\n```\n")
                count += 1
        return "\n".join(output)

    # Query provided — search only within changed files
    pid = project_id(project_path)
    db_path = DATA_DIR / pid
    if not db_path.exists():
        return "Project not indexed yet. Run index_project first."

    collection = get_collection(project_path)
    if collection.count() == 0:
        return "Project not indexed yet. Run index_project first."

    # Search semantically but filter to changed files
    fetch_n = min(max(n_results * 3, 20), 50)
    try:
        sem_results = collection.query(
            query_texts=[query],
            n_results=fetch_n,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        return f"Search error: {e}"

    changed_set = set(indexable)
    output = []
    if sem_results["documents"] and sem_results["documents"][0]:
        for doc, meta, dist in zip(
            sem_results["documents"][0],
            sem_results["metadatas"][0],
            sem_results["distances"][0],
        ):
            if meta["file"] not in changed_set:
                continue
            if len(output) >= n_results:
                break
            score = max(0.0, 1.0 - dist)
            sym_info = ""
            if meta.get("symbol_name"):
                sym_info = f" [{meta.get('symbol_type', '')} {meta['symbol_name']}]"
            header = f"### {meta['file']}:{meta['start_line']}-{meta['end_line']}{sym_info} (score: {score:.3f})"
            output.append(f"{header}\n```\n{doc}\n```\n")

    if not output:
        return f"No results for '{query}' in changed files since {ref}"

    return f"Found {len(output)} results in changed files since {ref}:\n\n" + "\n".join(
        output
    )


@mcp.tool()
def find_symbol(symbol_name: str, project_path: str, direction: str = "callers") -> str:
    """Find who calls a function/method, or what it calls (symbol dependency graph).

    Uses the symbol graph built during indexing (tree-sitter AST analysis).

    Args:
        symbol_name: Name of the function/method/class to look up.
        project_path: Absolute path to the indexed project.
        direction: "callers" = who calls this symbol, "callees" = what does this symbol call.
    """
    project_path = os.path.expanduser(project_path)
    sg = get_symbol_graph(project_path)

    if direction == "callers":
        results = sg.who_calls(symbol_name)
        if not results:
            return f"No callers found for '{symbol_name}'"
        output = [f"Functions that call '{symbol_name}':\n"]
        for file, caller in results:
            output.append(f"  - {file}: {caller}()")
        return "\n".join(output)
    elif direction == "callees":
        results = sg.what_calls(symbol_name)
        if not results:
            return f"No callees found for '{symbol_name}'"
        output = [f"Functions called by '{symbol_name}':\n"]
        for callee in results:
            output.append(f"  - {callee}()")
        return "\n".join(output)
    else:
        return f"Unknown direction '{direction}'. Use 'callers' or 'callees'."


@mcp.tool()
def index_status(project_path: str) -> str:
    """Check the indexing status of a project.

    Args:
        project_path: Absolute path to the project.
    """
    project_path = os.path.expanduser(project_path)
    pid = project_id(project_path)
    db_path = DATA_DIR / pid
    if not db_path.exists():
        return f"Project not indexed: {project_path}\nRun index_project to index it."

    collection = get_collection(project_path)
    count = collection.count()

    if count == 0:
        return f"Project not indexed: {project_path}\nRun index_project to index it."

    try:
        all_meta = collection.get(include=["metadatas"])
        files = set()
        languages = Counter()
        symbol_types = Counter()
        ast_chunks = 0
        if all_meta and all_meta["metadatas"]:
            for meta in all_meta["metadatas"]:
                if meta and "file" in meta:
                    files.add(meta["file"])
                if meta and meta.get("language"):
                    languages[meta["language"]] += 1
                if meta and meta.get("symbol_type"):
                    symbol_types[meta["symbol_type"]] += 1
                    if meta["symbol_type"] not in ("", "preamble"):
                        ast_chunks += 1

        bm25 = get_bm25(project_path)
        bm25._load()
        bm25_status = f"BM25 documents: {len(bm25.docs)}"

        sg = get_symbol_graph(project_path)
        sg._load()
        sg_status = f"Symbol graph: {len(sg.graph)} files"

        lang_info = (
            ", ".join(f"{lang}={cnt}" for lang, cnt in languages.most_common(5))
            if languages
            else "none detected"
        )
        sym_info = (
            ", ".join(f"{st}={cnt}" for st, cnt in symbol_types.most_common(5))
            if symbol_types
            else "none"
        )

        watcher_status = "active" if project_path in _watchers else "inactive"

        return (
            f"Project: {project_path}\n"
            f"Indexed files: {len(files)}\n"
            f"Total chunks: {count} (AST-aware: {ast_chunks})\n"
            f"Languages: {lang_info}\n"
            f"Symbol types: {sym_info}\n"
            f"{bm25_status}\n"
            f"{sg_status}\n"
            f"Model: {MODEL_NAME}\n"
            f"Hybrid alpha: {HYBRID_ALPHA} (semantic={HYBRID_ALPHA}, bm25={1 - HYBRID_ALPHA})\n"
            f"File watcher: {watcher_status}"
        )
    except Exception as e:
        return f"Status error: {e}"


@mcp.tool()
def drop_index(project_path: str) -> str:
    """Remove the index for a project.

    Args:
        project_path: Absolute path to the project.
    """
    project_path = os.path.expanduser(project_path)
    pid = project_id(project_path)
    db_path = DATA_DIR / pid

    _stop_watcher(project_path)

    if not db_path.exists():
        return f"No index found for {project_path}"

    shutil.rmtree(db_path)

    # Remove from registry
    registry = _get_project_registry()
    registry.pop(project_path, None)
    _save_project_registry(registry)

    return f"Index removed for {project_path}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
