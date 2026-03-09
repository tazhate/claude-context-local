#!/usr/bin/env python3
"""Local semantic code search MCP server.

Uses ChromaDB built-in ONNX embeddings and optional BM25 for hybrid search.
Per-project indexing with persistent storage.
"""

import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import chromadb
from mcp.server.fastmcp import FastMCP

# --- Config ---

MODEL_NAME = os.environ.get("CCL_MODEL", "all-MiniLM-L6-v2")
CHUNK_MAX_LINES = int(os.environ.get("CCL_CHUNK_LINES", "50"))
CHUNK_OVERLAP_LINES = int(os.environ.get("CCL_CHUNK_OVERLAP", "10"))
DATA_DIR = Path(os.environ.get("CCL_DATA_DIR", Path.home() / ".cache" / "claude-context-local"))
HYBRID_ALPHA = float(os.environ.get("CCL_HYBRID_ALPHA", "0.7"))  # 0=BM25 only, 1=semantic only

# File extensions to index
CODE_EXTENSIONS = {
    ".py", ".go", ".js", ".ts", ".tsx", ".jsx", ".rs", ".java", ".kt",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".scala",
    ".sh", ".bash", ".zsh", ".fish",
    ".yaml", ".yml", ".toml", ".json", ".hcl", ".tf",
    ".sql", ".graphql", ".proto",
    ".md", ".txt", ".rst",
    ".dockerfile", ".containerfile",
    ".html", ".css", ".scss", ".less",
    ".lua", ".zig", ".nim", ".ex", ".exs", ".erl",
    ".nix", ".dhall",
}

# Directories to always skip
SKIP_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".venv", "venv", "env", ".env",
    "vendor", "third_party",
    ".tox", ".eggs", "dist", "build",
    ".next", ".nuxt", ".output",
    "target",  # rust/java
    ".terraform",
    ".claude",
}

SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "go.sum", "Cargo.lock", "poetry.lock", "uv.lock",
    "Pipfile.lock", "composer.lock", "Gemfile.lock",
}

MAX_FILE_SIZE = 512 * 1024  # 512KB

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("claude-context-local")

mcp = FastMCP("claude-context-local")


# --- Embedding function ---

def get_embedding_function():
    """Get the embedding function based on config.

    Uses ChromaDB's built-in ONNX embedder by default (no PyTorch needed).
    Falls back to sentence-transformers if a custom model is specified.
    """
    if MODEL_NAME == "all-MiniLM-L6-v2":
        # Use ChromaDB's built-in ONNX embedder — no PyTorch, ~200MB RAM
        return chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
    else:
        # Custom model — requires sentence-transformers (pip install sentence-transformers)
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
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
        self.path.write_text(json.dumps({
            "docs": self.docs,
            "df": dict(self.df),
            "avgdl": self.avgdl,
        }))

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
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(self.avgdl, 1)))
                score += idf * tf_norm
            if score > 0:
                scores[doc_id] = score

        # Return top-k
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
            line.strip() for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]
    except OSError:
        return []


def matches_gitignore(rel_path: str, patterns: list[str]) -> bool:
    """Check if a relative path matches any gitignore pattern.

    Simple implementation covering common patterns:
    - exact name: "foo.txt"
    - directory: "build/"
    - extension: "*.pyc"
    - path prefix: "dist/"
    - negation: "!important.txt" (skip)
    """
    parts = rel_path.split("/")
    filename = parts[-1]

    for pattern in patterns:
        if pattern.startswith("!"):
            continue  # negation patterns not supported in simple mode

        # Remove trailing slash for directory patterns
        is_dir_pattern = pattern.endswith("/")
        p = pattern.rstrip("/")

        if p.startswith("**/"):
            # Match anywhere in path
            p = p[3:]
            if _simple_match(p, rel_path) or _simple_match(p, filename):
                return True
            for i in range(len(parts)):
                if _simple_match(p, "/".join(parts[i:])):
                    return True
        elif "/" in p:
            # Path pattern — match as prefix or exact
            if _simple_match(p, rel_path):
                return True
            if rel_path.startswith(p + "/"):
                return True
        elif is_dir_pattern:
            # Directory name match
            if p in parts[:-1]:
                return True
        else:
            # Filename or directory name match
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
        # Prune skip dirs in-place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            fpath = Path(dirpath) / fname
            if not should_index(fpath):
                continue

            # Check gitignore
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
    """Split a file into overlapping chunks with metadata."""
    try:
        content = filepath.read_text(errors="replace")
    except (OSError, UnicodeDecodeError):
        return []

    lines = content.splitlines()
    if not lines:
        return []

    rel_path = str(filepath.relative_to(project_root))
    chunks = []

    if len(lines) <= CHUNK_MAX_LINES:
        # Single chunk for small files
        chunks.append({
            "id": f"{rel_path}:0",
            "document": content,
            "metadata": {
                "file": rel_path,
                "start_line": 1,
                "end_line": len(lines),
                "total_lines": len(lines),
            },
        })
    else:
        # Sliding window
        start = 0
        while start < len(lines):
            end = min(start + CHUNK_MAX_LINES, len(lines))
            chunk_lines = lines[start:end]
            chunk_text = "\n".join(chunk_lines)

            chunks.append({
                "id": f"{rel_path}:{start}",
                "document": chunk_text,
                "metadata": {
                    "file": rel_path,
                    "start_line": start + 1,
                    "end_line": end,
                    "total_lines": len(lines),
                },
            })

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


# --- MCP Tools ---

@mcp.tool()
def index_project(project_path: str, force: bool = False) -> str:
    """Index or re-index a project for semantic code search.

    Scans all code files (respecting .gitignore), splits into chunks,
    and stores embeddings + BM25 index for hybrid search.
    Incremental by default — only re-indexes changed files.

    Args:
        project_path: Absolute path to project root directory.
        force: If True, re-index everything from scratch.
    """
    project_path = os.path.expanduser(project_path)
    if not os.path.isdir(project_path):
        return f"Error: {project_path} is not a directory"

    collection = get_collection(project_path)
    bm25 = get_bm25(project_path)

    if force:
        # Clear existing data
        pid = project_id(project_path)
        db_path = DATA_DIR / pid
        client = chromadb.PersistentClient(path=str(db_path))
        try:
            client.delete_collection("code")
        except Exception:
            pass
        collection = get_collection(project_path)
        bm25.clear()

    # Load .gitignore patterns
    gitignore_patterns = load_gitignore_patterns(project_path)

    # Load existing file hashes from metadata
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

        # Remove old chunks for this file
        try:
            old_ids = collection.get(
                where={"file": rel},
                include=[],
            )
            if old_ids and old_ids["ids"]:
                collection.delete(ids=old_ids["ids"])
        except Exception:
            pass
        bm25.remove_by_prefix(f"{rel}:")

        chunks = chunk_file(fpath, project_path)
        for chunk in chunks:
            chunk["metadata"]["hash"] = fh
            batch_ids.append(chunk["id"])
            batch_docs.append(chunk["document"])
            batch_metas.append(chunk["metadata"])
            bm25.add(chunk["id"], chunk["document"])

        indexed_files.add(rel)
        new_count += 1

        # Batch upsert every 100 chunks
        if len(batch_ids) >= 100:
            collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
            batch_ids, batch_docs, batch_metas = [], [], []

        # Progress logging every 10%
        progress = int((i + 1) / total_files * 100) if total_files else 100
        if progress >= last_progress + 10:
            log.info("Indexing: %d%% (%d/%d files)", progress, i + 1, total_files)
            last_progress = progress

    # Flush remaining
    if batch_ids:
        collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

    bm25.save()

    elapsed = time.time() - t0
    total = collection.count()
    gitignore_info = f"\n.gitignore: {len(gitignore_patterns)} patterns applied" if gitignore_patterns else ""

    return (
        f"Indexed {new_count} files ({skip_count} unchanged) in {elapsed:.1f}s\n"
        f"Total chunks in index: {total}\n"
        f"Project: {project_path}"
        f"{gitignore_info}"
    )


@mcp.tool()
def search_code(query: str, project_path: str, n_results: int = 10, file_filter: str = "") -> str:
    """Search code using hybrid semantic + keyword (BM25) search.

    Combines embedding similarity with BM25 keyword matching for better results.
    Use file_filter to restrict results to specific file types or directories.

    Args:
        query: Natural language or keyword search query.
        project_path: Absolute path to the indexed project.
        n_results: Number of results to return (default 10).
        file_filter: Optional filter for file paths (e.g. "*.py", "src/", "test").
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

    # Build semantic scores (normalize to 0-1)
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

    # Normalize BM25 scores to 0-1
    bm25_scores: dict[str, float] = {}
    if bm25_raw:
        max_bm25 = max(bm25_raw.values())
        if max_bm25 > 0:
            bm25_scores = {k: v / max_bm25 for k, v in bm25_raw.items()}

    # Resolve BM25 doc_ids to our format and fetch missing data
    bm25_mapped: dict[str, float] = {}
    for chunk_id, score in bm25_scores.items():
        # chunk_id format: "file:start_offset"
        parts = chunk_id.rsplit(":", 1)
        if len(parts) == 2:
            file_path, offset = parts[0], int(parts[1])
            doc_id = f"{file_path}:{offset + 1}"
            bm25_mapped[doc_id] = score
            # Fetch document data if not in semantic results
            if doc_id not in sem_data:
                try:
                    r = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
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

    # Sort by hybrid score
    ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

    # Post-filter by file path
    def matches_filter(filepath: str) -> bool:
        if not file_filter:
            return True
        if file_filter.startswith("*."):
            return filepath.endswith(file_filter[1:])
        return file_filter in filepath

    output = []
    for doc_id, score in ranked:
        if doc_id not in sem_data:
            continue
        doc, meta = sem_data[doc_id]
        if not matches_filter(meta["file"]):
            continue
        if len(output) >= n_results:
            break
        header = f"### {meta['file']}:{meta['start_line']}-{meta['end_line']} (score: {score:.3f})"
        output.append(f"{header}\n```\n{doc}\n```\n")

    if not output:
        return "No results found."

    return f"Found {len(output)} results for: {query}\n\n" + "\n".join(output)


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

    # Get unique files
    try:
        all_meta = collection.get(include=["metadatas"])
        files = set()
        if all_meta and all_meta["metadatas"]:
            for meta in all_meta["metadatas"]:
                if meta and "file" in meta:
                    files.add(meta["file"])

        bm25 = get_bm25(project_path)
        bm25._load()
        bm25_status = f"BM25 documents: {len(bm25.docs)}"

        return (
            f"Project: {project_path}\n"
            f"Indexed files: {len(files)}\n"
            f"Total chunks: {count}\n"
            f"{bm25_status}\n"
            f"Model: {MODEL_NAME}\n"
            f"Hybrid alpha: {HYBRID_ALPHA} (semantic={HYBRID_ALPHA}, bm25={1-HYBRID_ALPHA})"
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

    if not db_path.exists():
        return f"No index found for {project_path}"

    shutil.rmtree(db_path)
    return f"Index removed for {project_path}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
