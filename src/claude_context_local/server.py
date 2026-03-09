#!/usr/bin/env python3
"""Local semantic code search MCP server.

Uses sentence-transformers for embeddings and ChromaDB for vector storage.
Per-project indexing with persistent storage.
"""

import hashlib
import logging
import os
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from mcp.server.fastmcp import FastMCP

# --- Config ---

MODEL_NAME = os.environ.get("CCL_MODEL", "all-MiniLM-L6-v2")
CHUNK_MAX_LINES = int(os.environ.get("CCL_CHUNK_LINES", "50"))
CHUNK_OVERLAP_LINES = int(os.environ.get("CCL_CHUNK_OVERLAP", "10"))
DATA_DIR = Path(os.environ.get("CCL_DATA_DIR", Path.home() / ".cache" / "claude-context-local"))

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
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )
    return client.get_or_create_collection(
        name="code",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def should_index(path: Path) -> bool:
    """Check if a file should be indexed."""
    if path.name in SKIP_FILES:
        return False
    if path.name == "Dockerfile" or path.name == "Makefile":
        return True
    if path.suffix.lower() in CODE_EXTENSIONS:
        return True
    return False


def walk_project(project_path: str):
    """Yield files to index."""
    root = Path(project_path)
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip dirs in-place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            fpath = Path(dirpath) / fname
            if should_index(fpath):
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

    Scans all code files, splits into chunks, and stores embeddings.
    Incremental by default — only re-indexes changed files.

    Args:
        project_path: Absolute path to project root directory.
        force: If True, re-index everything from scratch.
    """
    project_path = os.path.expanduser(project_path)
    if not os.path.isdir(project_path):
        return f"Error: {project_path} is not a directory"

    collection = get_collection(project_path)

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

    files = list(walk_project(project_path))
    indexed_files = set()
    new_count = 0
    skip_count = 0
    batch_ids = []
    batch_docs = []
    batch_metas = []

    t0 = time.time()

    for fpath in files:
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

        chunks = chunk_file(fpath, project_path)
        for chunk in chunks:
            chunk["metadata"]["hash"] = fh
            batch_ids.append(chunk["id"])
            batch_docs.append(chunk["document"])
            batch_metas.append(chunk["metadata"])

        indexed_files.add(rel)
        new_count += 1

        # Batch upsert every 100 chunks
        if len(batch_ids) >= 100:
            collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
            batch_ids, batch_docs, batch_metas = [], [], []

    # Flush remaining
    if batch_ids:
        collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

    elapsed = time.time() - t0
    total = collection.count()

    return (
        f"Indexed {new_count} files ({skip_count} unchanged) in {elapsed:.1f}s\n"
        f"Total chunks in index: {total}\n"
        f"Project: {project_path}"
    )


@mcp.tool()
def search_code(query: str, project_path: str, n_results: int = 10, file_filter: str = "") -> str:
    """Search code semantically across the indexed project.

    Uses hybrid search: embedding similarity + optional file path filter.

    Args:
        query: Natural language search query (e.g. "error handling in API routes").
        project_path: Absolute path to the indexed project.
        n_results: Number of results to return (default 10).
        file_filter: Optional glob-like filter for file paths (e.g. "*.py" or "src/").
    """
    project_path = os.path.expanduser(project_path)
    pid = project_id(project_path)
    db_path = DATA_DIR / pid
    if not db_path.exists():
        return "Project not indexed yet. Run index_project first."

    collection = get_collection(project_path)

    if collection.count() == 0:
        return "Project not indexed yet. Run index_project first."

    # Fetch more results than needed if filtering, then post-filter
    fetch_n = min(n_results, 20) if not file_filter else min(n_results * 5, 50)

    try:
        results = collection.query(
            query_texts=[query],
            n_results=fetch_n,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        return f"Search error: {e}"

    if not results["documents"] or not results["documents"][0]:
        return "No results found."

    # Post-filter by file path
    def matches_filter(filepath: str) -> bool:
        if not file_filter:
            return True
        if file_filter.startswith("*."):
            return filepath.endswith(file_filter[1:])
        return file_filter in filepath

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if not matches_filter(meta["file"]):
            continue
        if len(output) >= n_results:
            break
        similarity = 1 - dist  # cosine distance to similarity
        header = f"### {meta['file']}:{meta['start_line']}-{meta['end_line']} (score: {similarity:.3f})"
        output.append(f"{header}\n```\n{doc}\n```\n")

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
        return (
            f"Project: {project_path}\n"
            f"Indexed files: {len(files)}\n"
            f"Total chunks: {count}\n"
            f"Model: {MODEL_NAME}"
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

    import shutil
    shutil.rmtree(db_path)
    return f"Index removed for {project_path}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
