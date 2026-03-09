import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def sample_project(tmp_path):
    """Create a temporary project with sample files."""
    # Python file
    (tmp_path / "main.py").write_text(
        'def hello_world():\n    """Greet the world."""\n    print("Hello, world!")\n\n'
        'def add_numbers(a: int, b: int) -> int:\n    """Add two numbers together."""\n    return a + b\n'
    )

    # Go file
    (tmp_path / "server.go").write_text(
        'package main\n\nimport "net/http"\n\n'
        'func healthCheck(w http.ResponseWriter, r *http.Request) {\n'
        '    w.WriteHeader(http.StatusOK)\n'
        '    w.Write([]byte("ok"))\n}\n\n'
        'func main() {\n    http.HandleFunc("/health", healthCheck)\n'
        '    http.ListenAndServe(":8080", nil)\n}\n'
    )

    # YAML config
    (tmp_path / "config.yaml").write_text(
        "database:\n  host: localhost\n  port: 5432\n  name: myapp\n\n"
        "redis:\n  host: localhost\n  port: 6379\n"
    )

    # Markdown doc
    (tmp_path / "README.md").write_text(
        "# My Project\n\nA sample project for testing semantic search.\n\n"
        "## Features\n\n- Fast\n- Reliable\n- Easy to use\n"
    )

    # Nested directory
    src = tmp_path / "src"
    src.mkdir()
    (src / "utils.py").write_text(
        'import hashlib\n\n'
        'def compute_hash(data: bytes) -> str:\n'
        '    """Compute SHA256 hash of data."""\n'
        '    return hashlib.sha256(data).hexdigest()\n\n'
        'def validate_email(email: str) -> bool:\n'
        '    """Check if email format is valid."""\n'
        '    return "@" in email and "." in email.split("@")[1]\n'
    )

    # File that should be skipped
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion": 3}')

    # Directory that should be skipped
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "something.js").write_text("module.exports = {}")

    # .git dir should be skipped
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]")

    return tmp_path


@pytest.fixture
def empty_project(tmp_path):
    """Create an empty temporary project."""
    return tmp_path


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Use a temporary data dir for all tests."""
    data_dir = tmp_path / "ccl-data"
    data_dir.mkdir()
    monkeypatch.setenv("CCL_DATA_DIR", str(data_dir))

    # Re-import to pick up new env var
    import claude_context_local.server as srv
    monkeypatch.setattr(srv, "DATA_DIR", Path(str(data_dir)))
