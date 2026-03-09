"""Tests for helper functions (no embedding model needed)."""

from pathlib import Path

from claude_context_local.server import (
    chunk_file,
    load_gitignore_patterns,
    matches_gitignore,
    project_id,
    should_index,
    walk_project,
    file_hash,
    tokenize,
    BM25Index,
)


class TestProjectId:
    def test_stable_hash(self):
        assert project_id("/home/user/project") == project_id("/home/user/project")

    def test_different_paths_different_ids(self):
        assert project_id("/path/a") != project_id("/path/b")

    def test_returns_16_chars(self):
        assert len(project_id("/any/path")) == 16


class TestShouldIndex:
    def test_python_file(self):
        assert should_index(Path("main.py"))

    def test_go_file(self):
        assert should_index(Path("server.go"))

    def test_typescript_file(self):
        assert should_index(Path("app.tsx"))

    def test_yaml_file(self):
        assert should_index(Path("config.yaml"))

    def test_dockerfile(self):
        assert should_index(Path("Dockerfile"))

    def test_makefile(self):
        assert should_index(Path("Makefile"))

    def test_lock_file_skipped(self):
        assert not should_index(Path("package-lock.json"))
        assert not should_index(Path("yarn.lock"))
        assert not should_index(Path("go.sum"))

    def test_binary_skipped(self):
        assert not should_index(Path("image.png"))
        assert not should_index(Path("binary.exe"))

    def test_case_insensitive_extension(self):
        assert should_index(Path("file.PY"))
        assert should_index(Path("file.Yaml"))


class TestWalkProject:
    def test_finds_code_files(self, sample_project):
        files = list(walk_project(str(sample_project)))
        names = {f.name for f in files}
        assert "main.py" in names
        assert "server.go" in names
        assert "config.yaml" in names
        assert "utils.py" in names

    def test_skips_lock_files(self, sample_project):
        files = list(walk_project(str(sample_project)))
        names = {f.name for f in files}
        assert "package-lock.json" not in names

    def test_skips_node_modules(self, sample_project):
        files = list(walk_project(str(sample_project)))
        names = {f.name for f in files}
        assert "something.js" not in names

    def test_skips_git_dir(self, sample_project):
        files = list(walk_project(str(sample_project)))
        paths = {str(f) for f in files}
        assert not any(".git" in p.split("/") for p in paths)

    def test_empty_project(self, empty_project):
        files = list(walk_project(str(empty_project)))
        assert files == []

    def test_skips_large_files(self, sample_project):
        large_file = sample_project / "huge.py"
        large_file.write_text("x" * (512 * 1024 + 1))
        files = list(walk_project(str(sample_project)))
        names = {f.name for f in files}
        assert "huge.py" not in names

    def test_skips_empty_files(self, sample_project):
        empty = sample_project / "empty.py"
        empty.write_text("")
        files = list(walk_project(str(sample_project)))
        names = {f.name for f in files}
        assert "empty.py" not in names

    def test_respects_gitignore(self, sample_project):
        # Create .gitignore
        (sample_project / ".gitignore").write_text("*.go\nsrc/\n")
        patterns = load_gitignore_patterns(str(sample_project))
        files = list(walk_project(str(sample_project), patterns))
        names = {f.name for f in files}
        assert "server.go" not in names
        assert "utils.py" not in names
        assert "main.py" in names


class TestGitignore:
    def test_load_patterns(self, sample_project):
        (sample_project / ".gitignore").write_text("*.pyc\n# comment\n\nbuild/\n")
        patterns = load_gitignore_patterns(str(sample_project))
        assert patterns == ["*.pyc", "build/"]

    def test_no_gitignore(self, empty_project):
        patterns = load_gitignore_patterns(str(empty_project))
        assert patterns == []

    def test_extension_match(self):
        assert matches_gitignore("main.pyc", ["*.pyc"])
        assert not matches_gitignore("main.py", ["*.pyc"])

    def test_directory_match(self):
        assert matches_gitignore("build/output.js", ["build/"])
        assert not matches_gitignore("build.py", ["build/"])

    def test_exact_name_match(self):
        assert matches_gitignore("src/foo.log", ["*.log"])

    def test_path_pattern(self):
        assert matches_gitignore("src/generated/api.py", ["src/generated"])

    def test_doublestar(self):
        assert matches_gitignore("deep/nested/test.pyc", ["**/*.pyc"])


class TestChunkFile:
    def test_small_file_single_chunk(self, sample_project):
        chunks = chunk_file(sample_project / "main.py", str(sample_project))
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["file"] == "main.py"
        assert chunks[0]["metadata"]["start_line"] == 1

    def test_chunk_metadata(self, sample_project):
        chunks = chunk_file(sample_project / "main.py", str(sample_project))
        meta = chunks[0]["metadata"]
        assert "file" in meta
        assert "start_line" in meta
        assert "end_line" in meta
        assert "total_lines" in meta

    def test_large_file_multiple_chunks(self, sample_project):
        big = sample_project / "big.py"
        big.write_text("\n".join(f"line_{i} = {i}" for i in range(120)))
        chunks = chunk_file(big, str(sample_project))
        assert len(chunks) > 1

        # Check overlap
        first_end = chunks[0]["metadata"]["end_line"]
        second_start = chunks[1]["metadata"]["start_line"]
        assert second_start < first_end

    def test_nonexistent_file(self, sample_project):
        chunks = chunk_file(sample_project / "nope.py", str(sample_project))
        assert chunks == []

    def test_chunk_ids_unique(self, sample_project):
        big = sample_project / "big.py"
        big.write_text("\n".join(f"line_{i}" for i in range(200)))
        chunks = chunk_file(big, str(sample_project))
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids))


class TestFileHash:
    def test_consistent_hash(self, sample_project):
        h1 = file_hash(sample_project / "main.py")
        h2 = file_hash(sample_project / "main.py")
        assert h1 == h2
        assert len(h1) == 32

    def test_different_files_different_hash(self, sample_project):
        h1 = file_hash(sample_project / "main.py")
        h2 = file_hash(sample_project / "server.go")
        assert h1 != h2

    def test_nonexistent_file(self, sample_project):
        assert file_hash(sample_project / "nope.py") == ""


class TestTokenize:
    def test_basic(self):
        tokens = tokenize("def hello_world(): pass")
        assert "def" in tokens
        assert "hello_world" in tokens
        assert "pass" in tokens

    def test_numbers(self):
        tokens = tokenize("x = 42 + y")
        assert "42" in tokens
        assert "x" in tokens

    def test_camelcase(self):
        tokens = tokenize("getUserById")
        assert "getuserbyid" in tokens

    def test_empty(self):
        assert tokenize("") == []
        assert tokenize("   ") == []


class TestBM25Index:
    def test_add_and_search(self, tmp_path):
        bm25 = BM25Index(tmp_path)
        bm25.add("doc1", "def hello_world(): print('hello')")
        bm25.add("doc2", "class UserAuth: def login(): pass")
        bm25.add("doc3", "def compute_hash(data): return sha256(data)")

        results = bm25.search("hello world")
        assert "doc1" in results
        assert results["doc1"] > results.get("doc2", 0)

    def test_remove(self, tmp_path):
        bm25 = BM25Index(tmp_path)
        bm25.add("doc1", "hello world")
        bm25.remove("doc1")
        results = bm25.search("hello")
        assert len(results) == 0

    def test_remove_by_prefix(self, tmp_path):
        bm25 = BM25Index(tmp_path)
        bm25.add("file.py:0", "first chunk")
        bm25.add("file.py:50", "second chunk")
        bm25.add("other.py:0", "other file")
        bm25.remove_by_prefix("file.py:")
        assert len(bm25.docs) == 1

    def test_persistence(self, tmp_path):
        bm25 = BM25Index(tmp_path)
        bm25.add("doc1", "persistent data")
        bm25.save()

        bm25_2 = BM25Index(tmp_path)
        results = bm25_2.search("persistent")
        assert "doc1" in results

    def test_clear(self, tmp_path):
        bm25 = BM25Index(tmp_path)
        bm25.add("doc1", "data")
        bm25.save()
        bm25.clear()
        assert len(bm25.docs) == 0

    def test_empty_search(self, tmp_path):
        bm25 = BM25Index(tmp_path)
        results = bm25.search("anything")
        assert results == {}
