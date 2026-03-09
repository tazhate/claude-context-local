"""Tests for MCP tool functions (requires embedding model)."""

import pytest

from claude_context_local.server import (
    drop_index,
    index_project,
    index_status,
    search_code,
    search_all,
    search_diff,
    find_symbol,
    _check_tree_sitter,
)


class TestIndexProject:
    def test_index_sample_project(self, sample_project):
        result = index_project(str(sample_project))
        assert "Indexed 5 files" in result
        assert "Total chunks in index:" in result

    def test_incremental_index(self, sample_project):
        index_project(str(sample_project))
        result = index_project(str(sample_project))
        assert "unchanged" in result

    def test_force_reindex(self, sample_project):
        index_project(str(sample_project))
        result = index_project(str(sample_project), force=True)
        # Force reindex should re-index all files (not skip any)
        assert "Indexed" in result
        assert "(0 unchanged)" in result or "0 unchanged" in result

    def test_nonexistent_path(self):
        result = index_project("/nonexistent/path")
        assert "Error" in result

    def test_incremental_detects_changes(self, sample_project):
        index_project(str(sample_project))
        (sample_project / "main.py").write_text("def changed(): pass\n")
        result = index_project(str(sample_project))
        assert "unchanged" in result
        assert "Indexed 0 files" not in result

    def test_gitignore_respected(self, sample_project):
        (sample_project / ".gitignore").write_text("*.go\n")
        result = index_project(str(sample_project), force=True)
        assert "Indexed 4 files" in result
        assert ".gitignore: 1 patterns" in result

    def test_ast_chunking_reported(self, sample_project):
        """v0.3.0: index should report AST-chunked file count."""
        result = index_project(str(sample_project))
        if _check_tree_sitter():
            assert "AST-chunked files:" in result
        else:
            assert "tree-sitter not installed" in result

    def test_symbol_graph_built(self, sample_project):
        """v0.3.0: indexing should build symbol graph."""
        index_project(str(sample_project))
        result = index_status(str(sample_project))
        assert "Symbol graph:" in result


class TestSearchCode:
    @pytest.fixture(autouse=True)
    def _index(self, sample_project):
        index_project(str(sample_project))
        self.project = sample_project

    def test_basic_search(self):
        result = search_code("hello world greeting", str(self.project))
        assert "Found" in result
        assert "main.py" in result

    def test_search_go_code(self):
        result = search_code("http health check endpoint", str(self.project))
        assert "Found" in result
        assert "server.go" in result

    def test_search_with_file_filter(self):
        result = search_code("function", str(self.project), file_filter="*.py")
        assert "Found" in result
        assert "server.go" not in result

    def test_search_with_path_filter(self):
        result = search_code("hash", str(self.project), file_filter="src/")
        assert "Found" in result
        assert "utils.py" in result

    def test_search_not_indexed(self, tmp_path):
        unindexed = tmp_path / "unindexed_project"
        unindexed.mkdir()
        result = search_code("anything", str(unindexed))
        assert "not indexed" in result

    def test_search_returns_scores(self):
        result = search_code("add numbers", str(self.project))
        assert "score:" in result

    def test_n_results_limit(self):
        result = search_code("code", str(self.project), n_results=2)
        assert result.count("###") <= 2

    def test_keyword_search_works(self):
        """BM25 should boost exact keyword matches."""
        result = search_code("compute_hash", str(self.project))
        assert "Found" in result
        assert "utils.py" in result

    def test_search_with_symbol_type_filter(self):
        """v0.3.0: filter results by symbol type."""
        if not _check_tree_sitter():
            return
        result = search_code("function", str(self.project), symbol_type="function")
        assert "Found" in result

    def test_search_with_context_lines(self):
        """v0.3.0: search results with surrounding context."""
        result = search_code("hello world", str(self.project), context_lines=3)
        assert "Found" in result

    def test_search_results_include_language(self):
        """v0.3.0: results should show language info."""
        if not _check_tree_sitter():
            return
        result = search_code("hello world greeting", str(self.project))
        assert "(python)" in result or "(go)" in result


class TestSearchAll:
    def test_search_across_projects(self, sample_project, tmp_path):
        """v0.3.0: multi-project search."""
        # Create second project
        proj2 = tmp_path / "project2"
        proj2.mkdir()
        (proj2 / "app.py").write_text("def unique_function(): return 42\n")

        index_project(str(sample_project))
        index_project(str(proj2))

        result = search_all("function")
        assert "Found" in result

    def test_no_projects(self):
        result = search_all("anything")
        assert "No projects indexed" in result


class TestSearchDiff:
    def test_not_a_git_repo(self, sample_project):
        result = search_diff(str(sample_project))
        # sample_project has a fake .git dir but isn't a real repo
        assert (
            "Error" in result or "not a git" in result or "No files changed" in result
        )

    def test_nonexistent_path(self):
        result = search_diff("/nonexistent/path")
        assert "Error" in result


class TestFindSymbol:
    def test_find_callers(self, sample_project):
        """v0.3.0: symbol graph — find callers."""
        index_project(str(sample_project))
        # Go file has healthCheck called from main
        if _check_tree_sitter():
            result = find_symbol(
                "healthCheck", str(sample_project), direction="callers"
            )
            # May or may not find callers depending on AST parsing depth
            assert "callers" in result.lower() or "no callers" in result.lower()

    def test_find_callees(self, sample_project):
        """v0.3.0: symbol graph — find callees."""
        index_project(str(sample_project))
        if _check_tree_sitter():
            result = find_symbol("main", str(sample_project), direction="callees")
            assert "called by" in result.lower() or "no callees" in result.lower()

    def test_invalid_direction(self, sample_project):
        result = find_symbol("foo", str(sample_project), direction="invalid")
        assert "Unknown direction" in result


class TestIndexStatus:
    def test_not_indexed(self, empty_project):
        result = index_status(str(empty_project))
        assert "not indexed" in result

    def test_indexed_status(self, sample_project):
        index_project(str(sample_project))
        result = index_status(str(sample_project))
        assert "Indexed files: 5" in result
        assert "Total chunks:" in result
        assert "BM25 documents:" in result
        assert "Hybrid alpha:" in result

    def test_status_includes_language_info(self, sample_project):
        """v0.3.0: status should show language breakdown."""
        index_project(str(sample_project))
        result = index_status(str(sample_project))
        assert "Languages:" in result

    def test_status_includes_symbol_graph(self, sample_project):
        """v0.3.0: status should show symbol graph info."""
        index_project(str(sample_project))
        result = index_status(str(sample_project))
        assert "Symbol graph:" in result

    def test_status_includes_watcher(self, sample_project):
        """v0.3.0: status should show watcher status."""
        index_project(str(sample_project))
        result = index_status(str(sample_project))
        assert "File watcher:" in result


class TestDropIndex:
    def test_drop_existing(self, sample_project):
        index_project(str(sample_project))
        result = drop_index(str(sample_project))
        assert "removed" in result

        status = index_status(str(sample_project))
        assert "not indexed" in status

    def test_drop_nonexistent(self, empty_project):
        result = drop_index(str(empty_project))
        assert "No index found" in result
