"""Tests for MCP tool functions (requires embedding model)."""

import pytest

from claude_context_local.server import (
    drop_index,
    index_project,
    index_status,
    search_code,
)


class TestIndexProject:
    def test_index_sample_project(self, sample_project):
        result = index_project(str(sample_project))
        assert "Indexed 5 files" in result
        assert "Total chunks in index:" in result

    def test_incremental_index(self, sample_project):
        index_project(str(sample_project))
        result = index_project(str(sample_project))
        assert "0 files" not in result or "unchanged" in result

    def test_force_reindex(self, sample_project):
        index_project(str(sample_project))
        result = index_project(str(sample_project), force=True)
        assert "Indexed 5 files" in result

    def test_nonexistent_path(self):
        result = index_project("/nonexistent/path")
        assert "Error" in result

    def test_incremental_detects_changes(self, sample_project):
        index_project(str(sample_project))

        # Modify a file
        (sample_project / "main.py").write_text("def changed(): pass\n")
        result = index_project(str(sample_project))
        assert "1 files" in result or "Indexed 1" in result


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
        # Should not return .go files
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


class TestIndexStatus:
    def test_not_indexed(self, empty_project):
        result = index_status(str(empty_project))
        assert "not indexed" in result

    def test_indexed_status(self, sample_project):
        index_project(str(sample_project))
        result = index_status(str(sample_project))
        assert "Indexed files: 5" in result
        assert "Total chunks:" in result
        assert "all-MiniLM-L6-v2" in result


class TestDropIndex:
    def test_drop_existing(self, sample_project):
        index_project(str(sample_project))
        result = drop_index(str(sample_project))
        assert "removed" in result

        # Verify it's gone
        status = index_status(str(sample_project))
        assert "not indexed" in status

    def test_drop_nonexistent(self, empty_project):
        result = drop_index(str(empty_project))
        assert "No index found" in result
