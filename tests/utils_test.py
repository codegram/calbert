import pytest
from pathlib import Path

from calbert import utils


@pytest.mark.describe("utils.path_to_str")
class TestUtils:
    @pytest.mark.it("Returns an absolute path as a str")
    def test_path_to_str(self):
        assert utils.path_to_str(Path("README.md")).startswith("/")
