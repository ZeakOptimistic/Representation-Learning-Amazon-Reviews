from pathlib import Path


def test_expected_top_level_directories_exist():
    expected = [
        ".github",
        "artifacts",
        "configs",
        "data",
        "docs",
        "experiments",
        "notebooks",
        "references",
        "reports",
        "src",
        "tests",
    ]
    root = Path(__file__).resolve().parents[1]
    for name in expected:
        assert (root / name).exists(), f"Missing expected path: {name}"
