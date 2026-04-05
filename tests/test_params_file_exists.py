from pathlib import Path
import yaml


def test_params_file_exists_and_has_categories():
    root = Path(__file__).resolve().parents[1]
    params_path = root / "params.yaml"
    assert params_path.exists()

    data = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    assert "data" in data
    assert "categories" in data["data"]
    assert len(data["data"]["categories"]) == 4
