import json
from pathlib import Path

nb_path = Path(r"d:\Projects\rep-learning-amazon-reviews\notebooks\06_latent_space_analysis.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

OLD = """umap_sample_df = val_df.groupby("main_category").sample(
    n=min(2500, val_df.groupby("main_category").size().min()),
    random_state=cfg.seed,
).reset_index(drop=True)
umap_indices = umap_sample_df.index.to_numpy()
X_umap = X_norm[umap_sample_df.index.to_numpy()]"""

NEW = """umap_sample_df = val_df.groupby("main_category").sample(
    n=min(2500, val_df.groupby("main_category").size().min()),
    random_state=cfg.seed,
).sort_index()
umap_indices = umap_sample_df.index.to_numpy()
X_umap = X_norm[umap_indices]"""

changed = 0
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if OLD in src:
        new_src = src.replace(OLD, NEW)
        lines = new_src.splitlines(keepends=True)
        if lines and lines[-1].endswith("\n"):
            lines[-1] = lines[-1].rstrip("\n")
        cell["source"] = lines
        changed += 1

print(f"Changed {changed} cell(s) in NB06")
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Saved NB06")
