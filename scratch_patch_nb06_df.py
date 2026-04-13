import json
from pathlib import Path

nb_path = Path(r"d:\Projects\rep-learning-amazon-reviews\notebooks\06_latent_space_analysis.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

OLD = """val_df['umap1'] = embedding[:, 0]
val_df['umap2'] = embedding[:, 1]"""

NEW = """umap_sample_df['umap1'] = embedding[:, 0]
umap_sample_df['umap2'] = embedding[:, 1]"""

OLD_PLOT = """    data=val_df,
    x='umap1',
    y='umap2',"""

NEW_PLOT = """    data=umap_sample_df,
    x='umap1',
    y='umap2',"""

changed = 0
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    modified = False
    
    if OLD in src:
        src = src.replace(OLD, NEW)
        modified = True
    if OLD_PLOT in src:
        src = src.replace(OLD_PLOT, NEW_PLOT)
        modified = True
        
    if modified:
        lines = src.splitlines(keepends=True)
        if lines and lines[-1].endswith("\n"):
            lines[-1] = lines[-1].rstrip("\n")
        cell["source"] = lines
        changed += 1

print(f"Changed {changed} cell(s) in NB06")
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Saved NB06")
