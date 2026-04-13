import json
with open(r'd:\Projects\rep-learning-amazon-reviews\notebooks\06_latent_space_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
errors = 0
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        for o in cell.get('outputs', []):
            if o.get('output_type') == 'error':
                print(f"\n--- ERROR IN CELL index {i} ---")
                print(f"{o['ename']}: {o['evalue']}")
                print('\n'.join(o.get('traceback', [])))
                errors += 1
if errors == 0:
    print('No errors found in the saved notebook outputs.')
