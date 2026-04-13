import json
with open(r'd:\Projects\rep-learning-amazon-reviews\notebooks\03_train_word2vec.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        for o in cell.get('outputs', []):
            if o.get('output_type') == 'error':
                print(f"ERROR IN CELL: {o['ename']}: {o['evalue']}")
                print('\n'.join(o.get('traceback', [])))
