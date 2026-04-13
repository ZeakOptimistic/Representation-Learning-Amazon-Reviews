import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

nb_path = Path(r"d:\Projects\rep-learning-amazon-reviews\notebooks\06_latent_space_analysis.ipynb")
with open(nb_path, encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

print("Executing Notebook 6...")
try:
    ep.preprocess(nb, {'metadata': {'path': str(nb_path.parent)}})
    print("Notebook executed successfully.")
except Exception as e:
    print(f"\nNotebook execution failed: {e}")
    # Print the specific error from the notebook cells
    for cell in nb.cells:
        if cell.cell_type == 'code':
            for output in cell.outputs:
                if output.output_type == 'error':
                    print("\n--- ERROR DETAILS ---")
                    print(f"Error Name: {output.ename}")
                    print(f"Error Value: {output.evalue}")
                    print("Traceback:")
                    print('\n'.join(output.traceback))
