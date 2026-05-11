import json
import re

def convert_py_to_ipynb(py_file, ipynb_file):
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()

    cells = []
    blocks = re.split(r'^# %%\s*', content, flags=re.MULTILINE)
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        if block.startswith('"""') and block.endswith('"""'):
            md = block.strip('"\n ')
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + '\n' for line in md.splitlines()]
            })
        else:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + '\n' for line in block.splitlines()]
            })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(ipynb_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

if __name__ == "__main__":
    convert_py_to_ipynb('spotify_analysis.py', 'spotify_pipeline.ipynb')
    print("Notebook generated successfully!")
