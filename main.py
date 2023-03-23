from runipy.notebook_runner import NotebookRunner
from IPython.nbformat.current import read

notebook = read(open("英文1800化学品.ipynb"), 'json')
r = NotebookRunner(notebook)
r.run_notebook()