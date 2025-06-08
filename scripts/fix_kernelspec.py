import nbformat
from glob import glob

# Define the kernelspec metadata to inject
kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}
language_info = {"name": "python"}

# Iterate over all notebooks in the notebooks directory
for nb_path in glob('notebooks/*.ipynb'):
    nb = nbformat.read(nb_path, as_version=4)
    # Inject metadata
    nb.metadata.kernelspec = kernelspec
    nb.metadata.language_info = language_info
    # Write back modifications
    nbformat.write(nb, nb_path)
    print(f"Injected kernelspec into {nb_path}") 