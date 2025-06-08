import nbformat

# Read the notebook
nb_path = 'notebooks/intention_decision_curiosity_math.ipynb'
nb = nbformat.read(nb_path, as_version=4)

# Update title quotes in cells
for cell in nb.cells:
    if cell.cell_type == 'code':
        source = cell.source
        if "Probabilité d'intention selon l'attitude" in source:
            cell.source = source.replace(
                "plt.title('Probabilité d'intention selon l'attitude')",
                "plt.title(\"Probabilité d'intention selon l'attitude\")"
            )

# Write back changes
nbformat.write(nb, nb_path)
print(f"Fixed title quotes in {nb_path}") 