import pandas as pd
import matplotlib.pyplot as plt

# Example molecular dataset
data = {
    "Molecule": ["A", "B", "C", "D", "E"],
    "MolecularWeight": [180.2, 250.4, 310.1, 150.3, 420.6],
    "BindingScore": [0.65, 0.82, 0.71, 0.55, 0.91]
}

df = pd.DataFrame(data)

print("Molecular Dataset:")
print(df)

# Simple analysis
avg_binding = df["BindingScore"].mean()
print("\nAverage Binding Score:", avg_binding)

# Visualization
plt.scatter(df["MolecularWeight"], df["BindingScore"])
plt.xlabel("Molecular Weight")
plt.ylabel("Binding Score")
plt.title("Molecule Binding Analysis")
plt.show()
