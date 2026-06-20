import pandas as pd
import matplotlib.pyplot as plt


file_path = "data/raw/parkinsons.data"

dataset = pd.read_csv(file_path)

status_counts = dataset["status"].value_counts().sort_index()

labels = ["Healthy", "Parkinson's Disease"]
values = [status_counts[0], status_counts[1]]

plt.figure(figsize=(6, 5))
plt.bar(labels, values)
plt.xlabel("Group")
plt.ylabel("Number of Samples")
plt.title("Sample Distribution in Parkinson's Voice Dataset")
plt.tight_layout()
plt.savefig("figures/sample_distribution.png", dpi=300)
plt.show()