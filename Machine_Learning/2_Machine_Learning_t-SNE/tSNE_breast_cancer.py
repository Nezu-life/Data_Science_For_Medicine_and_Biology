import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply the tSNE
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_scaled)

# Plot
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap="viridis")
plt.title('t-SNE of Breast Cancer Dataset')
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()