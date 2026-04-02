import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Load wine dataset
wine_data = load_wine(as_frame=True)
X = wine_data.data  # Feature data
y = wine_data.target  # Label data
feature_names = wine_data.feature_names  # Feature names
target_names = wine_data.target_names  # Class names

# Create figure
plt.figure(figsize=(12, 8))

# Set different colors and markers for each class
colors = ['red', 'blue', 'green']
markers = ['o', 's', '^']

# Plot scatter points for each class - corrected version
for i in range(3):  # 3 classes
    # Select samples for current class - using correct indexing method
    mask = (y == i)
    plt.scatter(X.iloc[mask.values, 0],  # First feature (alcohol)
               X.iloc[mask.values, 1],  # Second feature (malic_acid)
               c=colors[i],
               marker=markers[i],
               label=f'Class {i} ({target_names[i]})',
               s=60,  # Point size
               alpha=0.7,  # Transparency
               edgecolors='black',  # Point edge color
               linewidth=0.5)

# Set graph properties
plt.xlabel(feature_names[0] + ' (Feature 1)', fontsize=12)
plt.ylabel(feature_names[1] + ' (Feature 2)', fontsize=12)
plt.title('Wine Dataset: First Two Features Visualization\n(Colored by Class Labels)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Add statistical information
plt.text(0.02, 0.98, f'Total samples: {len(X)}',
         transform=plt.gca().transAxes, fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Display graph
plt.tight_layout()
plt.show()

# Print basic information
print("Dataset Basic Information:")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of classes: {len(set(y))}")
print(f"\nFirst two feature names:")
print(f"Feature 1: {feature_names[0]}")
print(f"Feature 2: {feature_names[1]}")
print(f"\nClass names: {list(target_names)}")

# Display first few samples' data
print(f"\nFirst 5 samples' first two feature values:")
print(X.iloc[:5, [0, 1]])
