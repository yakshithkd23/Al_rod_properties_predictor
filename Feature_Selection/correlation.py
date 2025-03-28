import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("processed_dataset.csv")

# Check for non-numeric values
print(df.dtypes)

# Convert categorical column "Processing" to numerical using Label Encoding
if "Processing" in df.columns:
    label_encoder = LabelEncoder()
    df["Processing"] = label_encoder.fit_transform(df["Processing"])

# Drop non-numeric values or handle 'outlier' text
df.replace("outlier", pd.NA, inplace=True)  # Replace with NaN
df.dropna(inplace=True)  # Remove rows with NaN values

# Define input features (X) and target variables (Y)
X = df.drop(columns=["Elongation (%)", "Tensile Strength (MPa)", "Yield Strength (MPa)"])  # Features
Y = df[["Elongation (%)", "Tensile Strength (MPa)", "Yield Strength (MPa)"]]  # Targets

corr_matrix = df.corr(numeric_only=True)

# Plot heatmap with improved visibility
plt.figure(figsize=(14, 10))  # Increase figure size
sns.heatmap(
    corr_matrix, 
    annot=True,               # Show values inside heatmap
    fmt=".2f",                # Limit decimal places
    cmap="coolwarm",          # Use a diverging colormap
    linewidths=0.5,           # Add grid lines for clarity
    linecolor="gray",         # Grid line color
    square=True,              # Make heatmap squares uniform
    annot_kws={"size": 10},   # Increase annotation font size
    cbar_kws={"shrink": 0.8, "aspect": 30}  # Adjust color bar size
)

# Improve label readability
plt.xticks(rotation=45, fontsize=12)  # Rotate x-axis labels
plt.yticks(rotation=0, fontsize=12)   # Keep y-axis labels horizontal
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")  # Add title

# Show the heatmap
plt.show()
