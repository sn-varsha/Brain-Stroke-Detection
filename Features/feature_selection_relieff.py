import pandas as pd
from sklearn.preprocessing import StandardScaler
from skrebate import ReliefF
import numpy as np

# 1. Read the data file
df = pd.read_csv("features_all_cleaned.csv")

# 2. Separate X and y
X = df.drop(columns=["label", "Disorder", "filename"])
# Keep only numeric columns
X = X.select_dtypes(include=[np.number])
y = df["label"]

# Remove columns with non-scalar values (if any)
X = X.loc[:, X.applymap(np.isscalar).all()]

# Fill or drop missing values
X = X.dropna()
y = df.loc[X.index, "label"]  # Keep labels in sync with X

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# 3. Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. ReliefF model
relief = ReliefF(n_neighbors=10)  # daha stabil sonuçlar için komşu sayısı
relief.fit(X_scaled, y)

# 5. Get the feature importance scores and sort them
feature_scores = pd.Series(relief.feature_importances_, index=X.columns)
feature_scores = feature_scores.sort_values(ascending=False)

# 6. Take the top N features (e.g., 15)
top_n = 15
top_features = feature_scores.head(top_n).index.tolist()

print("The best selected with ReliefF", top_n, "feature:")
for i, feat in enumerate(top_features, 1):
    print(f"{i}. {feat}")

# 7. Create a new DataFrame (filename, label, disease + selected features)
df_selected = pd.concat([
    df[["filename", "label", "Disorder"]],
    df[top_features]
], axis=1)

# 8. Save
df_selected.to_csv("features_selected_relieff.csv", index=False)
print("\nThe features selected with ReliefF were saved to the file 'features_selected_relieff.csv'.")