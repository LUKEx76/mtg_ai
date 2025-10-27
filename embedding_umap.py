import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap   # pip install umap-learn

DIMENSION = 128  # embedding dimension you trained
TOP = 1000       # number of cards to visualize
HIGHLIGHT_CARD = "Sol Ring"

# --- Load data ---
df = pd.read_parquet("data/cards_with_embeddings.parquet")

# remove double faced cards for now
mask_double_faced = df["name"].astype(str).str.contains("//", regex=False, na=False)
df_filtered = df[~mask_double_faced].copy()
print(f"Removed {mask_double_faced.sum()} double-faced or split cards")

# Keep only rows with valid EDHREC rank
df_ranked = df_filtered[pd.to_numeric(df_filtered["edhrec_rank"], errors="coerce").notna()].copy()
df_ranked["edhrec_rank"] = df_ranked["edhrec_rank"].astype(int)

# Take the TOP most played cards
top = df_ranked.nsmallest(TOP, "edhrec_rank").copy()

# Build a (TOP x DIMENSION) matrix of embeddings
X = np.vstack(top[f"embedding_{DIMENSION}"].to_numpy())

# --- UMAP dimensionality reduction ---
umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=30,        # like perplexity; controls local vs global balance
    min_dist=0.1,          # smaller = tighter clusters, larger = smoother map
    metric="cosine",       # good for normalized embeddings
    random_state=42,
)

X_2d = umap_model.fit_transform(X)
top["x"] = X_2d[:, 0]
top["y"] = X_2d[:, 1]

# --- Color setup ---
def color_id_to_str(cid):
    if isinstance(cid, (list, np.ndarray)):
        if len(cid) == 0:
            return "C"  # colorless
        return "".join(map(str, cid))
    if pd.isna(cid):
        return "C"
    return str(cid)

top["color_id_str"] = top["color_identity"].apply(color_id_to_str)

COLOR_MAP = {
    "W": np.array([0.95, 0.95, 0.85]),
    "U": np.array([0.20, 0.40, 0.80]),
    "B": np.array([0.05, 0.05, 0.05]),
    "R": np.array([0.80, 0.15, 0.15]),
    "G": np.array([0.10, 0.50, 0.15]),
    "C": np.array([0.50, 0.50, 0.50]),
}

def color_identity_to_rgb(cid):
    if isinstance(cid, (list, np.ndarray)):
        if len(cid) == 0:
            return tuple(COLOR_MAP["C"])
        basis = [COLOR_MAP[c] for c in cid]
    else:
        return tuple(COLOR_MAP["C"])
    avg = np.mean(basis, axis=0)
    return tuple(avg)

top["plot_color"] = top["color_identity"].apply(color_identity_to_rgb)

# --- Plot ---
plt.figure(figsize=(12, 9))
plt.scatter(
    top["x"],
    top["y"],
    c=top["plot_color"].tolist(),
    s=40,
    alpha=0.85,
    edgecolors="grey",
)

for _, row in top.iterrows():
    if row["name"] == HIGHLIGHT_CARD:
        plt.text(
            row["x"], row["y"], row["name"],
            fontsize=10, weight="bold", color="green", alpha=1.0
        )
    else:
        plt.text(
            row["x"], row["y"], row["name"],
            fontsize=6, alpha=0.7
        )


plt.title(f"Top {TOP} Commander cards (UMAP on oracle-text embedding)")
plt.xlabel("UMAP dim 1")
plt.ylabel("UMAP dim 2")
plt.tight_layout()
plt.show()
