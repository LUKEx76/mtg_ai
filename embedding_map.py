import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

DIMENSION = 128 # out of [64, 128]
TOP = 2000

df = pd.read_parquet("data/cards_with_embeddings.parquet")

# keep only rows with a valid rank
df_ranked = df[
    pd.to_numeric(df["edhrec_rank"], errors="coerce").notna()
].copy()
df_ranked["edhrec_rank"] = df_ranked["edhrec_rank"].astype(int)

# take the 500 most played cards
top = df_ranked.nsmallest(TOP, "edhrec_rank").copy()

X = np.vstack(top[f"embedding_{DIMENSION}"].to_numpy())

tsne = TSNE(
    n_components=2,
    perplexity=30,      # ~how many neighbors each point cares about
    learning_rate="auto",
    init="random",
    random_state=42,
)

X_2d = tsne.fit_transform(X)

top["x"] = X_2d[:, 0]
top["y"] = X_2d[:, 1]

# plotting
def color_id_to_str(cid):
    if isinstance(cid, (list, np.ndarray)):
        if len(cid) == 0:
            return "C"  # colorless
        return "".join(map(str, cid))
    if pd.isna(cid):
        return "C"
    return str(cid)


top["color_id_str"] = top["color_identity"].apply(color_id_to_str)

plt.figure(figsize=(10, 8))

# base colors for each identity symbol
COLOR_MAP = {
    "W": np.array([0.95, 0.95, 0.85]),  # off-white / pale yellowish
    "U": np.array([0.20, 0.40, 0.80]),  # blue
    "B": np.array([0.05, 0.05, 0.05]),  # near black
    "R": np.array([0.80, 0.15, 0.15]),  # red
    "G": np.array([0.10, 0.50, 0.15]),  # green
    "C": np.array([0.50, 0.50, 0.50]),  # colorless / artifacts
}

def color_identity_to_rgb(cid):
    # cid can be ['U', 'B'] or [] or np.array([...]) or NaN
    if isinstance(cid, (list, np.ndarray)):
        if len(cid) == 0:
            # colorless
            return tuple(COLOR_MAP["C"])
        basis = [COLOR_MAP[c] for c in cid]
    else:
        # handle NaN or weird cases
        return tuple(COLOR_MAP["C"])
    
    avg = np.mean(basis, axis=0)
    return tuple(avg)

top["plot_color"] = top["color_identity"].apply(color_identity_to_rgb)

plt.scatter(
    top["x"],
    top["y"],
    c=top["plot_color"].tolist(),
    s=40,
    alpha=0.8,
    edgecolors="grey"
)

for _, row in top.iterrows():
    plt.text(
        row["x"], row["y"], row["name"],
        fontsize=6, alpha=0.7
    )

plt.title(f"Top {TOP} most popular Commander cards (t-SNE on oracle text embedding)")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.tight_layout()
plt.show()
