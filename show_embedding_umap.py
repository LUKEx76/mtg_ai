import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

HIGHLIGHT_CARD = "Goblin King"

# 1. Load your processed UMAP data
print("Loading UMAP data...")
df = pd.read_parquet("data/cards_with_umap_coords.parquet")

# 2. Load Scryfall data (Only columns we need)
print("Loading Scryfall reference data...")
# Loading the full JSON can be heavy, so we extract just the map we need
scry_df = pd.read_json("data/scryfall/oracle-cards-20251022090300.json")

# --- IMPLEMENTATION START ---

# Create a lookup dictionary: Oracle ID -> Color Identity
# We use drop_duplicates just in case, though oracle-cards usually is unique.
print("Mapping color identities...")
oracle_id_to_ci = scry_df.drop_duplicates('oracle_id').set_index('oracle_id')['color_identity'].to_dict()

# Map the values to your main dataframe
# If an oracle_id isn't found, it defaults to NaN
df['color_identity'] = df['oracle_id'].map(oracle_id_to_ci)

# Fill missing values with empty lists (treat as Colorless)
df['color_identity'] = df['color_identity'].apply(lambda x: x if isinstance(x, list) else [])

# --- IMPLEMENTATION END ---


# --- Color setup ---
def color_id_to_str(cid):
    if isinstance(cid, (list, np.ndarray)):
        if len(cid) == 0:
            return "C"  # colorless
        return "".join(map(str, cid))
    if pd.isna(cid):
        return "C"
    return str(cid)

df["color_id_str"] = df["color_identity"].apply(color_id_to_str)

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
        basis = [COLOR_MAP.get(c, COLOR_MAP["C"]) for c in cid]
    else:
        return tuple(COLOR_MAP["C"])
    
    # Average the colors (e.g., Blue + Red = Purple-ish)
    avg = np.mean(basis, axis=0)
    return tuple(avg)

print("calculating plot colors...")
df["plot_color"] = df["color_identity"].apply(color_identity_to_rgb)

# --- Plot ---
print("Rendering plot...")
plt.figure(figsize=(12, 9))
plt.scatter(
    df["x"],
    df["y"],
    c=df["plot_color"].tolist(),
    s=10, # Reduced size slightly for clarity with 30k points
    alpha=0.6,
    edgecolors="none", # Removed edgecolors for speed
)

# Optimized Text Labeling
# Plotting 30k text labels will freeze matplotlib. 
# We plot the HIGHLIGHT + a random sample of context cards.
texts_to_plot = []

# 1. Find the Highlight Card
highlight_row = df[df["name"] == HIGHLIGHT_CARD]
if not highlight_row.empty:
    r = highlight_row.iloc[0]
    plt.scatter(r["x"], r["y"], c='yellow', s=200, marker='*', edgecolors='black', zorder=10)
    plt.text(
        r["x"], r["y"] + 0.5, r["name"],
        fontsize=12, weight="bold", color="black", 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )
    print(f"Found {HIGHLIGHT_CARD} at {r['x']}, {r['y']}")

# 2. Add some random context labels (e.g., 50 random cards)
sample_df = df# df.sample(50)
for _, row in sample_df.iterrows():
    plt.text(
        row["x"], row["y"], row["name"],
        fontsize=6, alpha=0.6, color='black'
    )

plt.title(f"Magic: The Gathering UMAP (Colors by Identity)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.tight_layout()
plt.show()