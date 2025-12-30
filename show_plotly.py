import numpy as np
import plotly.express as px
import pandas as pd

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

# --- Color Helper (Same logic as before, but adapted for Plotly) ---
# Plotly prefers hex codes or standard names over RGB tuples
def get_hex_color(cid):
    # Simple mapping for major colors
    if not isinstance(cid, list) or len(cid) == 0: return "#888888" # Colorless/Grey
    if len(cid) > 1: return "#D4AF37" # Gold/Multicolor
    
    mapping = {
        "W": "#F0E68C", # Khaki (White-ish)
        "U": "#1E90FF", # Dodger Blue
        "B": "#2F4F4F", # Dark Slate Grey (Black-ish)
        "R": "#DC143C", # Crimson
        "G": "#228B22", # Forest Green
    }
    return mapping.get(cid[0], "#888888")

# If you haven't mapped color_identity yet, do it here or assume it's done
# df['color_hex'] = df['color_identity'].apply(get_hex_color) 
# For now, let's just color by generic 'type' if color isn't ready, or use a column you have.
# Assuming you have the 'color_identity' from the previous script:
df['color_hex'] = df['color_identity'].apply(get_hex_color)

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

# --- The Interactive Plot ---
fig = px.scatter(
    df, 
    x='x', 
    y='y', 
    hover_name='name',  # This makes the name appear on hover!
    hover_data=['mana_cost', 'type_line'], # Extra info in tooltip
    color='color_id_str', # This automatically creates a legend
    title="Magic: The Gathering UMAP",
    render_mode='webgl' # Crucial for 30k points speed
)

# Customize the look to be dark mode (easier on eyes for maps)
fig.update_traces(marker=dict(size=5, opacity=0.7))
fig.update_layout(
    template='plotly_dark',
    dragmode='pan', # Default mouse tool is Pan instead of Zoom
    width=1200,
    height=800
)

# Show it!
fig.show()

# Or save it to share with friends/keep
fig.write_html("mtg_interactive_map.html")