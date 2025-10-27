import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from oracle_text_normalizer import clean_oracle, normalize_cardname

DIMENSIONS = 64

# 1. Load data and build cleaned oracle text
df = pd.read_parquet("data/cards_commander.parquet")

df["oracle_text_clean"] = df["oracle_text"].apply(clean_oracle)
df["oracle_text_clean"] = df.apply(normalize_cardname, axis=1)

# keep only rows that actually have text for training
df_train = df[df["oracle_text_clean"].notna() & (df["oracle_text_clean"].str.strip() != "")].copy()

# 2. Build TF-IDF -> SVD -> Normalizer pipeline
tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),   # unigrams + bigrams
    min_df=2,             # ignore super rare phrases
    max_df=0.8,           # ignore extremely common boilerplate-ish stuff
)

svd = TruncatedSVD(
    n_components=DIMENSIONS,
    random_state=1893
)

normalizer = Normalizer(copy=False)
lsa = make_pipeline(tfidf, svd, normalizer)

# 3. FIT the pipeline, and get embeddings for the training rows
embeddings_train = lsa.fit_transform(df_train["oracle_text_clean"])

# 4. Create zero embeddings for cards with no oracle text
df_textless = df[df["oracle_text_clean"].isna() | (df["oracle_text_clean"].str.strip() == "")].copy()

zero_vec = np.zeros(DIMENSIONS)
df_textless[f"embedding_{DIMENSIONS}"] = [zero_vec] * len(df_textless)

# 5. Attach embeddings to df_train as well
df_train = df_train.copy()
df_train[f"embedding_{DIMENSIONS}"] = list(embeddings_train)

# 6. Combine back together so you keep ALL cards
df_all = pd.concat([df_train, df_textless], ignore_index=True)

print("Training embeddings shape:", embeddings_train.shape)
print("Final df_all rows:", len(df_all))

# 7. Save the fitted model + the dataframe with embeddings
joblib.dump(lsa, f"oracletext_lsa_{DIMENSIONS}_model.joblib")
df_all.to_parquet(f"cards_with_embeddings_{DIMENSIONS}.parquet", index=False)
