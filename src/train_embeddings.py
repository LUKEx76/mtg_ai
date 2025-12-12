import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from scripts.oracle_text_normalizer import clean_oracle, normalize_cardname

SEED = 1893
DIMENSIONS = [64, 128]

def train_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df_result = df.copy()

    df["oracle_text_clean"] = df["oracle_text"].apply(clean_oracle)
    df["oracle_text_clean"] = df.apply(normalize_cardname, axis=1)

    df_result["oracle_text_clean"] = df["oracle_text_clean"]

    mask_has_text = df["oracle_text_clean"].notna() & (df["oracle_text_clean"].str.strip() != "")
    df_train = df.loc[mask_has_text].copy()

    for dim in DIMENSIONS:
        # Build pipeline fresh per dim
        tfidf = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),   # unigrams + bigrams
            min_df=2,
            max_df=0.8,
        )
        svd = TruncatedSVD(
            n_components=dim,
            random_state=SEED,
        )
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(tfidf, svd, normalizer)

        embeddings_train = lsa.fit_transform(df_train["oracle_text_clean"])

        zero_vec = np.zeros(dim)
        df_result[f"embedding_{dim}"] = [zero_vec] * len(df_result)

        emb_series = pd.Series(
            list(embeddings_train),
            index=df_train.index,
            dtype=object,
        )

        df_result.loc[df_train.index, f"embedding_{dim}"] = emb_series

        joblib.dump(lsa, f"oracletext_lsa_{dim}_model.joblib")


    print(f"Trained embeddings for dimensions: {DIMENSIONS}")
    return df_result