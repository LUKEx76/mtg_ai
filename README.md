# MTG AI: Commander Power Level Analysis & Card Embeddings

An AI-powered toolkit for analyzing Magic: The Gathering (MTG) cards and decks, specifically focusing on the Commander (EDH) format. This project combines data scraping, natural language processing, and machine learning to map card similarities and estimate deck power levels.

## Project Overview

MTG AI aims to bridge the gap between raw card data and functional gameplay analysis. By leveraging state-of-the-art NLP models and popularity data from EDHREC, the project provides a data-driven approach to understanding the "Commander meta."

### Key Features
*   **Semantic Card Embeddings:** Uses `Alibaba-NLP/gte-modernbert-base` to generate dense vector representations of card functionality.
*   **Interactive Card Maps:** UMAP-based dimensionality reduction to visualize "card space" and identify functionally similar cards.
*   **Power Level Regression:** An ensemble model (XGBoost + MLP) that predicts a card's "strength" or popularity bracket based on its attributes and rules text.
*   **Robust Data Pipelines:** Automated filtering of Scryfall bulk data and scraping of EDHREC statistics.

## Current Approach
*   **Data Source:** Use JSON from Scryfall for basic card information (filtered for `commander: legal`).
*   **Scraping:** Scrape EDHREC for decklists and detailed inclusion statistics for cards and commanders.
*   **Embeddings:** Train/Generate embeddings using normalized oracle text, mana costs, and type lines.
*   **Visualization:** Perform UMAP reduction to display an interactable graph of similar cards.
*   **Modeling:** Train a regressor to determine the bracket or power level of a deck based on card composition.

## Ideas & Future Work
*   **Soft-Clustering:** Group cards by archetypes (e.g., Tokens, Mill, Theft) based on their embedding distances.
*   **Synergy Metrics:** Measure synergy between cards by analyzing co-occurrence in decks vs. overall play rate.
*   **Deck Builder:** Create a tool to build decks of a specific power level or optimize for a "best bang for your buck" budget.
*   **Deck Bracket Classifier:**
    *   Train with EDHREC decks against their bracket label.
    *   *Approach 1:* Use card embeddings to feed into a transformer model.
    *   *Approach 2:* Use XGBoost Regression Tree with one-hot encoded input and engineered features (combo, game-changer, etc.).

## TODOs
- [ ] Add planeswalkers that can be commanders to the scraping logic.
- [ ] Implement full decklist scraping from EDHREC.
- [ ] Refine "game changer" and "combo" feature engineering.

## Remarks
*   **Legality:** Only commander-legal cards are currently considered.
*   **Partner Commanders:** Multi-commander setups (Partner, Friends Forever, etc.) require specialized handling in the power level model.
*   **Normalization:** Always use `src/oracle_text_normalizer.py` to ensure consistency across different card printings.
