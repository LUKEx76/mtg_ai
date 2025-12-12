# MTG AI

## Ideas
- Train embeddings on cards
  - Display card graph with similar cards
- Deck bracket classifier (regressor)
  - Train with edhrec decks against their bracket label
  - Approach 1: Use card embeddings to feed into transformer model
  - Approach 2: Use XGBoost Regression Tree with one-hot encoded input and engineer features (combo, game changer, ...)
- _for the future:_ Deck builder, build deck of given power level, build best "bang for your buck" deck, ...

## Approach
- Use json from scryfall for basic card information
- Scrape edhrec for decklists and further information about cards, decks and commanders
- Train embeddings with relevant features of a card
- Display interactable graph of similar cards
- Train regressor to determine bracket/power level of a deck
- _for the future:_ Train soft-clustering (Amount of tags as clusters [Token, crabs, theft, ...])
- _for the future:_ Measure for synergy between two cards: How often do 2 cards appear in the same deck divided/minus how often does it appear in general


## TODOs:
- Use latest version of oracle text
- Handle two faced cards
- Add planeswalkers that can be commanders to the scraping
- Train new embedding that takes more than oracle text into account
- Scrape decks from edhrec
- Train bracket regressor
- Create interactive graph of similar cards


### Remarks
- Only commander-legal cards are considered
- Be aware of partner commanders
- Normalization for oracle text could improve (Planeswalkers, TREASURE TOKEN, empty text, CARD_NAME, ...)
