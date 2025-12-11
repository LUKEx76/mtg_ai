# MTG ai

## Ideas
- Card graph
- Similar card finder
- Deck bracket classifier (regressor)
- Deck builder

## Approach
- Scrape decks on big fashion
  - First only edhrec (seems to have decks imported from other sources):
    - download json for each possible commander _done_
    - for each go through each deck and download the json for that
- Embedding with oracle text
- Train soft-clustering (Amount of tags as clusters [Token, crabs, theft, ...])
- Measure for synergy between two cards: How often do 2 cards appear in the same deck divided/minus how often does it appear in general

### Remarks
- Be aware of two faced cards (they do not have oracle text, instead you have to expand the faces)
- Normalization for oracle text could improve (Planeswalkers, TREASURE TOKEN, empty text, ...)
- Currently only commander-legal cards _(will stay that way)_
- Use the oracle text of the latest version of the card!!
- What about partner commanders?
