# MTG ai

## Ideas
- Card graph
- Similar card finder
- Deck bracket classifier (regressor)
- Deck builder

## Approach
- Scrape decks on big fashion
- Embedding with oracle text
- Train soft-clustering (Amount of archetypes/flavors as clusters [Token, crabs, theft, ...])

### Remarks
- Be aware of two faced cards (they do not have oracle text, instead you have to expand the faces)
- Normalization for oracle text could improve (Planeswalkers, TREASURE TOKEN, empty text, ...)
- Currently only commander-legal cards
