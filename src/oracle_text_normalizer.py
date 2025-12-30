import re

# regex to capture anything inside {}
def normalize_symbol(match) -> str:
    replacement = replace_symbol(match)
    return replacement + " "


def replace_symbol(match):
    symbol = match.group(1)
    s = symbol.upper()

    # numeric generic mana, e.g. {1}, {2}
    if s.isdigit():
        return "MANA_GENERIC "

    # single color mana
    if s in {"W", "U", "B", "R", "G"}:
        return f"MANA_{s} "

    # variable or special
    if s == "X":
        return "MANA_VARIABLE "
    if s == "T":
        return "TAP "
    if s == "C":
        return "MANA_COLORLESS "
    if s == "Q":
        return "UNTAP "
    if s == "S":
        return "MANA_SNOW "
    if s == "E":
        return "ENERGY "
    if s == "CHAOS":
        return "CHAOS "
    if s == "PW":
        return "PLANESWALKER_SYMBOL "
    if s == "TK":
        return "RESOURCE_TICKET"
    if s == "P":
        return "RESOURCE_PAWPRINT"
    if s == "H":
        return "HALF_MANA"

    # hybrid mana (e.g., {W/U}, {2/G})
    if "/" in s and "P" not in s:
        parts = s.split("/")
        joined = "_".join(parts)
        return f"MANA_HYBRID_{joined} "

    # phyrexian mana (e.g., {W/P})
    if s.endswith("/P"):
        color = s.split("/")[0]
        return f"MANA_{color}_PHYREXIAN "

    return f"MANA_{s} "


def clean_tf_idf_oracle_text(text):
    if not isinstance(text, str):
        return ""

    # remove reminder text in parentheses
    text = re.sub(r"\([^)]*\)", "", text)

    # replace newlines with a sentence boundary
    text = text.replace(" \n", ". ")
    text = text.replace("\n", ". ")

    # normalize bullet points and option separators
    text = text.replace("•", " OPTION ")

    # normalize "—" after 'Choose one —' to a readable token
    text = re.sub(r"choose\s+(one|two|one or both)\s+—", r"choose \1 :", text, flags=re.IGNORECASE)

    # replace mana and activation symbols
    text = re.sub(r"\{([^}]*)\}", normalize_symbol, text)

    # collapse multiple spaces and dots
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\.\s*\.", ".", text)

    text = text.replace(" ,", ",").replace(" .", ".").replace(" :", ":")
    text = text.lower().strip()

    return text

def replace_card_name_from_oracle_text(row):
    """Replace occurrences of the card's name in its oracle text with CARDNAME."""
    name = row["name"]
    text = row["oracle_text_clean"]

    # escape special regex characters in the name (like apostrophes)
    pattern = re.escape(name)

    # replace all case-insensitive occurrences of the name with CARDNAME
    normalized_text = re.sub(pattern, "CARDNAME", text, flags=re.IGNORECASE)
    return normalized_text
