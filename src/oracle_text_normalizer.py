import re
from typing import Match

def normalize_symbol(match: Match) -> str:
    replacement = replace_symbol(match=match)
    return replacement + " "


def replace_symbol(match: Match) -> str:
    symbol = match.group(1)
    s = symbol.upper()

    # numeric mana
    if s.isdigit():
        return "MANA_GENERIC "

    # color mana
    if s in {"W", "U", "B", "R", "G"}:
        return f"MANA_{s} "

    # special symbols
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

    # hybrid mana
    if "/" in s and "P" not in s:
        parts = s.split("/")
        joined = "_".join(parts)
        return f"MANA_HYBRID_{joined} "

    # phyrexian mana
    if s.endswith("/P"):
        color = s.split("/")[0]
        return f"MANA_{color}_PHYREXIAN "

    return f"MANA_{s} "


def clean_tf_idf_oracle_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # remove reminder text
    text = re.sub(pattern=r"\([^)]*\)", repl="", string=text)

    # replace newlines
    text = text.replace(" \n", ". ")
    text = text.replace("\n", ". ")

    # normalize bullet points
    text = text.replace("•", " OPTION ")

    # normalize choice separators
    text = re.sub(pattern=r"choose\s+(one|two|one or both)\s+—", repl=r"choose \1 :", string=text, flags=re.IGNORECASE)

    # replace symbols
    text = re.sub(pattern=r"\{([^}]*)\}", repl=normalize_symbol, string=text)

    # fix spacing
    text = re.sub(pattern=r"\s+", repl=" ", string=text)
    text = re.sub(pattern=r"\.\s*\.", repl=".", string=text)

    text = text.replace(" ,", ",").replace(" .", ".").replace(" :", ":")
    text = text.lower().strip()

    return text

def replace_card_name_from_oracle_text(row: dict) -> str:
    """replace card name with cardname"""
    name = row["name"]
    text = row["oracle_text_clean"]

    # escape regex characters
    pattern = re.escape(string=name)

    # replace card name
    normalized_text = re.sub(pattern=pattern, repl="CARDNAME", string=text, flags=re.IGNORECASE)
    return normalized_text
