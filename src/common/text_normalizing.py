import re
from typing import Match, Optional, List
import unicodedata
import ftfy


def _replace_symbol(match: Match) -> str:
    symbol = match.group(1)
    s = symbol.upper()

    # numeric mana
    if s.isdigit():
        return f"{s} generic mana "

    # color mana
    color_map = {
        "W": "white mana",
        "U": "blue mana",
        "B": "black mana",
        "R": "red mana",
        "G": "green mana"
    }
    if s in color_map:
        return f"{color_map[s]} "

    # TODO: check if all special symbols are replaced
    # special symbols
    if s == "X":
        return "variable mana "
    if s == "T":
        return "tap "
    if s == "C":
        return "colorless mana "
    if s == "Q":
        return "untap "
    if s == "S":
        return "snow mana "
    if s == "E":
        return "energy "
    if s == "CHAOS":
        return "chaos "
    if s == "PW":
        return "planeswalker symbol "
    if s == "TK":
        return "ticket "
    if s == "P":
        return "pawprint "
    if s == "H":
        return "half mana "

    # hybrid mana
    if "/" in s and "P" not in s:
        parts = s.split("/")
        colors = [color_map.get(p, p.lower()) for p in parts]
        return f"either {' or '.join(colors)} mana "

    # phyrexian mana
    if s.endswith("/P"):
        color_code = s.split("/")[0]
        color = color_map.get(color_code, color_code.lower())
        return f"one {color} mana or two life "

    return f"{s.lower()} "


def normalize_symbol(match: Match) -> str:
    replacement = _replace_symbol(match=match)
    return replacement + " "


def clean_text_encoding(text: Optional[str]) -> Optional[str]:
    if isinstance(text, str):
        # ensure correct text encoding
        return ftfy.fix_text(text=text) 
    return text


def sluggify_card_name(name: str) -> List[str]:
    """ sluggify card name based on EDHREC slugs """
    slug_list = []

    def sluggify(sub_name: str) -> str:
        normalized = unicodedata.normalize('NFKD', sub_name)
        name_ascii = normalized.encode('ascii', 'ignore').decode('utf-8')
        name_lower = name_ascii.lower()
        name_clean = str(name_lower).replace("//","-")
        name_clean = re.sub(pattern=r'[^\w\s-]', repl='', string=name_clean)
        slug = re.sub(pattern=r'[-\s]+', repl='-', string=name_clean).strip('-')
        return slug
    
    slug_list.append(sluggify(sub_name=name))

    if "//" in name:
        split_name = name.split("//")[0]
        slug_list.append(sluggify(sub_name=split_name))

    # name exceptions
    if name == "+2 Mace":
        slug_list.append("plus-2-mace")
    elif name == "Summon: Choco/Mog":
        slug_list.append("summon-choco-mog")
    elif name == "Welcome to . . . // Jurassic Park":
        slug_list.append("welcome-to-")
    elif name == "SP//dr, Piloted by Peni":
        slug_list.append("sp-dr-piloted-by-peni")
    
    return slug_list
