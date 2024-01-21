import re


def clean_string(contents_raw: str, replace_dict: dict) -> str:
    contents_clean = contents_raw
    for initial_char, replace_char in replace_dict.items():
        contents_clean = re.sub(initial_char, replace_char, contents_clean)
    return contents_clean.strip()
