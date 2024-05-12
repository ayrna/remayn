def get_deep_item_from_dict(d, key, separator="."):
    """Access a nested item in a dictionary using a string separated by a given
    separator. Each separator indicates a level in the nested dictionary.
    For example, a.b.c will access the item d['a']['b']['c'].

    Parameters
    ----------
    d : dict
        The dictionary to access.
    key : str
        The key to access in the dictionary. It should be separated by the
        separator.
    separator : str, optional, default='.'
        The separator to use to split the key.

    Returns
    -------
    value : Any
        The value in the dictionary at the given key. None is returned if not found.
    """

    splitted = key.split(separator)
    value = d
    for k in splitted:
        if isinstance(value, dict) and k in value.keys():
            value = value[k]
        else:
            return None

    return value


def dict_contains_dict(d: dict, sub_d: dict) -> bool:
    """Check if a dictionary contains another dictionary.
    The dictionary can contain other dictionaries, that will be checked recursively.

    Parameters
    ----------
    d : dict
        The dictionary to check.
    sub_d : dict
        The dictionary to check if it is contained in `d`.

    Returns
    -------
    contains : bool
        True if `d` contains `sub_d`, False otherwise.
    """

    for key, value in sub_d.items():
        if key not in d:
            return False

        if isinstance(value, dict):
            if not dict_contains_dict(d[key], value):
                return False
        elif d[key] != value:
            return False

    return True
