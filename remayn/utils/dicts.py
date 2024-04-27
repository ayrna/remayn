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
        if k in value.keys():
            value = value[k]
        else:
            return None

    return value
