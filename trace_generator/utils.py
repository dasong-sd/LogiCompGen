def get_nested_path_string(dct_str: str, key_list: list):
    """Return string representation of nested dictionary access path"""
    path = dct_str
    for key in key_list:
        if isinstance(key, str):
            path += f'["{key}"]'
        else:
            path += f'[{key}]'
    return path
