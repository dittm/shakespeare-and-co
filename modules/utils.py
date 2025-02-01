### used in 02_geospatial-analysis.ipynb and 03_statistical_analysis.ipynb ###

def sort_names(names):
    """
    Sorts a list of names in "last, first" format and returns a single string
    with names in "first last" format, joined by "and".

    Args:
        names (list of str): List of names in "last, first" format.

    Returns:
        str: A single string with names in "first last" format, joined by "and".
    """
    sorted_names = sorted(
        [(name.split(", ") if ", " in name else [name, '']) for name in names],
        key=lambda x: (x[1], x[0])
    )
    return ", ".join([f"{last} {first}" for first, last in sorted_names])