from typing import Literal


DATAFRAME_LIBRARY_PREFERENCE :Literal["pd", "pl"] = 'pd' 

def set_dataframe_library_preference(library: Literal["pd", "pl"]) -> None:
    """ Set the library preference for the DataFrame library.

    Args:
        library (DATAFRAME_LIBRARY_PREFERENCE): The library to use.
            Either "pd" for pandas or "pl" for polars.
    """
    if library not in ["pd", "pl"]:
        raise ValueError("The library preference must be either 'pd' or 'pl'.")
    globals()["DATAFRAME_LIBRARY_PREFERENCE"] = library


def _get_dataframe_library_preference() -> Literal["pd", "pl"]:
    """ Get the library preference for the DataFrame library.

    Returns:
        DATAFRAME_LIBRARY_PREFERENCE: The library preference.
    """
    return globals()["DATAFRAME_LIBRARY_PREFERENCE"]