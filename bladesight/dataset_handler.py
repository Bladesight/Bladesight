import json
import os
import pathlib
from typing import Dict, List, Literal, Union, Any, Optional
from typing_extensions import Self
import duckdb
import pandas as pd
import polars as pl
import s3fs
from yaspin import yaspin
import time
BLADESIGHT_DATASETS_S3_BUCKET = "bladesight-datasets"

def get_path_to_local_bladesight() -> pathlib.Path:
    """This function returns the path to the local datasets folder.
    If there is no environmental variable called BLADESIGHT_DATASETS_PATH, it
    will return ~/.bladesight.

    Returns:
        pathlib.Path: The path to the local datasets folder. It does
            not necessarily exist.
    """
    if "BLADESIGHT_DATASETS_PATH" in os.environ:
        return pathlib.Path(os.environ["BLADESIGHT_DATASETS_PATH"]) / ".bladesight"
    else:
        return pathlib.Path.home() / ".bladesight"

def get_local_datasets() -> List[str]:
    """This function returns a list of the names of the datasets in the local
    datasets folder.

    Returns:
        List[str]: A list of the dataset names in the in the 
            local datasets folder.
    """
    BLADESIGHT_DATASETS_PATH = get_path_to_local_bladesight()
    if not BLADESIGHT_DATASETS_PATH.exists():
        return []
    else:
        local_datasets = []
        for path_root, _, files in os.walk(BLADESIGHT_DATASETS_PATH):
            for file in files:
                if file.endswith(".db"):
                    path_parts = pathlib.Path(path_root).parts
                    path_prefix = None
                    add_parts = False
                    for part in path_parts:
                        if add_parts:
                            if path_prefix is None:
                                path_prefix = part
                            else:
                                path_prefix = path_prefix + "/" + part
                        if part == ".bladesight":
                            add_parts = True
                    local_datasets.append(f"{path_prefix}/{file}"[:-3])
        return local_datasets

def get_bladesight_datasets() -> List[str]:
    """This function returns a list of all the datasets in
        the bladesight-datasets bucket.

    Returns:
        List[str]: A list of the names of the datasets in the bucket.
    """
    s3 = s3fs.S3FileSystem(anon=True)
    datasets = []
    with yaspin(text="Getting all datasets from Bladesight Data..."):
        for bucket_root, _, files in s3.walk(BLADESIGHT_DATASETS_S3_BUCKET + "/"):
            for file in files:
                if file.endswith(".db"):
                    datasets.append(f"{bucket_root}/{file}"[:-3])
    return datasets

def download_dataset_from_bladesight_data(dataset_path_on_s3: str) -> None:
    """This function downloads a dataset from S3 and saves it locally.
    
    Args:
        dataset_path_on_s3 (str): The path to the dataset on S3.

    Examples:
    ---------
        Download a dataset into the local .bladesight directory

        >>> download_dataset_from_bladesight_data("bladesight-datasets/intro_to_btt/intro_to_btt_ch02")
    """
    s3 = s3fs.S3FileSystem(anon=True)
    PATH_TO_LOCAL_DB = get_path_to_local_bladesight() / "data"
    for s3_subfolder in dataset_path_on_s3.split("/")[1:]:
        PATH_TO_LOCAL_DB = PATH_TO_LOCAL_DB / s3_subfolder

    if not PATH_TO_LOCAL_DB.parent.exists():
        PATH_TO_LOCAL_DB.parent.mkdir(parents=True)

    with yaspin(
        text=f"Downloading {dataset_path_on_s3} from Bladesight Data..."
    ) as spinner:
        s3.download(dataset_path_on_s3 + ".db", str(PATH_TO_LOCAL_DB) + ".db")
    spinner.text = f"Done downloading {dataset_path_on_s3} from Bladesight Data... "
    spinner.ok("✅ ")

def _confirm_dataset_is_valid(path_to_db : pathlib.Path) -> None:
    """This function checks if a dataset exists 
    and has a .db extension in its name.

    Args:
        path_to_db (pathlib.Path): The path to the dataset.

    Raises:
        FileNotFoundError: If the dataset does not exist.
        ValueError: If the dataset does not have a .db extension.

    Examples:
    ---------
        Check that a dataset has a .db extention.
        
        >>> _confirm_dataset_is_valid("bladesight-data/intro_to_btt/intro_to_btt_ch02.db")
    """
    if not path_to_db.exists():
        raise FileNotFoundError(
            f"You are trying to open the dataset {path_to_db},"
            " but it does not exist!"
        )
    if path_to_db.suffix != ".db":
        raise ValueError(
            f"You are trying to open the dataset {path_to_db}, "
            "but it does not have a .db extension!"
        )

def _execute_sql_without_arg(
    path_to_db : pathlib.Path, 
    sql_query: str, 
    return_mode: Literal['pd', 'pl', 'no_return'] = 'pd'
) -> Union[pd.DataFrame, pl.DataFrame, None]:
    """This function executes a DuckDB SQL query on the dataset
    and returns its result as a pandas or polars DataFrame.

    Args:
        path_to_db (pathlib.Path): The path to the dataset.
        sql_query (str): The SQL query to execute.
        return_mode (Literal['pd', 'pl', 'no_return'], optional): The return 
            mode. Defaults to 'pd'.

    Returns:
        Union[pd.DataFrame, pl.DataFrame, None]: The results of the query.

    Raises:
        ValueError: If the return_mode is not 'pd', 'pl' or 'no_return'.
    
    Examples:
    ---------
        Read a SQL query from a dataset into a Polars DataFrame.

        >>> _execute_sql_without_arg(
        ... "bladesight-data/intro_to_btt/intro_to_btt_ch02.db", 
        ... "SELECT * FROM metadata;",
        ... "pl"
        ... )
    """
    if return_mode not in ["pd", "pl", "no_return"]:
        raise ValueError("return_mode must be 'pd', 'pl' or 'no_return'")
    
    with duckdb.connect(str(path_to_db)) as con:
        if return_mode == "pd":
            df = con.execute(sql_query).df()
        elif return_mode == "pl":
            df = con.execute(sql_query).pl()
        elif return_mode == "no_return":
            con.execute(sql_query)
            df = None
    return df

def _execute_sql_with_arg(
    path_to_db : pathlib.Path, 
    sql_query: str, 
    df_in_memory: Union[pd.DataFrame, pl.DataFrame],
    return_mode: Literal["pd", "pl", 'no_return'] = "pd"
) -> Union[pd.DataFrame, pl.DataFrame, None]:
    """This function executes a DuckDB SQL query on the dataset
    and returns its result as a pandas or polars DataFrame.

    Args:
        path_to_db (pathlib.Path): The path to the dataset.
        sql_query (str): The SQL query to execute.
        df_in_memory (Union[pd.DataFrame, pl.DataFrame]): The DataFrame to use 
            in the query.
        return_mode (Literal["pd", 'pl', 'no_return'], optional): The return 
            mode. Defaults to 'pd'.

    Returns:
        Union[pd.DataFrame, pl.DataFrame, None]: The results of the query.

    Raises:
        ValueError: If the return_mode is not 'pd', 'pl' or 'no_return'.
    
    Examples:
    ---------
        Read a SQL query from a dataset into a Polars DataFrame.

        >>> _execute_sql_without_arg(
        ... "bladesight-data/intro_to_btt/intro_to_btt_ch02.db", 
        ... "SELECT * FROM metadata;",
        ... "pl"
        ... )
    """
    if return_mode not in ["pd", "pl", "no_return"]:
        raise ValueError("return_mode must be 'pd', 'pl' or 'no_return'")
    
    with duckdb.connect(str(path_to_db)) as con:
        if return_mode == "pd":
            df = con.execute(sql_query).df()
        elif return_mode == "pl":
            df = con.execute(sql_query).pl()
        elif return_mode == "no_return":
            con.execute(sql_query)
            df = None
    return df

def _set_metadata_key(
        path_to_db : pathlib.Path, 
        metadata_key : str, 
        metadata: Dict[str, Dict]
    ) -> None:
    """This function sets the metadata in the dataset.

    Args:
        path_to_db (pathlib.Path): The path to the dataset.
        metadata_key (str): The key for the metadata.
        metadata (Dict[str, Dict]): The metadata.

    Examples:
    ---------
        Set the metadata in a dataset.

        >>> _set_metadata(
        ... "bladesight-data/intro_to_btt/intro_to_btt_ch02.db",
        ... {
        ...     "CITATION": {
        ...         "repr": "This is a citation",
        ...         "url": "https://example.com",
        ...         "doi": "10.1234/5678"
        ...     }
        ... }
        ... )
    """
    with duckdb.connect(str(path_to_db)) as con:
        con.execute("DELETE FROM metadata WHERE metadata_key = ?;", (metadata_key, ))
        metadata_value_encoded = json.dumps(metadata)
        con.execute(
            "INSERT INTO metadata VALUES (?, ?);",
            (
                metadata_key,
                metadata_value_encoded,
            ),
        )

def _get_all_metadata(path_to_db : pathlib.Path) -> Dict[str, Union[Dict, Any]]:
    """This function returns a metadata dictionary
    from the metadata table in the dataset.

    Args:
        path_to_db (pathlib.Path): The path to the dataset.

    Returns:
        Dict[str, Union[Dict, Any]]: The metadata.

    Examples:
    ---------
        Get all the metadata from a dataset.

        >>> _get_all_metadata("bladesight-data/intro_to_btt/intro_to_btt_ch02.db")
        {
            "CITATION": {
                "repr": "This is a citation",
                "url": "https://example.com",
                "doi": "10.1234/5678"
            }
        }
    """
    df_metadata = _execute_sql_without_arg(path_to_db, "SELECT * FROM metadata;")
    metadata = {}
    for _, row in df_metadata.iterrows():
        metadata[row["metadata_key"]] = json.loads(row["metadata_value"])
    return metadata

def _get_db_tables(path_to_db : pathlib.Path) -> List[str]:
    """This method gets the tables in the dataset. It
    excludes the metadata table.

    Args:
        path_to_db (pathlib.Path): The path to the dataset.

    Returns:
        List[str]: The tables in the dataset.

    Examples:
    ---------
        Get all the tables in a dataset.

        >>> _get_db_tables("bladesight-data/intro_to_btt/intro_to_btt_ch02.db")
        ['dataset_1', 'dataset_2']
    """
    all_tables = _execute_sql_without_arg(
        path_to_db,
        "SHOW TABLES;"
    )["name"].to_list()
    data_tables = list(set(all_tables) - set(["metadata"]))
    return sorted(data_tables)

def _get_printable_citation(metadata: Dict[str, Dict]) -> str:
    """This function returns a printable citation 
    from the metadata.

    Args:
        metadata (Dict[str, Dict]): The metadata.

    Returns:
        str: The printable citation.

    Examples:
    ---------
        Get a printable citation from the metadata.

        >>> _get_printable_citation({
        ... "CITATION": {
        ...     "repr": "This is a citation",
        ...     "url": "https://example.com",
        ...     "doi": "10.1234/5678"
        ...     }
        ... })
        "If you use this dataset in published work, please \
         use the below citation:\n\nThis is a citation\nLink\
         to paper: https://example.com\nDOI: 10.1234/5678"
    """
    if "CITATION" not in metadata.keys():
        return "No citation provided in metadata table."
    
    citation = metadata["CITATION"]
    if not isinstance(citation, dict):
        return "Citation format error. Here is the raw citation:\n\n" + str(citation)
    
    if "repr" not in citation.keys():
        return "Citation format error. Here is the raw citation:\n\n" + str(citation)

    cite_main = f"""If you use this dataset in published work, please use the below citation:\n\n{citation['repr']}"""

    if "url" in citation.keys():
        cite_main += f"""\nLink to paper: {citation['url']}"""

    if "doi" in citation.keys():
        cite_main += f"""\nDOI: {citation['doi']}"""
    
    return cite_main

class Dataset:
    """This object is used to access data from a dataset.
    
    Args:
        path (pathlib.Path): The path to the dataset.
    
    Examples:
    ---------
        >>> dataset = Dataset("bladesight-data/intro_to_btt/intro_to_btt_ch02.db")
        >>> dataset.tables
        ['dataset_1', 'dataset_2']
        >>> dataset.metadata
        {
            "CITATION": {
                "repr": "This is a citation",
                "url": "https://example.com",
                "doi": "10.1234/5678"
            }
        }
        >>> dataset.set_dataframe_library("pl")
        >>> df_table = dataset["table/dataset_1"]
        >>> dataset.print_citation()
    """
    def __init__(self, path: pathlib.Path):
        _confirm_dataset_is_valid(path)
        self.path = path
        self.tables: List[str] = _get_db_tables(self.path)
        self.metadata: Dict[str, Dict] = _get_all_metadata(self.path)
        self.dataframe_library: Literal["pd", "pl"] = "pd"
        self.print_citation()
    
    # Create a getter and setter for the metadata.doi property
    @property
    def doi(self) -> str:
        """This function returns the DOI from the metadata.

        Returns:
            str: The DOI.

        Examples:
        ---------
            Get the DOI from the metadata.

            >>> dataset = Dataset("bladesight-data/intro_to_btt/intro_to_btt_ch02.db")
            >>> dataset.doi
            "10.1234/5678"
        """
        return self.metadata["CITATION"]["doi"]
    
    def _set_citation_field(
            self, 
            metadata_key : Literal["repr", "url", "doi"], 
            new_value : str
        ):
        """This function sets a metadata field in the dataset.

        Args:
            metadata_key (Literal["repr", "url", "doi"]): The metadata key.
            new_value (str): The new value.
        
        Examples:
        ---------
            Set the metadata field in the dataset.

            >>> dataset = Dataset("bladesight-data/intro_to_btt/intro_to_btt_ch02.db")
            >>> dataset._set_citation_field("doi", "10.1234/5678")
        """
        self.metadata["CITATION"][metadata_key] = new_value
        _set_metadata_key(self.path, "CITATION", self.metadata['CITATION'])
        self.metadata: Dict[str, Dict] = _get_all_metadata(self.path)

    @doi.setter
    def doi(self, new_doi: str):
        """This function sets the DOI in the metadata.

        Args:
            new_doi (str): The new DOI.

        Examples:
        ---------
            Set the DOI in the metadata.

            >>> dataset = Dataset("bladesight-data/intro_to_btt/intro_to_btt_ch02.db")
            >>> dataset.doi = "10.1234/5678"
        """
        self._set_citation_field("doi", new_doi)

    @property
    def url(self) -> str:
        """This function returns the URL from the metadata.

        Returns:
            str: The URL.

        Examples:
        ---------
            Get the URL from the metadata.

            >>> dataset = Dataset("bladesight-data/intro_to_btt/intro_to_btt_ch02.db")
            >>> dataset.url
            "https://example.com"
        """
        return self.metadata["CITATION"]["url"]
    
    @url.setter
    def url(self, new_url: str):
        """This function sets the URL in the metadata.

        Args:
            new_url (str): The new URL.

        Examples:
        ---------
            Set the URL in the metadata.

            >>> dataset = Dataset("bladesight-data/intro_to_btt/intro_to_btt_ch02.db")
            >>> dataset.url = "https://example.com"
        """
        self._set_citation_field("url", new_url)

    @property
    def citation(self) -> str:
        """This function returns the citation from the metadata.
        This is syntactic sugar for the repr field in the CITATION.

        Returns:
            str: The citation.

        Examples:
        ---------
            Get the citation from the metadata.

            >>> dataset = Dataset("bladesight-data/intro_to_btt/intro_to_btt_ch02.db")
            >>> dataset.citation
            "This is a citation\nLink to paper: https://example.com\nDOI: 10.1234/5678"
        """
        return self.metadata["CITATION"]["repr"]

    @citation.setter
    def citation(self, new_citation: str):
        """This function sets the citation in the metadata.
        This is syntactic sugar for the repr field in the CITATION.

        Args:
            new_citation (str): The new citation.

        Examples:
        ---------
            Set the citation in the metadata.

            >>> dataset = Dataset("bladesight-data/intro_to_btt/intro_to_btt_ch02.db")
            >>> dataset.citation = "This is a citation"
        """
        self._set_citation_field("repr", new_citation)
    
    def query(
            self, 
            sql_query: str, 
            df_in_memory : Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
            no_return : bool = False
        ) -> Union[pd.DataFrame, pl.DataFrame]:
        """This function executes a SQL query on the dataset
        and returns its result as a pandas or polars DataFrame.

        Args:
            sql_query (str): The SQL query to execute.
            df_in_memory (Optional[Union[pd.DataFrame, pl.DataFrame]], optional):
                An in-memory DataFrame to pass through to the query statement.
                Defaults to None.
            no_return (bool, optional): If True, the query will not return a DataFrame.
                Use this with custom create statements.

        Returns:
            Union[pd.DataFrame, pl.DataFrame]: The results of the query.

        Examples:
        ---------
            Query a dataset using SQL.

            >>> dataset = Dataset("bladesight-data/intro_to_btt/intro_to_btt_ch02.db")
            >>> df = dataset.query("SELECT * FROM table_1;")
        """
        if df_in_memory is not None:
            if not isinstance(df_in_memory, (pd.DataFrame, pl.DataFrame)):
                raise ValueError("df_in_memory must be a pandas or polars DataFrame")
            return _execute_sql_with_arg(
                self.path, 
                sql_query, 
                df_in_memory, 
                return_mode=self.dataframe_library if not no_return else "no_return"
            )
        return _execute_sql_without_arg(
            self.path, 
            sql_query, 
            return_mode=self.dataframe_library if not no_return else "no_return"
        )

    def set_dataframe_library(self, library: Literal["pd", "pl"]):
        """This function sets the dataframe library to 
        use when returning data.

        Args:
            library (Literal['pd', 'pl']): The dataframe library to use.
        
        Raises:
            ValueError: If the library is not 'pd' or 'pl'.

        Examples:
        ---------
            Set the dataframe library to polars.

            >>> dataset = Dataset("bladesight-data/intro_to_btt/intro_to_btt_ch02.db")
            >>> dataset.set_dataframe_library("pl")
        """
        if library in ["pd", "pl"]:
            self.dataframe_library = library
        else:
            raise ValueError("library must be 'pd' or 'pl'")    
    
    def __getitem__(self, key: str) -> Union[pd.DataFrame, pl.DataFrame]:
        """ This function returns a table from the dataset.
        
        Args:
            key (str): The name of the table, prefixed with "table/".

        Raises:
            KeyError: If the table is not found.

        Returns:
            Union[pd.DataFrame, pl.DataFrame]: The table.

        Examples:
        ---------
            Load a table from the dataset into memory:

            >>> dataset = Dataset("bladesight-data/intro_to_btt/intro_to_btt_ch02.db")
            >>> df_table = dataset["table/dataset_1"]
        """
        table_name = key.replace("table/", "")
        if table_name in self.tables:
            return _execute_sql_without_arg(
                self.path,
                f"SELECT * FROM {table_name};", 
                return_mode=self.dataframe_library
            )
        else:
            raise KeyError(
                f"Table {table_name} not found. These are the tables in the dataset: {self.tables}"
            )
    
    def print_citation(self):
        """Print the citation provided in the metadata table."""
        print(_get_printable_citation(self.metadata))
    
    def _ipython_key_completions_(self):
        return ["table/" + i for i in self.tables]
    
    def __repr__(self) -> str:
        """Show the dataset and its tables.

        Returns:
            str: The dataset and its tables in a string.
        """
        table_string = "[\n"
        for table in self.tables:
            table_string += f"\t'table/{table}',\n "
        table_string += "]"
        return f"Dataset({self.path}),\n\n Tables: \n {table_string}"

    def __setitem__(self, key: str, df: Union[pd.DataFrame, pl.DataFrame]):
        """This function sets a table in the dataset.

        Args:
            key (str): The name of the table.
            df (Union[pd.DataFrame, pl.DataFrame]): The table.

        Examples:
        ---------
            Set a table in the dataset.

            >>> dataset = Dataset("bladesight-data/intro_to_btt/intro_to_btt_ch02.db")
            >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
            >>> dataset["table/my_table"] = df
        """
        if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
            raise ValueError("df must be a pandas or polars DataFrame")
        table_name = key.replace("table/", "")
        if table_name in self.tables:
            raise ValueError(
                f"""Table {table_name} already exists in the dataset. """
                """\nIf you want to overwrite this table, use the .drop_table method to drop"""
                """it first."""
            )
        _execute_sql_with_arg(
            self.path,
            f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df",
            df,
            return_mode="no_return"
        )
        print(f"✅ Created table {table_name} in dataset {self.path}")
        self.tables: List[str] = _get_db_tables(self.path)
    
    def refresh_tables(self):
        """This function refreshes the tables in the dataset.

        Examples:
        ---------
            Refresh the tables in the dataset.

            >>> dataset = Dataset("bladesight-data/intro_to_btt/intro_to_btt_ch02.db")
            >>> dataset.refresh_tables()
        """
        self.tables: List[str] = _get_db_tables(self.path)
    
    def drop_table(self, table_name: str) -> Self:
        """This function drops a table in the dataset.

        Args:
            key (str): The name of the table.

        Examples:
        ---------
            Drop a table in the dataset.

            >>> dataset = Dataset("bladesight-data/intro_to_btt/intro_to_btt_ch02.db")
            >>> dataset.drop_table("table/my_table")
        """
        table_name = table_name.replace("table/", "")
        if table_name in self.tables:
            _execute_sql_without_arg(
                self.path,
                f"DROP TABLE {table_name};",
                return_mode="no_return"
            )
            print(f"✅ Dropped table {table_name} in dataset {self.path}")
            self.tables: List[str] = _get_db_tables(self.path)
        else:
            pass
        return self

class BladesightDatasetDirectory:
    """This object is used to access datasets from the 
    Bladesight Data bucket on S3. 
    
    It also lists the local datasets.

    Examples:
    ---------
    Load a dataset into memory:

        >>> Datasets = BladesightDatasetDirectory()
        >>> dataset = Datasets["data/intro_to_btt/intro_to_btt_ch02"]
        >>> df_table = dataset["table/dataset_1"]
    """
    def __init__(self):
        self.path = get_path_to_local_bladesight()
        self.local_datasets = [
            self._replace_path_prefix(i) for i in get_local_datasets()
        ]
        self.local_datasets :  List[str] = get_local_datasets()
        self.online_datasets : List[str] = []

        self.online_loaded = False

    @staticmethod
    def _getitem_key_correct_format(key: str) -> bool:
        """This function checks if the key is in the correct format. The key
        should be in the format "data/intro_to_btt/intro_to_btt_ch02".

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key is in the correct format, False otherwise.

        Examples:
        ---------
        Check if a key is in the correct format:

            >>> BladesightDatasetDirectory._getitem_key_correct_format(
            ... "data/intro_to_btt/intro_to_btt_ch02"
            ... )
            True

            >>> BladesightDatasetDirectory._getitem_key_correct_format(
            ... "intro_to_btt/intro_to_btt_ch02"
            ... )
            False
        """
        if key.startswith("data/"):
            return True
        return False
    
    def __getitem__(self, key: str) -> Dataset:
        """Get the dataset specified by a key. If the dataset is not found, it
        will be downloaded from the Bladesight Data bucket.

        Args:
            key (str): The name of the dataset.

        Raises:
            KeyError: If the dataset is not found.

        Returns:
            Dataset: The dataset.

        Examples:
        ---------
        Load a dataset into memory:

            >>> Datasets = BladesightDatasetDirectory()
            >>> dataset = Datasets["data/intro_to_btt/intro_to_btt_ch02"]
        """
        if self._getitem_key_correct_format(key) is False:
            raise KeyError(
                f"Dataset {key} does not start with data/. The key should be in the format 'data/../../etc'."
            )
        for local_dataset in self.local_datasets:
            homogenized_local_name = self._replace_path_prefix(local_dataset)
            if key == homogenized_local_name:
                return Dataset(self.path / pathlib.Path(local_dataset + ".db"))
        else:
            # Download the dataset from the online datasets
            if self.online_loaded is False:
                self._refresh_available_datasets()
                self.online_loaded = True
            for online_set in self.online_datasets:
                homogenized_online_name = self._replace_path_prefix(online_set)
                if key == homogenized_online_name:
                    download_dataset_from_bladesight_data(
                        self._replace_path_prefix(key, BLADESIGHT_DATASETS_S3_BUCKET)
                    )
                    self.local_datasets = get_local_datasets()
                    return self[key]

            else:
                raise KeyError(f"Dataset {key} not found.")
    
    @staticmethod
    def _replace_path_prefix(
        dataset_full_path: str, replace_prefix: str = "data"
    ) -> str:
        """This function is used to replace the first path prefix with the
            replace_prefix argument. For example, if the dataset path
            is "bladesight-data/intro_to_btt/intro_to_btt_ch02", and the
            replace_prefix is "data", and the path is returned as
            "data/intro_to_btt/intro_to_btt_ch02".

        Args:
            dataset_full_path (str): The full path to the dataset.
            replace_prefix (str, optional): The prefix to replace. Defaults to "data".

        Returns:
            str: The new path.

        Examples:
        ---------
        Replace the first path prefix with "data":
        
            >>> _replace_path_prefix(
            ... "bladesight-data/intro_to_btt/intro_to_btt_ch02", 
            ... "data"
            ... )
            "data/intro_to_btt/intro_to_btt_ch02"
        """
        new_path = [replace_prefix] + dataset_full_path.split("/")[1:]
        return "/".join(new_path)
    
    def _ipython_key_completions_(self):
        """ 
        We replace whatever prefix is in the 
        dataset with "data" 
        """
        if self.online_loaded is False:
            self._refresh_available_datasets()
            self.online_loaded = True
        local_sets_homogenized = [self._replace_path_prefix(i) for i in self.local_datasets]
        online_sets_homogenized = [self._replace_path_prefix(i) for i in self.online_datasets]
        composite_set = local_sets_homogenized
        for i in online_sets_homogenized:
            if i not in composite_set:
                composite_set.append(i)
        return composite_set
    
    def _refresh_available_datasets(self):
        """
        This function refreshes the local and online datasets.
        If the online datasets cannot be read, it will only 
        list the local datasets.
        """
        self.local_datasets = get_local_datasets()
        try:
            self.online_datasets = get_bladesight_datasets()
        except Exception as _:
            print("Could not read remote datasets. Only listing local datasets")
            self.online_datasets = self.local_datasets

    def new_dataset(
            self, 
            dataset_name : str, 
            exist_ok : bool = False 
        ) -> Dataset:
        """This function creates a new dataset object.

        Args:
            dataset_name (str): The name of the dataset.
            exist_ok (bool, optional): If True, the function will not raise an error
                if the dataset already exists. Defaults to False.
        
        Returns:
            Dataset: The dataset object.

        Examples:
        ---------
        Create a new dataset object:

            >>> Datasets = BladesightDatasetDirectory()
            >>> dataset = Datasets.new_dataset("intro_to_btt/intro_to_btt_ch02_my_dev")
        """
        if not dataset_name.startswith("data/"):
            add_prefix = "data/"
        else:
            add_prefix = ""
        new_dataset_path = self.path / pathlib.Path(add_prefix + dataset_name + ".db")
        
        # Check if the dataset already exists
        if new_dataset_path.exists():
            if exist_ok:
                return self[add_prefix + dataset_name]
            else:
                raise FileExistsError(f"Dataset {dataset_name} already exists.")
        
        # Create a new dataset
        with duckdb.connect(str(new_dataset_path)) as con:
            con.execute("CREATE TABLE metadata (metadata_key TEXT, metadata_value TEXT);")
        init_citation = {
                "url" : "No citation provided in metadata table.",
                "doi" : "No citation provided in metadata table.",
                "repr": "No citation provided in metadata table."
        }
        _set_metadata_key(new_dataset_path, "CITATION", init_citation)
        ds = Dataset(new_dataset_path)
        self.local_datasets = get_local_datasets()
        return ds

Datasets = BladesightDatasetDirectory()
