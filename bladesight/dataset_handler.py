import json
import os
import pathlib
from typing import Dict, List, Literal, Union

import duckdb
import pandas as pd
import polars as pl
import s3fs
from yaspin import yaspin

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

# Untested
def download_dataset_from_bladesight_data(dataset_path_on_s3: str) -> None:
    """This function downloads a dataset from S3 and saves it locally.
    
    Args:
        dataset_path_on_s3 (str): The path to the dataset on S3.

    Example usage:
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

# Untested
class Dataset:
    """This object is used to access data from a dataset."""
    
    # Untested
    def __init__(self, path: pathlib.Path):
        self.path = path
        if self._confirm_dataset_valid() is False:
            raise ValueError(f"{self.path} is not a valid dataset.")
        self.tables: List[str] = self._get_tables()
        self.metadata: Dict[str, Dict] = self._get_all_metadata()
        self.dataframe_library: Literal["pd", "pl"] = "pd"
        self.print_citation()
    
    # Untested
    def set_dataframe_library(self, library: Literal["pd", "pl"]):
        """This function sets the dataframe library to use when returning data.

        Args:
            library (Literal['pd', 'pl']): The dataframe library to use.
        """
        if library in ["pd", "pl"]:
            self.dataframe_library = library
        else:
            raise ValueError("library must be 'pd' or 'pl'")
    
    # Untested
    def _confirm_dataset_valid(self) -> bool:
        """This function checks if a dataset exists and has a .db extension in s"""
        if self.path.exists() and self.path.suffix == ".db":
            return True
        return False
    
    # Untested
    def _get_tables(self) -> List[str]:
        """This method gets the tables in the dataset. It
        excludes the metadata table.

        Returns:
            List[str]: The tables in the dataset.
        """
        all_tables = self._read_sql("SHOW TABLES;")["name"].to_list()
        data_tables = list(set(all_tables) - set(["metadata"]))
        return data_tables
    
    # Untested
    def __getitem__(self, key: str):
        table_name = key.replace("table/", "")
        if table_name in self.tables:
            return self._read_sql(
                f"SELECT * FROM {table_name};", return_mode=self.dataframe_library
            )
        else:
            raise KeyError(
                f"Table {table_name} not found. These are the tables in the dataset: {self.tables}"
            )
    
    # Untested
    def print_citation(self):
        """Print the citation provided in the metadata table."""
        citation = self.metadata["CITATION"]
        if "repr" not in citation.keys():
            print("No citation provided in metadata table.")
            return

        cite_main = f"""If you use this dataset in published work, please use the below citation:\n\n{citation['repr']}"""
        print(cite_main)

        if "url" in citation.keys():
            print(f"""\nLink to paper: {citation['url']}""")

        if "doi" in citation.keys():
            print(f"""\nDOI: {citation['doi']}""")
    
    # Untested
    def _ipython_key_completions_(self):
        return ["table/" + i for i in self.tables]

    # Untested
    def _read_sql(
        self, sql: str, return_mode: Literal["pd", "pl"] = "pd"
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """This function executes a SQL query on the dataset
        and returns its results as a pandas or polars DataFrame.

        Args:
            sql (str): The SQL query to execute.
            return_mode (Literal["pd", 'pl'], optional): The return 
                mode. Defaults to 'pd'.

        Returns:
            Union[pd.DataFrame, pl.DataFrame]: The results of the query.
        """
        with duckdb.connect(str(self.path)) as con:
            if return_mode == "pd":
                df = con.sql(sql).df()
            elif return_mode == "pl":
                df = con.sql(sql).pl()
        return df
    
    # Untested
    def _get_all_metadata(self) -> Dict[str, Dict]:
        """This function returns the value of a metadata field.

        Returns:
            Dict[str, Dict]: The metadata.
        """
        df_metadata = self._read_sql("SELECT * FROM metadata;")
        metadata = {}
        for _, row in df_metadata.iterrows():
            metadata[row["metadata_key"]] = json.loads(row["metadata_value"])
        return metadata
    # Untested
    def __repr__(self):
        table_string = "[\n"
        for table in self.tables:
            table_string += f"\t'table/{table}',\n "
        table_string += "]"
        return f"Dataset({self.path}),\n\n Tables: \n {table_string}"

# Untested
class BladesightDatasetDirectory:
    # Untested
    def __init__(self):
        self.path = get_path_to_local_bladesight()
        self.local_datasets = [
            self.replace_path_prefix(i) for i in get_local_datasets()
        ]
        self.online_datasets: List[str] = ...
        try:
            self._ipython_key_completions_()
        except Exception as e:
            print("Could not read remote datasets. Only listing local datasets")
            self.online_datasets = self.local_datasets
    
    # Untested
    def _getitem_key_correct_format(self, key: str) -> bool:
        """This function checks if the key is in the correct format. The key
        should be in the format "data/intro_to_btt/intro_to_btt_ch02".

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key is in the correct format, False otherwise.
        """
        if key.startswith("data/"):
            return True
        return False
    
    # Untested
    def __getitem__(self, key: str) -> Dataset:
        """Get the dataset specified by a key. If the dataset is not found, it
        will be downloaded from the Bladesight Data bucket.

        Args:
            key (str): The name of the dataset.

        Raises:
            KeyError: If the dataset is not found.

        Returns:
            Dataset: The dataset.
        """
        if self._getitem_key_correct_format(key) is False:
            raise KeyError(
                f"Dataset {key} does not start with data/. The key should be in the format 'data/../../etc'."
            )

        for local_dataset in self.local_datasets:
            homogenized_local_name = self.replace_path_prefix(local_dataset)
            if key == homogenized_local_name:
                return Dataset(self.path / pathlib.Path(local_dataset + ".db"))
        else:
            # Download the dataset from the online datasets
            for online_set in self.online_datasets:
                homogenized_online_name = self.replace_path_prefix(online_set)
                if key == homogenized_online_name:
                    download_dataset_from_bladesight_data(
                        self.replace_path_prefix(key, BLADESIGHT_DATASETS_S3_BUCKET)
                    )
                    self.local_datasets = get_local_datasets()
                    return self[key]

            else:
                raise KeyError(f"Dataset {key} not found.")
    
    # Untested
    def replace_path_prefix(
        self, dataset_full_path: str, replace_prefix: str = "data"
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
        """

        new_path = [replace_prefix] + dataset_full_path.split("/")[1:]
        return "/".join(new_path)
    
    # Untested
    def _ipython_key_completions_(self):
        if self.online_datasets is ...:
            self.online_datasets = get_bladesight_datasets()
        return [self.replace_path_prefix(i) for i in self.online_datasets]
    
    # Untested
    def refresh_available_datasets(self):
        """This function refreshes the local and online datasets.
        """
        self.local_datasets = get_local_datasets()
        self.online_datasets = get_bladesight_datasets()

Datasets = BladesightDatasetDirectory()