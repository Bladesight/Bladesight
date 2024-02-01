from bladesight.dataset_handler import (
    get_path_to_local_bladesight,
    get_local_datasets,
    get_bladesight_datasets,
    _confirm_dataset_is_valid,
    _read_sql,
    _get_db_tables,
    _get_all_metadata
)
from pathlib import Path
import os
import pytest
import duckdb
from duckdb import CatalogException
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal
import json

def testget_path_to_local_bladesight():
    bladesight_data_root_path = str(get_path_to_local_bladesight())
    assert str(bladesight_data_root_path).endswith(".bladesight")
    os.environ["BLADESIGHT_DATASETS_PATH"] = "/user/defined/path/"
    bladesight_data_root_path = get_path_to_local_bladesight()
    assert bladesight_data_root_path == Path("/user/defined/path/.bladesight")

def test_get_local_datasets(tmp_path):
    os.environ["BLADESIGHT_DATASETS_PATH"] = str(tmp_path)
    assert get_local_datasets() == []
    path_to_bladesight = get_path_to_local_bladesight()
    path_to_bladesight.mkdir(parents=True, exist_ok=True)
    assert get_local_datasets() == []
    path_to_folder1 = path_to_bladesight / "folder1"
    assert get_local_datasets() == []
    path_to_folder1.mkdir(parents=True, exist_ok=True)
    assert get_local_datasets() == []
    # Create a .txt file in get_path_to_local_bladesight()
    with open(path_to_folder1 / "test.txt", 'w') as f:
        f.write("test")
    assert get_local_datasets() == []
    # Now write a file with a .db extention
    with open(path_to_folder1 / "test.db", "w") as f:
        f.write("test")
    assert get_local_datasets() == ["folder1/test"]

def test_get_bladesight_datasets():
    """ This is terrible, I know
    """
    minimum_sets_present = [
        'bladesight-datasets/intro_to_btt/intro_to_btt_ch02', 
        'bladesight-datasets/intro_to_btt/intro_to_btt_ch03', 
        'bladesight-datasets/intro_to_btt/intro_to_btt_ch05', 
        'bladesight-datasets/intro_to_btt/intro_to_btt_ch06'
    ]
    bladesight_online_sets = get_bladesight_datasets()
    assert all([dataset in bladesight_online_sets for dataset in minimum_sets_present])

def test_confirm_dataset_is_valid(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path_to_file = tmp_path / "test.txt"
    # Check that we get a FileNotFoundError if the file doesn't exist
    with pytest.raises(FileNotFoundError):
        _confirm_dataset_is_valid(path_to_file)
    # Create a file, now we should get a ValueError because 
    # the extention
    # is not .db
    path_to_file.touch()
    with pytest.raises(ValueError):
        _confirm_dataset_is_valid(path_to_file)
    # Now write a file with a .db extention
    path_to_file = tmp_path / "test.db"
    path_to_file.touch()
    assert _confirm_dataset_is_valid(path_to_file) is None

def test___read_sql(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path_to_file = tmp_path / "test.db"
    with duckdb.connect(str(path_to_file)) as _:
        pass
    df_empty_tables_pd = _read_sql(path_to_file, "SHOW TABLES;")
    assert df_empty_tables_pd.shape == (0, 1)
    assert df_empty_tables_pd.columns[0] == "name"
    assert isinstance(df_empty_tables_pd, pd.DataFrame)
    df_empty_tables_pl = _read_sql(path_to_file, "SHOW TABLES;", "pl")
    assert df_empty_tables_pl.shape == (0, 1)
    assert df_empty_tables_pl.columns[0] == "name"
    assert isinstance(df_empty_tables_pl, pl.DataFrame)
    with pytest.raises(ValueError):
        _read_sql(path_to_file, "SHOW TABLES;", "invalid")
    df_table_1 = pl.DataFrame({
        "A" : [1,2,3,4],
        "B" : [5,6,7,8]
    })
    df_table_2 = pl.DataFrame({
        "A" : [9,10,11,12],
        "B" : [13,14,15,16]
    })
    with duckdb.connect(str(path_to_file)) as con:
        con.execute("CREATE TABLE table_1 AS SELECT * FROM df_table_1")
        con.execute("CREATE TABLE table_2 AS SELECT * FROM df_table_2")
    df_tables = _read_sql(path_to_file, "SHOW TABLES;")
    assert df_tables.shape == (2, 1)
    assert df_tables.columns[0] == "name"
    assert df_tables["name"].to_list() == ["table_1", "table_2"]
    assert_frame_equal(
        _read_sql(path_to_file, "SELECT * FROM table_1;","pl"), 
        df_table_1
    )
    assert_frame_equal(
        _read_sql(path_to_file, "SELECT * FROM table_2;","pl"), 
        df_table_2
    )

def test__get_db_tables(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path_to_file = tmp_path / "test.db"
    with duckdb.connect(str(path_to_file)) as _:
        pass
    assert _get_db_tables(path_to_file) == []
    df_table_1 = pl.DataFrame({
        "A" : [1,2,3,4],
        "B" : [5,6,7,8]
    })
    df_table_2 = pl.DataFrame({
        "A" : [9,10,11,12],
        "B" : [13,14,15,16]
    })
    with duckdb.connect(str(path_to_file)) as con:
        con.execute("CREATE TABLE table_1 AS SELECT * FROM df_table_1")
        con.execute("CREATE TABLE table_2 AS SELECT * FROM df_table_2")
    assert _get_db_tables(path_to_file) == ["table_1", "table_2"]
    df_metadata = pl.DataFrame({
        "A" : [1,2,3,4],
        "B" : [5,6,7,8]
    })
    with duckdb.connect(str(path_to_file)) as con:
        con.execute("CREATE TABLE metadata AS SELECT * FROM df_metadata")
    assert _get_db_tables(path_to_file) == ["table_1", "table_2"]

def test__get_all_metadata(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path_to_file = tmp_path / "test.db"
    with duckdb.connect(str(path_to_file)) as _:
        pass
    with pytest.raises(CatalogException):
        _get_all_metadata(path_to_file)
    json_object_1 = {
        "A" : [1,2,3,4],
        "B" : [5,6,7,8]
    }
    json_object_2 = {
        "CITATION" : """@article{diamond2021constant,
            title={Constant speed tip deflection determination using the instantaneous phase of blade tip timing data},
            author={Diamond, DH and Heyns, Philippus Stephanus and Oberholster, AJ},
            journal={Mechanical Systems and Signal Processing},
            volume={150},
            pages={107151},
            year={2021},
            publisher={Elsevier}
            }""",
        "URL" : """https://repository.up.ac.za/bitstream/handle/2263/86905/Diamond_Constant_2021.pdf?sequence=1""" 
    }
    
    df_metadata = pl.DataFrame({
        "metadata_key" : ["A", "B"],
        "metadata_value" : [json.dumps(json_object_1), json.dumps(json_object_2)]
    })
    with duckdb.connect(str(path_to_file)) as con:
        con.execute("CREATE TABLE metadata AS SELECT * FROM df_metadata")
    metadata = _get_all_metadata(path_to_file)
    assert metadata["A"] == json_object_1
    assert metadata["B"] == json_object_2
