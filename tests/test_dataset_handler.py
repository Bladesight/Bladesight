from bladesight.dataset_handler import (
    get_path_to_local_bladesight,
    get_local_datasets,
    get_bladesight_datasets,
    _confirm_dataset_is_valid,
    _read_sql,
    _get_db_tables,
    _get_all_metadata,
    _get_printable_citation,
    Dataset,
    BladesightDatasetDirectory
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

def test__get_printable_citation():
    metadata_dict = {}
    assert _get_printable_citation(metadata_dict) == "No citation provided in metadata table."
    metadata_dict = {
        "CITATIONs" : {}
    }
    assert _get_printable_citation(metadata_dict) == "No citation provided in metadata table."
    metadata_dict = {
        "CITATION" : "test"
    }
    assert _get_printable_citation(metadata_dict) == "Citation format error. Here is the raw citation:\n\n" + str(metadata_dict["CITATION"])
    metadata_dict = {
        "CITATION" : {
            "title" : "test"
        }
    }
    assert _get_printable_citation(metadata_dict) == "Citation format error. Here is the raw citation:\n\n" + str(metadata_dict["CITATION"])
    metadata_dict = {
        "CITATION" : {
            "repr" : "uber test"
        }
    }
    assert _get_printable_citation(metadata_dict) == f"""If you use this dataset in published work, please use the below citation:\n\n{metadata_dict['CITATION']['repr']}"""
    metadata_dict = {
        "CITATION" : {
            "repr" : "uber test",
            "url" : "https://test.com",            
        }
    }
    assert_test = f"""If you use this dataset in published work, please use the below citation:\n\n{metadata_dict['CITATION']['repr']}""" + f"""\nLink to paper: {metadata_dict['CITATION']['url']}"""
    assert _get_printable_citation(metadata_dict) == assert_test
    metadata_dict = {
        "CITATION" : {
            "repr" : "uber test",
            "url" : "https://test.com",
            "doi" : "10.1234/5678"
        }
    }
    assert_test += f"""\nDOI: {metadata_dict['CITATION']['doi']}"""
    assert _get_printable_citation(metadata_dict) == assert_test

def test_Dataset(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path_to_file = tmp_path / "test.db"
    with duckdb.connect(str(path_to_file)) as _:
        pass
    df_metadata = pl.DataFrame({
        "metadata_key" : ["CITATION"],
        "metadata_value" : [
            json.dumps({
                "repr" : "uber test",
                "url" : "https://test.com",
                "doi" : "10.1234/5678"
            })
        ]
    })
    df_test_1 = pl.DataFrame({
        "A" : [1,2,3,4],
        "B" : [5,6,7,8]
    })
    df_test_2 = pl.DataFrame({
        "A" : [9,10,11,12],
        "B" : [13,14,15,16]
    })
    with duckdb.connect(str(path_to_file)) as con:
        con.execute("CREATE TABLE metadata AS SELECT * FROM df_metadata")
        con.execute("CREATE TABLE test_1 AS SELECT * FROM df_test_1")
        con.execute("CREATE TABLE test_2 AS SELECT * FROM df_test_2")

    ds = Dataset(path_to_file)
    assert ds.dataframe_library == 'pd'
    ds.set_dataframe_library('pl')
    assert ds.dataframe_library == 'pl'
    ds.set_dataframe_library('pd')
    assert ds.dataframe_library == 'pd'
    with pytest.raises(ValueError):
        ds.set_dataframe_library('invalid')
    assert sorted(ds._ipython_key_completions_()) == ["table/test_1", "table/test_2"]
    repr = ds.__repr__()
    assert "table/test_1" in repr
    assert "table/test_2" in repr
    assert sorted(ds.tables) == ["test_1", "test_2"]
    ds.set_dataframe_library('pl')
    df_test_1_read = ds["table/test_1"]
    assert_frame_equal(df_test_1, df_test_1_read)
    df_test_2_read = ds["table/test_2"]
    assert_frame_equal(df_test_2, df_test_2_read)
    assert ds.metadata["CITATION"]["repr"] == "uber test"
    assert ds.metadata["CITATION"]["url"] == "https://test.com"
    assert ds.metadata["CITATION"]["doi"] == "10.1234/5678"

def test_replace_path_prefix_static_method():
    old_path = "bladesight-datasets/intro_to_btt/intro_to_btt_ch02"

    new_path = BladesightDatasetDirectory.replace_path_prefix(
        old_path,
        "data"
    )
    assert new_path == "data/intro_to_btt/intro_to_btt_ch02"

def test__getitem_key_correct_format():
    assert BladesightDatasetDirectory._getitem_key_correct_format("data/test_1")
    assert BladesightDatasetDirectory._getitem_key_correct_format("table/test_1/") is False