from bladesight.dataset_handler import (
    get_path_to_local_bladesight,
    get_local_datasets,
    get_bladesight_datasets,
    _confirm_dataset_is_valid,
)
from pathlib import Path
import os
import pytest

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