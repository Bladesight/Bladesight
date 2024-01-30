from bladesight.dataset_handler import (
    get_local_datasets,
    download_dataset_from_bladesight_data,
    Dataset,
    Datasets
)
import os
import pytest

def test_download_dataset_from_bladesight_data(tmp_path):
    os.environ["BLADESIGHT_DATASETS_PATH"] = str(tmp_path)
    assert get_local_datasets() == []
    download_dataset_from_bladesight_data("bladesight-datasets/intro_to_btt/intro_to_btt_ch03")
    assert get_local_datasets() == ["data/intro_to_btt/intro_to_btt_ch03"]
    download_dataset_from_bladesight_data("bladesight-datasets/intro_to_btt/intro_to_btt_ch05")
    assert get_local_datasets() == ["data/intro_to_btt/intro_to_btt_ch03", "data/intro_to_btt/intro_to_btt_ch05"]

def test_Datasets(tmp_path):
    """This function will test whether we can return 
    datasets that have been downloaded. 
    """
    os.environ["BLADESIGHT_DATASETS_PATH"] = str(tmp_path)
    ds = Datasets["data/intro_to_btt/intro_to_btt_ch03"]
    df = ds["table/du_toit_2017_test_1_opr_zero_crossings"]
    assert df.shape == (791, 1)
    assert df.columns == ["time"]
