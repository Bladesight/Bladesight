from bladesight.dataset_handler import (
    get_local_datasets,
    download_dataset_from_bladesight_data
)
import os

def test_download_dataset_from_bladesight_data(tmp_path):
    os.environ["BLADESIGHT_DATASETS_PATH"] = str(tmp_path)
    assert get_local_datasets() == []
    download_dataset_from_bladesight_data("bladesight-datasets/intro_to_btt/intro_to_btt_ch03")
    assert get_local_datasets() == ["data/intro_to_btt/intro_to_btt_ch03"]
    download_dataset_from_bladesight_data("bladesight-datasets/intro_to_btt/intro_to_btt_ch05")
    assert get_local_datasets() == ["data/intro_to_btt/intro_to_btt_ch03", "data/intro_to_btt/intro_to_btt_ch05"]