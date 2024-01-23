from bladesight.dataset_handler import get_path_to_local_bladesight
from pathlib import Path
import os

def testget_path_to_local_bladesight():
    bladesight_data_root_path = str(get_path_to_local_bladesight())
    assert str(bladesight_data_root_path).endswith(".bladesight")
    os.environ["BLADESIGHT_DATASETS_PATH"] = "/user/defined/path/"
    bladesight_data_root_path = get_path_to_local_bladesight()
    assert bladesight_data_root_path == Path("/user/defined/path/.bladesight")