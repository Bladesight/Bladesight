import bladesight.dataset_handler
import os

def test_online_dataset_lookup_failing(monkeypatch):    
    def mock_get_bladesight_datasets():
        return []

    monkeypatch.setattr(bladesight.dataset_handler, "get_bladesight_datasets", mock_get_bladesight_datasets)
    Datasets = bladesight.dataset_handler.BladesightDatasetDirectory()
    assert Datasets.online_datasets == []
    Datasets._ipython_key_completions_()
    Datasets.local_datasets = []
    assert Datasets._ipython_key_completions_() == []

    def mock_get_bladesight_datasets():
        return [
            "bladesight-data-mock/test1",
            "bladesight-data-mock/test2",
            "bladesight-data-mock/test3"
        ]
    monkeypatch.setattr(bladesight.dataset_handler, "get_bladesight_datasets", mock_get_bladesight_datasets)
    Datasets = bladesight.dataset_handler.BladesightDatasetDirectory()
    Datasets._ipython_key_completions_()
    assert Datasets.online_datasets == [
        "bladesight-data-mock/test1",
        "bladesight-data-mock/test2",
        "bladesight-data-mock/test3"
    ]
    
    Datasets.local_datasets = []
    assert Datasets._ipython_key_completions_() == [
        "data/test1",
        "data/test2",
        "data/test3"
    ]
    
def test_default_to_local_datasets(tmp_path, monkeypatch):
    def mock_get_bladesight_datasets():
        raise ValueError("Mocked error")
    monkeypatch.setattr(bladesight.dataset_handler, "get_bladesight_datasets", mock_get_bladesight_datasets)
    os.environ["BLADESIGHT_DATASETS_PATH"] = str(tmp_path)
    path_to_bladesight = bladesight.dataset_handler.get_path_to_local_bladesight()
    path_to_folder1 = path_to_bladesight / "folder1"
    path_to_folder1.mkdir(parents=True, exist_ok=True)
    Datasets = bladesight.dataset_handler.BladesightDatasetDirectory()
    assert Datasets.local_datasets == []
    assert Datasets._ipython_key_completions_() == []
    path_to_db = path_to_folder1 / "file1.db"
    path_to_csv = path_to_folder1 / "file2.csv"
    path_to_db.touch()
    path_to_csv.touch()
    Datasets = bladesight.dataset_handler.BladesightDatasetDirectory()
    assert Datasets.local_datasets == ["folder1/file1"]
    Datasets._ipython_key_completions_()
    assert Datasets.online_datasets == ["folder1/file1"]
    assert Datasets._ipython_key_completions_() == ["data/file1"]

    path_to_db = path_to_folder1 / "file2.db"
    path_to_db.touch()
    Datasets._refresh_available_datasets()
    assert Datasets.local_datasets == ["folder1/file1", "folder1/file2"]
    assert Datasets.online_datasets == ["folder1/file1", "folder1/file2"]
    assert Datasets._ipython_key_completions_() == ["data/file1", "data/file2"]