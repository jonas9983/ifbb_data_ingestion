import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from ifbb_data_ingestion.etl.extraction.drive_extraction import download_from_drive, fetch_and_extract_zip

@patch("subprocess.run")
def test_download_from_drive_calls_rclone(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    
    remote = "gdrive:test.zip"
    local = str(tmp_path / "test.zip")
    
    success = download_from_drive(remote, local)
    
    assert success is True
    # Verify rclone was called
    args, kwargs = mock_run.call_args
    assert "rclone" in args[0]
    assert "copyto" in args[0]
    assert remote in args[0]
    assert local in args[0]

@patch("ifbb_data_ingestion.etl.extraction.drive_extraction.download_from_drive")
@patch("zipfile.ZipFile")
@patch("pathlib.Path.unlink")
def test_fetch_and_extract_zip(mock_unlink, mock_zipfile, mock_download, tmp_path):
    mock_download.return_value = True
    
    # Mock zipfile context manager
    mock_zip_instance = MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    
    remote = "gdrive:2025.zip"
    extract_to = str(tmp_path / "extracted")
    
    # We need to ensure the data dir exists for the tmp_download.zip
    Path("data").mkdir(exist_ok=True)
    
    success = fetch_and_extract_zip(remote, extract_to)
    
    assert success is True
    mock_download.assert_called_once()
    mock_zip_instance.extractall.assert_called_once_with(extract_to)
    mock_unlink.assert_called_once()
