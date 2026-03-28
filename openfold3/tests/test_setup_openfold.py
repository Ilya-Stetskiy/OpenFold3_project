from openfold3.setup_openfold import setup_biotite_ccd


def test_setup_biotite_ccd(tmp_path, monkeypatch):
    ccd_path = tmp_path / "test_ccd.cif"
    calls = {"download": 0, "matches": 0}

    def fake_download(bucket, key, local_path):
        calls["download"] += 1
        local_path.write_text("ccd", encoding="utf-8")

    def fake_matches(local_path, bucket, key):
        calls["matches"] += 1
        return True

    monkeypatch.setattr("openfold3.setup_openfold.download_s3_file", fake_download)
    monkeypatch.setattr("openfold3.setup_openfold.s3_file_matches_local", fake_matches)

    has_downloaded = setup_biotite_ccd(ccd_path=ccd_path, force_download=False)
    assert ccd_path.exists()
    assert has_downloaded

    has_downloaded = setup_biotite_ccd(ccd_path=ccd_path, force_download=False)
    assert not has_downloaded
    assert calls == {"download": 1, "matches": 1}
