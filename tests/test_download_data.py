"""Verify direct and nested UCI archive extraction without network access."""

from io import BytesIO
from zipfile import ZipFile

import pytest

from scripts import download_data


def make_archive(members):
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


@pytest.mark.parametrize("nested", [False, True])
def test_download_extracts_only_the_requested_variant(tmp_path, monkeypatch, nested):
    csv = b'age;duration;y\n30;10;yes\n'
    inner = make_archive({
        "bank-additional/bank-additional-full.csv": csv,
        "bank-additional/bank-additional.csv": b"wrong variant",
        "../../unexpected.txt": b"do not extract",
    })
    payload = make_archive({"bank-additional.zip": inner}) if nested else inner
    destination = tmp_path / "raw/bank-additional/bank-additional-full.csv"
    monkeypatch.setattr(download_data, "DESTINATION", destination)
    monkeypatch.setattr(download_data, "urlopen", lambda *args, **kwargs: BytesIO(payload))

    download_data.main()

    assert destination.read_bytes() == csv
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == [destination]

    # Existing data is preserved and does not trigger another network request.
    def unexpected_request(*args, **kwargs):
        raise AssertionError("Existing datasets must not be downloaded again")

    monkeypatch.setattr(download_data, "urlopen", unexpected_request)
    download_data.main()
    assert destination.read_bytes() == csv
