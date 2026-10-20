"""Download the exact UCI Bank Marketing variant used by this project."""

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
DATASET_NAME = "bank-additional-full.csv"
DESTINATION = (
    Path(__file__).resolve().parents[1] / "data/raw/bank-additional" / DATASET_NAME
)


def main() -> None:
    if DESTINATION.exists():
        print(f"Dataset already exists: {DESTINATION}")
        return

    with urlopen(DATA_URL, timeout=60) as response:
        payload = response.read()

    with ZipFile(BytesIO(payload)) as archive:
        direct_members = [
            name for name in archive.namelist()
            if Path(name).name == DATASET_NAME
        ]
        if len(direct_members) == 1:
            csv_bytes = archive.read(direct_members[0])
        else:
            # UCI distributes this variant in a zip nested inside the archive.
            nested_members = [
                name for name in archive.namelist()
                if Path(name).name == "bank-additional.zip"
            ]
            if len(nested_members) != 1:
                raise ValueError("Cannot uniquely locate the requested dataset archive")
            with ZipFile(BytesIO(archive.read(nested_members[0]))) as nested:
                matches = [
                    name for name in nested.namelist()
                    if Path(name).name == DATASET_NAME
                ]
                if len(matches) != 1:
                    raise ValueError("Cannot uniquely locate the requested dataset")
                csv_bytes = nested.read(matches[0])

    # Write only the requested CSV; do not extract archive paths to disk.
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_bytes(csv_bytes)
    print(f"Saved {DESTINATION}")
    print("Source: UCI Bank Marketing, DOI 10.24432/C5K306; license: CC BY 4.0")


if __name__ == "__main__":
    main()
