"""Extract the three required course CSVs directly into the data directory."""

import shutil
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path

DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"
ARCHIVE = DATA_DIRECTORY / "dataset_to_release.zip"
REQUIRED_FILES = ("x_train.csv", "y_train.csv", "x_test.csv")


def matches_member(path, member):
    """Check an existing CSV against the ZIP member's size and checksum."""
    if not path.is_file() or path.stat().st_size != member.file_size:
        return False
    checksum = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum == member.CRC


def extract_data(archive_path=ARCHIVE, destination=DATA_DIRECTORY):
    """Stage validated CSVs before replacing incomplete or changed local files."""
    destination = Path(destination)
    with zipfile.ZipFile(archive_path) as archive:
        members = {}
        for filename in REQUIRED_FILES:
            matches = [
                member
                for member in archive.infolist()
                if member.filename == "dataset/" + filename
            ]
            if len(matches) != 1 or matches[0].file_size == 0:
                raise ValueError(
                    "ZIP must contain exactly one nonempty dataset/{}.".format(filename)
                )
            members[filename] = matches[0]

        destination.mkdir(parents=True, exist_ok=True)
        pending = [
            filename
            for filename, member in members.items()
            if not matches_member(destination / filename, member)
        ]
        if not pending:
            print(
                "All three CSVs already match the ZIP; nothing to extract.", flush=True
            )
            return

        with tempfile.TemporaryDirectory(
            prefix=".extract-", dir=destination
        ) as temporary:
            staging = Path(temporary)
            for filename in pending:
                # Only fixed filenames are written; archive paths are never extracted.
                with archive.open(members[filename]) as source:
                    with (staging / filename).open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                if not matches_member(staging / filename, members[filename]):
                    raise ValueError(
                        "Extracted CSV failed validation: {}.".format(filename)
                    )

            # Wait until every pending file validates before replacing existing CSVs.
            for filename in pending:
                (staging / filename).replace(destination / filename)
                print("Extracted: {}".format(destination / filename), flush=True)


def main():
    """Report extraction failures clearly and return a nonzero exit status."""
    try:
        extract_data()
    except (OSError, ValueError, zipfile.BadZipFile, EOFError, zlib.error) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nExtraction cancelled.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
