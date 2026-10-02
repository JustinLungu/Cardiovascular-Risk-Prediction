"""Extract and verify every dataset ZIP member, then remove the archive."""

import json
import shutil
import stat
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath

DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "data"
ARCHIVE = DATA_DIRECTORY / "dataset_to_release.zip"
MANIFEST = ".extracted.json"
REQUIRED_FILES = ("x_train.csv", "y_train.csv", "x_test.csv")


def safe_path(destination, name):
    """Keep archive and manifest paths inside the intended data directory."""
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError("Unsafe dataset path: {}.".format(name))
    target = destination.joinpath(*relative.parts)
    try:
        target.resolve().relative_to(destination.resolve())
    except ValueError:
        raise ValueError("Dataset path escapes its directory: {}.".format(name))
    return target


def matches_file(path, size, expected_checksum):
    """Check an existing file against its expected size and ZIP checksum."""
    if not path.is_file() or path.stat().st_size != size:
        return False
    checksum = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum == expected_checksum


def dataset_is_ready(destination=DATA_DIRECTORY):
    """Verify all extracted files using the record saved before deleting the ZIP."""
    destination = Path(destination)
    try:
        record = json.loads((destination / MANIFEST).read_text())
        files = record["files"]
        if not all(
            name in files and files[name]["size"] > 0 for name in REQUIRED_FILES
        ):
            return False
        return all(
            matches_file(safe_path(destination, name), details["size"], details["crc"])
            for name, details in files.items()
        ) and all(
            safe_path(destination, name).is_dir() for name in record["directories"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def extract_data(archive_path=ARCHIVE, destination=DATA_DIRECTORY):
    """Extract all members, preserving subfolders below the dataset wrapper."""
    archive_path = Path(archive_path)
    destination = Path(destination)
    if not archive_path.exists() and dataset_is_ready(destination):
        print(
            "All extracted dataset files are verified; ZIP already removed.", flush=True
        )
        return

    with zipfile.ZipFile(archive_path) as archive:
        files = {}
        directories = set()
        for member in archive.infolist():
            # Strip the archive's dataset/ wrapper so the CSVs sit directly in data/.
            name = member.filename
            safe_path(destination, name)
            if name.startswith("dataset/"):
                name = name[len("dataset/") :]
            if not name:
                continue
            name = str(PurePosixPath(name))
            safe_path(destination, name)
            if name in (MANIFEST, archive_path.name, "Readme.md", "README.md"):
                raise ValueError(
                    "Archive conflicts with a local data file: {}.".format(name)
                )
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Archive contains a symbolic link: {}.".format(name))
            if member.is_dir():
                directories.add(name)
            elif name in files:
                raise ValueError("Archive contains duplicate file: {}.".format(name))
            else:
                files[name] = member

        for name in REQUIRED_FILES:
            if name not in files or files[name].file_size == 0:
                raise ValueError("ZIP must contain a nonempty {}.".format(name))
        if directories.intersection(files):
            raise ValueError("Archive contains conflicting file and directory paths.")

        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".extract-", dir=destination
        ) as temporary:
            staging = Path(temporary)
            pending = []
            for name, member in files.items():
                if matches_file(
                    safe_path(destination, name), member.file_size, member.CRC
                ):
                    continue
                output_path = safe_path(staging, name)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, output_path.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if not matches_file(output_path, member.file_size, member.CRC):
                    raise ValueError(
                        "Extracted file failed validation: {}.".format(name)
                    )
                pending.append(name)

            # Every new file validates before existing files are replaced.
            for name in sorted(directories):
                safe_path(destination, name).mkdir(parents=True, exist_ok=True)
            for name in pending:
                output_path = safe_path(destination, name)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                safe_path(staging, name).replace(output_path)
                print("Extracted: {}".format(output_path), flush=True)

            record = {
                "files": {
                    name: {"size": member.file_size, "crc": member.CRC}
                    for name, member in files.items()
                },
                "directories": sorted(directories),
            }
            manifest_path = staging / MANIFEST
            manifest_path.write_text(json.dumps(record, indent=2) + "\n")
            manifest_path.replace(destination / MANIFEST)

    # The manifest lets subsequent runs validate files without downloading again.
    if not dataset_is_ready(destination):
        raise ValueError("Dataset verification failed; ZIP retained for recovery.")
    archive_path.unlink()
    print(
        "All dataset files verified; removed ZIP: {}".format(archive_path), flush=True
    )


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
