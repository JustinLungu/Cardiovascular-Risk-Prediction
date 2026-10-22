"""Download the course dataset ZIP from public Google Drive using the stdlib."""

import shutil
import sys
import tempfile
import zipfile
from html.parser import HTMLParser
from http.client import IncompleteRead
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPCookieProcessor, build_opener

from .extract_data import dataset_is_ready

FILE_ID = "1OPPHkYV74cr678qnmaAoX217nCrt678K"
ARCHIVE = Path(__file__).resolve().parents[2] / "data" / "dataset_to_release.zip"
DOWNLOAD_URL = "https://drive.google.com/uc?" + urlencode(
    {"export": "download", "id": FILE_ID}
)


class DownloadFormParser(HTMLParser):
    """Read the hidden fields from Drive's large-file confirmation form."""

    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}
        self.in_download_form = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "form" and attributes.get("id") == "download-form":
            self.in_download_form = True
            self.action = attributes.get("action")
        elif tag == "input" and self.in_download_form:
            if attributes.get("type") == "hidden" and attributes.get("name"):
                self.fields[attributes["name"]] = attributes.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.in_download_form = False


def validate_archive(path):
    """Check the ZIP structure and member checksums without extracting data."""
    with zipfile.ZipFile(path) as archive:
        if not archive.infolist():
            raise ValueError("The downloaded ZIP is empty.")
        damaged_member = archive.testzip()
        if damaged_member is not None:
            raise ValueError("ZIP checksum failed for {}.".format(damaged_member))


def download_archive(destination=ARCHIVE):
    """Reuse a valid ZIP or download and validate it before atomic replacement."""
    destination = Path(destination)
    if not destination.exists() and dataset_is_ready(destination.parent):
        print("Extracted dataset is verified; no ZIP download needed.", flush=True)
        return
    if destination.exists():
        try:
            validate_archive(destination)
        except (zipfile.BadZipFile, ValueError, EOFError):
            print(
                "Existing ZIP is incomplete or corrupt; downloading again.", flush=True
            )
        else:
            print("Using existing verified ZIP: {}".format(destination), flush=True)
            return

    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    temporary_path = None
    response = None
    print("Downloading dataset ZIP from Google Drive...", flush=True)
    try:
        response = opener.open(DOWNLOAD_URL, timeout=60)
        if response.headers.get_content_type() == "text/html":
            parser = DownloadFormParser()
            parser.feed(response.read(1024 * 1024).decode("utf-8", errors="replace"))
            response.close()
            action = urlparse(parser.action or "")
            if (
                action.scheme != "https"
                or action.netloc != "drive.usercontent.google.com"
                or action.path != "/download"
                or parser.fields.get("id") != FILE_ID
                or not parser.fields.get("confirm")
            ):
                raise ValueError(
                    "Drive did not provide a download confirmation. Check public "
                    "sharing, download permissions, and the file's download quota."
                )
            confirmation_url = parser.action + "?" + urlencode(parser.fields)
            response = opener.open(confirmation_url, timeout=60)

        if response.headers.get_content_type() == "text/html":
            raise ValueError(
                "Drive returned a webpage instead of the ZIP. Check download "
                "permissions or try again later if the download quota is exceeded."
            )

        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=".dataset-",
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            shutil.copyfileobj(response, temporary, length=1024 * 1024)

        expected_size = response.headers.get("Content-Length")
        if expected_size and temporary_path.stat().st_size != int(expected_size):
            raise ValueError("Download was interrupted: ZIP size does not match.")
        validate_archive(temporary_path)
        temporary_path.replace(destination)
        print("Downloaded and verified: {}".format(destination), flush=True)
    finally:
        if response is not None:
            response.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main():
    """Report download failures clearly and return a nonzero exit status."""
    try:
        download_archive()
    except (
        OSError,
        URLError,
        ValueError,
        zipfile.BadZipFile,
        EOFError,
        IncompleteRead,
    ) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nDownload cancelled.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
