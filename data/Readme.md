# Dataset

This directory contains the course-provided Project 1 dataset. Downloaded and generated data artifacts are ignored by Git; this README is tracked.

## Source and preparation

Source: [Google Drive dataset ZIP](https://drive.google.com/file/d/1OPPHkYV74cr678qnmaAoX217nCrt678K/view?usp=sharing), named `dataset_to_release.zip` (approximately 73 MiB).

From the repository root, run:

```bash
bash scripts/prepare_data.sh
```

Install `uv` first using the main README. The shell script resolves the repository directory itself and runs the Python scripts with the locked environment. Dataset downloading and extraction use only Python's standard library.

## Expected files

```text
data/
  Readme.md
  x_train.csv
  y_train.csv
  x_test.csv
  sample_submission.csv
  .extracted.json
```

- `x_train.csv`: training features and row IDs.
- `y_train.csv`: training labels and row IDs.
- `x_test.csv`: test features and row IDs.
- `sample_submission.csv`: example competition submission format.
- `.extracted.json`: local record of extracted file sizes, ZIP checksums, and directories, used to verify repeat runs.

Every file in the ZIP is extracted. The outer `dataset/` folder is removed; any nested folders below it are preserved. Keep the three required CSV filenames unchanged and in the same directory for the course loading helper. Extracted CSVs occupy approximately 309 MiB; allow additional space for the ZIP, environment, and temporary extraction files.

## Repeat runs and recovery

The ZIP is downloaded into a temporary file and promoted only after verification. Extraction stages and checks new files before replacing existing ones. Once all extracted files verify successfully, the checksum record is saved and the ZIP is deleted.

Repeat runs compare every recorded file's size and checksum and skip downloading when the dataset is intact. If a file or the checksum record is missing or changed, rerun the preparation command to download the ZIP and repair the dataset. Keep raw files unchanged; later preprocessing should operate separately.

Handled download failures clean up partial downloads. Extraction failures keep the ZIP for another attempt and clean up the temporary extraction directory. The command returns a nonzero status on failure. For Drive access or quota errors, check that the source allows public downloads or retry later.
