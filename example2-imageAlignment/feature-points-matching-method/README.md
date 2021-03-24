## Usage

```text
usage: main.py [-h] [--draw] [--nomask] [--ref REF] [--scan SCAN]
```
- For example, you can run `main.py` in the command line:
    ```cmd
    python main.py --draw --ref ../assets/form.jpg --scan ../assets/scan.jpg
    ```
- `--draw` means to draw the intermediate result.
- `--nomask` means draw all the feature points matches, but the default is to only draw the inliers.
- `--ref` means the path of the reference image.
- `--scan` means the the path of the scanned image.