# Building an optional Windows GUI executable

Keep the source repo as the primary scientific release. Build the GUI executable separately and upload it as a release asset.

## Recommended approach

Use PyInstaller in **one-folder** mode:

```bash
pip install pyinstaller
pyinstaller --noconfirm --clean --windowed --name LDSFL-Meander-GUI gui_ldsfl.py
```

The packaged GUI will be created under `dist/LDSFL-Meander-GUI/`.

## What to upload

For a GitHub release, upload these separately:

- source archive / tagged source release
- optional Windows GUI bundle: `dist/LDSFL-Meander-GUI.zip`

Do not commit the built executable into the source repository.

## Suggested release workflow

1. Tag the source release, for example `v1.0.0`.
2. Build the GUI executable on Windows with the same tagged source.
3. Zip the `dist/LDSFL-Meander-GUI/` folder.
4. Upload the zip as a release asset.
5. Keep Zenodo linked to the source tag, not to the binary asset.

## Notes

- One-folder builds are usually more reliable than one-file builds for scientific Python GUIs.
- If you add a `.ico` icon later, pass it with `--icon path/to/icon.ico`.
- Test the built GUI on a clean Windows machine before distributing it.
