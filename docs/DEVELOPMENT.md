# Development guide

## Requirements

- Android Studio with JDK 17 and Android SDK Platform 35
- A physical device or emulator running Android 7.0 (API 24) or later
- Python 3 and a locally installed, license-reviewed COLMAP build for the optional local Companion

## Build

```text
gradlew.bat test
gradlew.bat assembleDebug
gradlew.bat assembleRelease
```

`local.properties` contains a machine-specific Android SDK path and is ignored by Git.

## Companion workflow

1. Copy a deliberately overlapping group of photos into a local non-synced directory.
2. Run `tools/reconstruct_memory.py` with a local COLMAP executable.
3. Import the resulting `reconstruction.json` into the matching memory in the app.

The tool only exports a sparse reconstruction when it can verify its configured evidence thresholds. It otherwise exits with a failure, which is the expected result for insufficient overlap.
