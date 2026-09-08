# Development guide

## Prerequisites

- Latest stable Android Studio
- Android SDK Platform 35
- JDK 17, supplied by Android Studio or configured locally

## Common tasks

Use Android Studio to sync, run, and test the app. From a terminal with JDK 17 configured, use:

```text
gradlew.bat assembleDebug
gradlew.bat test
```

Android SDK locations belong in `local.properties`, which is intentionally ignored by Git.
