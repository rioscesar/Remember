# Privacy

## Local-only boundary

The Android manifest declares no `INTERNET` permission. Remember includes no analytics, cloud sync, account, or media-upload path. The app disables Android backup and device-transfer extraction for its data.

The Companion invokes a locally installed COLMAP executable. It does not send selected photos, reconstruction results, names, or logs to a service. Use a local workspace outside synchronized folders and install dependencies before processing private media.

The Android system picker may show cloud providers. A person should select images already available locally when operating offline. The app reads only URIs selected by the person.

## Stored data

The app's private shared preferences contain memory name, creation time, selected-photo URI strings and names, plus imported sparse landmarks and their support/quality data. It does not copy original images. The imported bundle is read and not retained as a second file.

## Deletion

Deleting a memory removes its local URI references, names, and reconstruction data. It never deletes an original photo, the source bundle, or files outside Remember's private storage. Revoking a URI in system settings can prevent Remember from reopening that source, while preserving its own derived scene.
