# FaceML

<img src="img/ss.png"/>

A local-first Rust face geometry detector for Apple Silicon. The first backend calls Apple's
Vision framework directly from Rust and returns face bounds, head pose, named landmark regions,
and scale-independent geometry measurements as JSON.

No Xcode project, Swift code, cloud API, or image upload is involved.

## Build and run

The Apple Command Line Tools and Rust are sufficient:

```sh
cargo build --release
./target/release/facefeature detect /path/to/photo.jpg --pretty
```

You can also run it during development:

```sh
cargo run -- detect /path/to/photo.jpg --pretty
```

### Live camera overlay

Run the native AppKit camera window directly from Cargo:

```sh
cargo run --release --bin facefeature-camera
```

This is an ordinary executable, not an `.app` bundle and not a WebView. macOS may ask for camera
permission for whichever terminal launches it. Each face box includes live confidence, landmark
count, normalized bounds and center, plus yaw/pitch/roll in degrees. Close the window to stop the
camera and exit.

Face numbers are persistent tracking IDs rather than per-frame detection indexes. The tracker uses
global face-to-track assignment, motion prediction, geometry smoothing, and a short missing-face
grace period so IDs survive detector reordering and brief occlusion.

### Optional face identity

Geometry tracking cannot recognize somebody after their track expires. Enable the separate local
identity pipeline when that behavior is needed:

```sh
cargo run --release --bin facefeature-camera -- --face-id
```

`--face-id` uses automatic read/write mode: it matches existing identities and enrolls unmatched
faces after multi-frame consensus. Ordinary matches never rewrite a stored centroid, preventing a
borderline or false match from gradually drifting a known identity.

For higher-quality enrollment, use guided capture for a new named person:

```sh
cargo run --release --bin facefeature-camera -- --capture --name "Radit"
```

The overlay guides the subject through straight, left, right, up, and down poses. It requires a
stable hold, captures three embeddings per pose, and shows yaw, pitch, detection quality, and
progress. All 15 pose samples are held in memory and committed together only after capture
completes. Recognition loads five averaged pose templates per identity and selects only the
center, horizontal, or vertical template group relevant to the live head pose, alongside the main
centroid. To replace/add guided templates for an existing identity:

```sh
cargo run --release --bin facefeature-camera -- --capture --person 1
```

When the active Apple Vision revision does not expose native pitch, capture calibrates a vertical
pose estimate from eye/nose/mouth geometry during the straight samples. The overlay marks this as
an approximate value such as `pitch ~0.0°`. The regular face diagnostic label uses the same
landmark fallback against canonical neutral proportions, while native Vision pitch remains
unmarked when available.

To recognize against the existing gallery without changing the SQLite file:

```sh
cargo run --release --bin facefeature-camera -- --read-only
```

In read-only mode, matched people retain their stored identity and unmatched faces appear as
`Unknown`; no identities are inserted and no centroids, counters, names, or timestamps are updated.
`--read-only` and `--capture` cannot be combined. Both flags enable the face-ID pipeline.

For each new tracking ID, the program collects three quality-scored 112x112 aligned face crops at
90 ms intervals and runs the bundled SFace MobileFaceNet model on a dedicated worker. If those
embeddings disagree, it collects up to five observations and selects the most mutually consistent
three before matching. Flat, badly exposed, undersized, low-confidence, extreme-roll, and
extreme-yaw samples are rejected before inference. It prints events such as:

```text
face-id track=31 person=1 similarity=new best=0.112 fingerprint=6abf4e91d2c84410 frames=3/3 consistency=0.914 quality=0.88 pose=center
face-id track=36 person=1 similarity=0.812 fingerprint=6abf4e91d2c84410 frames=3/4 consistency=0.901 quality=0.91 pose=horizontal
```

`track` is temporary geometry state; `person` and `fingerprint` represent the matched persistent
identity. The fingerprint is a compact label for the gallery entry, not an exact or universally
stable hash of a face. Identities survive application restarts in
`data/face_identities.sqlite3`. No face image or embedding is uploaded.

The SQLite database stores normalized embeddings, which are biometric data. It and its WAL files
are git-ignored. Protect or delete the database like any other sensitive local data. For another
gallery location, use `--face-id-db /path/to/identities.sqlite3`.

Each identity has an optional editable `name`. For example, after person 1 has been enrolled:

```sh
sqlite3 data/face_identities.sqlite3 \
  "UPDATE face_identities SET name = 'Radit' WHERE person_id = 1;"
```

Restart the camera after editing a name. The overlay displays that name (or `Person N`), the stable
gallery fingerprint, and the latest cosine similarity. Existing databases are automatically
migrated; schema version 3 adds the pose-specific `face_identity_samples` table.

Use `--face-id-model /path/to/model.onnx` to select another SFace-compatible ONNX model. The
default model is `models/face_recognition_sface_2021dec.onnx`; its license is included beside it.

The live path keeps the 1280x720 camera preset, accepts frames at up to 30 FPS, predicts geometry
forward by the measured Vision latency, and refreshes diagnostic text separately at 5 FPS. The
milliseconds shown in each label are the latest full face-landmark inference time.

Coordinates are normalized to the complete image. The origin is at the lower-left, matching
Vision's native coordinate system.

## Hardware acceleration

Vision chooses the appropriate execution devices internally. The request is not restricted to
CPU-only execution, so macOS may use its accelerated image-processing path and GPU. Apple does
not provide an API that guarantees its built-in face-landmark request runs on the Neural Engine.

With `--face-id`, ONNX Runtime uses its Core ML execution provider with compute units set to `all`,
allowing Core ML to schedule compatible SFace operations on the M1 CPU, GPU, and Neural Engine.
Unsupported operations may fall back to the CPU. Recognition runs for three to five spaced samples
of each new tracking ID, not continuously on every 1280x720 camera frame.

## Project shape

- `src/model.rs`: portable face and geometry data types.
- `src/detector/mod.rs`: backend-neutral detector trait.
- `src/detector/apple_vision.rs`: macOS Vision implementation written in Rust.
- `src/face_id.rs`: Core ML-backed SFace embeddings, alignment, and SQLite identity gallery.
- `src/tracker.rs`: portable multi-face association, stable IDs, and geometry smoothing.
- `src/main.rs`: small JSON CLI.
- `src/bin/facefeature-camera.rs`: AVFoundation capture, native AppKit window, and live overlay.
