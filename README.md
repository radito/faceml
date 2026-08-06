# FaceML

<img src="https://radito.vercel.app/3f7d92f4a938d6e32f8802eb74c1cd9841b94d9a6ad5eec933c80fc0133d6917/68747470733a2f2f7777772e64726f70626f782e636f6d2f73636c2f66692f356c703563303675656a68667572727337396667682f313939393761383037393564633337613533386637313136356362386239336233633230633563326138646661326535306361373062316337363432643061342e6a70673f726c6b65793d727430357838696763616c367431626279356761737566646d26646c3d30267261773d31"/>

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

Add a dense translucent wireframe that triangulates the available face landmarks and follows the
smoothed cheek, jaw, nose, eye, mouth, and estimated forehead geometry:

```sh
cargo run --release --bin facefeature-camera -- --face-mask polygon
```

Hide the camera image while retaining the generated geometry, mesh, labels, and capture prompts on
a dark background:

```sh
cargo run --release --bin facefeature-camera -- --face-mask polygon --mask-only
```

`--mask-only` changes presentation only. Camera frames are still captured privately inside the
process for Vision detection and optional face-ID embedding, and are never written by this flag.

Render the same geometry as a pseudo-3D wireframe with smoothly graded surface lighting:

```sh
cargo run --release --bin facefeature-camera -- --face-mask depth --mask-only
```

Depth mode maps the landmarks into roll-normalized canonical face coordinates and fits a smooth
rational dome that remains curved even when profile-warped points extend beyond the nominal face
ellipse. The detected nose and lips move forward and the eye regions recess. Yaw, pitch, and roll
rotate the resulting 3D surface normals once for lighting and visibility while Vision's projected
2D points remain the drawing anchors, preventing double-yaw drift. Eight subtle intensity layers
average the two adjacent surfaces at shared edges, and raw Vision landmark outlines are drawn more
softly. This is an inferred artistic Z value from a monocular camera, not measured physical depth,
and it is used only for rendering.

Several roll-aware synthetic forehead rows close the region that Apple Vision does not directly
landmark. Their width uses the contour's temple endpoints rather than roll-sensitive jaw extrema,
and their height is capped from measured eye-to-chin distance. The eyes and inner mouth remain
open. Shared edge midpoints are interpolated along
every oversized edge before repeated Delaunay triangulation, producing a consistent face-relative
triangle size across the forehead, cheeks, nose, and mouth while retaining the original Vision
landmark outlines. The polygon mask is rendered with a lightweight Core Animation shape layer. It
does not require face recognition and can be combined with `--face-id`, `--read-only`, or
`--capture`. At strong yaw, the generated mesh progressively warps its self-occluded side toward a
forehead/eye/nose/mouth/chin visibility axis so Vision's inferred hidden landmarks do not form a
wireframe wing over the background. The mesh is not clipped to the axis-aligned detection box,
because valid rotated contour points can extend beyond that rectangle at strong roll.

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
progress. Guided capture accepts exactly one visible face; zero or multiple detected faces block
sampling and turn the instruction banner red. Invalid pose/quality and capture pipeline errors use
the same red warning state. All 15 pose samples are held in memory and committed together after
capture completes. Recognition loads five averaged pose templates per identity and selects only the
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

## Benchmark

Benchmark the actual headless SFace/Core ML path on the current Mac without opening the camera:

```sh
cargo run --release --bin facefeature-camera -- --benchmark
```

The report separates Core ML session initialization, the first inference, and steady-state average,
median, p95, min/max, and throughput. Change the timed sample count with
`--benchmark-iterations N`. This uses a deterministic synthetic 112x112 input, so it measures the
embedding stage rather than AVFoundation capture, Apple Vision detection, alignment, or rendering.

Tested on an 8 GB Apple M1 MacBook Air using the release build (100 measured runs after five
warmups):

```text
SFace/Core ML benchmark
hardware: MacBook Air | Apple M1 | arm64 | memory: 8 GB | macOS: 27.0
model: models/face_recognition_sface_2021dec.onnx
backend: CoreML | compute units: all | input: 1x3x112x112
session initialization: 42.972 ms
first inference:       8.357 ms
steady state (100 runs after 5 warmups):
  average:             7.956 ms
  median (p50):        7.891 ms
  p95:                 8.383 ms
  min / max:           6.874 / 8.767 ms
  throughput:          125.7 embeddings/s
embedding dimensions:  128
```

These are representative results from one run; temperature, power mode, background load, Core ML
cache state, and macOS version can change the measurements.

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
