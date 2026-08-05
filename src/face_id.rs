//! Optional local face embeddings and persistent identity matching.
//!
//! The model used by the camera is OpenCV Zoo's Apache-2.0 SFace model, a
//! MobileFaceNet instance trained with the SFace loss. Geometry tracking and
//! identity recognition intentionally remain separate: the camera only sends a
//! small aligned face tensor when it encounters a new tracking ID.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use ort::ep::coreml::{ComputeUnits, ModelFormat, SpecializationStrategy};
use ort::session::Session;
use ort::value::Tensor;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use sha2::{Digest, Sha256};

use crate::{FaceGeometry, LandmarkKind, Point};

pub const SFACE_WIDTH: usize = 112;
pub const SFACE_HEIGHT: usize = 112;
pub const SFACE_COSINE_THRESHOLD: f32 = 0.363;

const SFACE_TEMPLATE: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];
const DATABASE_SCHEMA_VERSION: i64 = 3;
const EMBEDDING_FORMAT: &str = "f32-le-l2-v1";
const CAPTURE_SAMPLES_PER_POSE: usize = 3;
const CAPTURE_HOLD_DURATION: Duration = Duration::from_millis(350);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceIdDatabaseMode {
    ReadOnly,
    ReadWrite,
}

impl Default for FaceIdDatabaseMode {
    fn default() -> Self {
        Self::ReadWrite
    }
}

impl FaceIdDatabaseMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::ReadWrite => "read-write",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FaceCaptureTarget {
    New { name: String },
    Existing { person_id: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapturePose {
    Center,
    Left,
    Right,
    Up,
    Down,
}

impl CapturePose {
    const ALL: [Self; 5] = [Self::Center, Self::Left, Self::Right, Self::Up, Self::Down];

    fn database_label(self) -> &'static str {
        match self {
            Self::Center => "center",
            Self::Left => "left",
            Self::Right => "right",
            Self::Up => "up",
            Self::Down => "down",
        }
    }

    fn instruction(self) -> &'static str {
        match self {
            Self::Center => "LOOK STRAIGHT",
            Self::Left => "TURN LEFT",
            Self::Right => "TURN RIGHT",
            Self::Up => "LOOK UP",
            Self::Down => "LOOK DOWN",
        }
    }
}

#[derive(Clone, Debug)]
pub struct FaceCaptureStatus {
    pub subject: String,
    pub instruction: String,
    pub pose_number: usize,
    pub pose_count: usize,
    pub samples: usize,
    pub samples_required: usize,
    pub yaw_degrees: Option<f64>,
    pub pitch_degrees: Option<f64>,
    pub pitch_estimated: bool,
    pub quality: f32,
    pub message: String,
    pub completed: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct FaceCaptureRequest {
    pose: CapturePose,
    quality: f32,
    yaw_degrees: f64,
    pitch_degrees: f64,
    vertical_proxy: Option<f64>,
}

#[derive(Debug)]
struct FaceIdJob {
    track_id: u64,
    tensor: Vec<f32>,
    capture: Option<FaceCaptureRequest>,
}

/// A non-blocking handle to the dedicated embedding worker.
pub struct FaceIdClient {
    sender: mpsc::SyncSender<FaceIdJob>,
    submitted_tracks: Mutex<HashSet<u64>>,
    results: Arc<Mutex<HashMap<u64, FaceIdentityMatch>>>,
    capture: Option<Arc<Mutex<GuidedCaptureState>>>,
}

impl fmt::Debug for FaceIdClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FaceIdClient")
            .field("submitted_tracks", &self.submitted_tracks)
            .finish_non_exhaustive()
    }
}

impl FaceIdClient {
    pub fn start(
        model_path: &Path,
        cache_dir: &Path,
        database_path: &Path,
        database_mode: FaceIdDatabaseMode,
        capture_target: Option<FaceCaptureTarget>,
    ) -> Result<Arc<Self>, String> {
        if capture_target.is_some() && database_mode != FaceIdDatabaseMode::ReadWrite {
            return Err("guided capture requires a read/write database".to_owned());
        }
        let model_checksum = file_sha256(model_path)?;
        let engine = FaceEmbeddingEngine::new(model_path, cache_dir)?;
        let gallery = IdentityGallery::open(database_path, &model_checksum, database_mode)?;
        if let Some(target) = &capture_target {
            gallery.validate_capture_target(target)?;
        }
        println!(
            "face-id gallery={} identities={} schema={} mode={}",
            database_path.display(),
            gallery.identities.len(),
            DATABASE_SCHEMA_VERSION,
            database_mode.label()
        );
        let (sender, receiver) = mpsc::sync_channel::<FaceIdJob>(4);
        let results = Arc::new(Mutex::new(HashMap::new()));
        let worker_results = Arc::clone(&results);
        let capture = capture_target
            .map(GuidedCaptureState::new)
            .map(|state| Arc::new(Mutex::new(state)));
        let worker_capture = capture.clone();

        thread::Builder::new()
            .name("facefeature-face-id".to_owned())
            .spawn(move || run_worker(engine, receiver, gallery, worker_results, worker_capture))
            .map_err(|error| format!("could not start face-ID worker: {error}"))?;

        Ok(Arc::new(Self {
            sender,
            submitted_tracks: Mutex::new(HashSet::new()),
            results,
            capture,
        }))
    }

    /// Queues at most one embedding attempt for a tracking ID. A full queue is
    /// harmless: the track is left unmarked so a later frame can retry.
    pub fn try_submit(&self, track_id: u64, tensor: Vec<f32>) {
        let mut submitted = self
            .submitted_tracks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if submitted.contains(&track_id) {
            return;
        }

        match self.sender.try_send(FaceIdJob {
            track_id,
            tensor,
            capture: None,
        }) {
            Ok(()) => {
                submitted.insert(track_id);
            }
            Err(mpsc::TrySendError::Full(_)) => {}
            Err(mpsc::TrySendError::Disconnected(_)) => {
                eprintln!("face-ID worker stopped unexpectedly");
                submitted.insert(track_id);
            }
        }
    }

    pub fn wants_track(&self, track_id: u64) -> bool {
        !self
            .submitted_tracks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(&track_id)
    }

    pub fn identity_for_track(&self, track_id: u64) -> Option<FaceIdentityMatch> {
        self.results
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&track_id)
            .cloned()
    }

    pub fn is_guided_capture(&self) -> bool {
        self.capture.is_some()
    }

    pub fn capture_status(&self) -> Option<FaceCaptureStatus> {
        self.capture.as_ref().map(|capture| {
            capture
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .status()
        })
    }

    pub fn capture_request(&self, face: &FaceGeometry, now: Instant) -> Option<FaceCaptureRequest> {
        self.capture.as_ref().and_then(|capture| {
            capture
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .observe(face, now)
        })
    }

    pub fn try_submit_capture(
        &self,
        track_id: u64,
        tensor: Vec<f32>,
        capture_request: FaceCaptureRequest,
    ) {
        match self.sender.try_send(FaceIdJob {
            track_id,
            tensor,
            capture: Some(capture_request),
        }) {
            Ok(()) => {}
            Err(mpsc::TrySendError::Full(_)) => self.release_capture("embedding queue is busy"),
            Err(mpsc::TrySendError::Disconnected(_)) => {
                self.release_capture("face-ID worker stopped");
                eprintln!("face-ID worker stopped unexpectedly");
            }
        }
    }

    pub fn release_capture(&self, message: &str) {
        if let Some(capture) = &self.capture {
            capture
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .release_pending(message);
        }
    }
}

#[derive(Debug)]
struct GuidedCaptureState {
    target: FaceCaptureTarget,
    pose_index: usize,
    samples: usize,
    stable_since: Option<Instant>,
    pending: bool,
    horizontal_sign: Option<f64>,
    vertical_sign: Option<f64>,
    neutral_vertical: Option<f64>,
    yaw_degrees: Option<f64>,
    pitch_degrees: Option<f64>,
    pitch_estimated: bool,
    quality: f32,
    message: String,
    completed_identity: Option<FaceIdentityMatch>,
}

impl GuidedCaptureState {
    fn new(target: FaceCaptureTarget) -> Self {
        Self {
            target,
            pose_index: 0,
            samples: 0,
            stable_since: None,
            pending: false,
            horizontal_sign: None,
            vertical_sign: None,
            neutral_vertical: None,
            yaw_degrees: None,
            pitch_degrees: None,
            pitch_estimated: false,
            quality: 0.0,
            message: "find the pose and hold still".to_owned(),
            completed_identity: None,
        }
    }

    fn subject(&self) -> String {
        match &self.target {
            FaceCaptureTarget::New { name } => name.clone(),
            FaceCaptureTarget::Existing { person_id } => format!("Person {person_id}"),
        }
    }

    fn status(&self) -> FaceCaptureStatus {
        let pose = CapturePose::ALL[self.pose_index.min(CapturePose::ALL.len() - 1)];
        FaceCaptureStatus {
            subject: self.subject(),
            instruction: if self.completed_identity.is_some() {
                "CAPTURE COMPLETE".to_owned()
            } else {
                pose.instruction().to_owned()
            },
            pose_number: self.pose_index.min(CapturePose::ALL.len() - 1) + 1,
            pose_count: CapturePose::ALL.len(),
            samples: self.samples,
            samples_required: CAPTURE_SAMPLES_PER_POSE,
            yaw_degrees: self.yaw_degrees,
            pitch_degrees: self.pitch_degrees,
            pitch_estimated: self.pitch_estimated,
            quality: self.quality,
            message: self.message.clone(),
            completed: self.completed_identity.is_some(),
        }
    }

    fn observe(&mut self, face: &FaceGeometry, now: Instant) -> Option<FaceCaptureRequest> {
        if self.pending
            || self.completed_identity.is_some()
            || self.pose_index >= CapturePose::ALL.len()
        {
            if self.pose_index >= CapturePose::ALL.len() && self.completed_identity.is_none() {
                self.message = "saving enrollment...".to_owned();
            }
            return None;
        }
        let yaw = face.yaw_radians.map(f64::to_degrees);
        let native_pitch = face.pitch_radians.map(f64::to_degrees);
        let vertical_proxy = vertical_pose_proxy(face);
        let estimated_pitch = match (self.neutral_vertical, vertical_proxy) {
            (Some(neutral), Some(current)) => Some((current - neutral) * 200.0),
            (None, Some(_)) if self.pose_index == 0 => Some(0.0),
            _ => None,
        };
        let pitch = native_pitch.or(estimated_pitch);
        self.pitch_estimated = native_pitch.is_none() && pitch.is_some();
        self.yaw_degrees = yaw;
        self.pitch_degrees = pitch;
        self.quality = capture_quality(face);
        let pose = CapturePose::ALL[self.pose_index];
        let Some(yaw) = yaw else {
            self.message = "yaw is unavailable".to_owned();
            self.stable_since = None;
            return None;
        };
        let Some(pitch) = pitch else {
            self.message = "pitch is unavailable".to_owned();
            self.stable_since = None;
            return None;
        };
        if self.quality < 0.65 {
            self.message = "move closer and improve lighting".to_owned();
            self.stable_since = None;
            return None;
        }
        if !pose_matches(pose, yaw, pitch, self.horizontal_sign, self.vertical_sign) {
            self.message =
                pose_guidance(pose, yaw, pitch, self.horizontal_sign, self.vertical_sign);
            self.stable_since = None;
            return None;
        }
        let stable_since = self.stable_since.get_or_insert(now);
        if now.duration_since(*stable_since) < CAPTURE_HOLD_DURATION {
            self.message = "HOLD STILL...".to_owned();
            return None;
        }
        self.pending = true;
        self.stable_since = None;
        self.message = "capturing sample...".to_owned();
        Some(FaceCaptureRequest {
            pose,
            quality: self.quality,
            yaw_degrees: yaw,
            pitch_degrees: pitch,
            vertical_proxy,
        })
    }

    fn sample_completed(&mut self, request: FaceCaptureRequest) -> bool {
        if request.pose == CapturePose::Center {
            if let Some(vertical_proxy) = request.vertical_proxy {
                self.neutral_vertical = Some(match self.neutral_vertical {
                    Some(neutral) => {
                        (neutral * self.samples as f64 + vertical_proxy)
                            / (self.samples as f64 + 1.0)
                    }
                    None => vertical_proxy,
                });
            }
        }
        if request.pose == CapturePose::Left && self.horizontal_sign.is_none() {
            self.horizontal_sign = Some(request.yaw_degrees.signum());
        }
        if request.pose == CapturePose::Up && self.vertical_sign.is_none() {
            self.vertical_sign = Some(request.pitch_degrees.signum());
        }
        self.pending = false;
        self.samples += 1;
        self.message = "sample accepted".to_owned();
        if self.samples >= CAPTURE_SAMPLES_PER_POSE {
            self.samples = 0;
            self.pose_index += 1;
        }
        self.pose_index >= CapturePose::ALL.len()
    }

    fn release_pending(&mut self, message: &str) {
        self.pending = false;
        self.stable_since = None;
        self.message = message.to_owned();
    }

    fn complete(&mut self, identity: FaceIdentityMatch) {
        self.pose_index = CapturePose::ALL.len() - 1;
        self.samples = CAPTURE_SAMPLES_PER_POSE;
        self.pending = false;
        self.message = format!("saved as {}", identity.display_name());
        self.completed_identity = Some(identity);
    }
}

fn capture_quality(face: &FaceGeometry) -> f32 {
    let size_score = ((face.bounding_box.width * face.bounding_box.height) / 0.08).min(1.0) as f32;
    let roll_score = face
        .roll_radians
        .map(|roll| (1.0 - roll.to_degrees().abs() / 35.0).clamp(0.0, 1.0) as f32)
        .unwrap_or(0.75);
    (face.confidence * 0.4 + face.landmark_confidence * 0.3 + size_score * 0.2 + roll_score * 0.1)
        .clamp(0.0, 1.0)
}

fn vertical_pose_proxy(face: &FaceGeometry) -> Option<f64> {
    let left_eye = region_center(face, LandmarkKind::LeftEye)?;
    let right_eye = region_center(face, LandmarkKind::RightEye)?;
    let nose = region_center(face, LandmarkKind::Nose)
        .or_else(|| region_center(face, LandmarkKind::NoseCrest))?;
    let mouth = region_center(face, LandmarkKind::OuterLips)?;
    let eye_y = (left_eye.y + right_eye.y) / 2.0;
    let eye_to_mouth = eye_y - mouth.y;
    if eye_to_mouth.abs() <= f64::EPSILON {
        return None;
    }
    Some((nose.y - mouth.y) / eye_to_mouth)
}

/// Approximate pitch for diagnostics when Vision does not expose native pitch.
/// A tilde should be shown alongside this value because neutral facial proportions vary.
pub fn landmark_pitch_degrees(face: &FaceGeometry) -> Option<f64> {
    const CANONICAL_NEUTRAL_PROXY: f64 = 0.5;
    const PROXY_TO_DEGREES: f64 = 200.0;
    vertical_pose_proxy(face)
        .map(|proxy| ((proxy - CANONICAL_NEUTRAL_PROXY) * PROXY_TO_DEGREES).clamp(-45.0, 45.0))
}

fn pose_matches(
    pose: CapturePose,
    yaw: f64,
    pitch: f64,
    horizontal_sign: Option<f64>,
    vertical_sign: Option<f64>,
) -> bool {
    match pose {
        CapturePose::Center => yaw.abs() <= 10.0 && pitch.abs() <= 10.0,
        CapturePose::Left => {
            (12.0..=50.0).contains(&yaw.abs())
                && horizontal_sign.is_none_or(|sign| yaw.signum() == sign)
        }
        CapturePose::Right => {
            (12.0..=50.0).contains(&yaw.abs())
                && horizontal_sign.is_some_and(|sign| yaw.signum() == -sign)
        }
        CapturePose::Up => {
            yaw.abs() <= 14.0
                && (12.0..=30.0).contains(&pitch.abs())
                && vertical_sign.is_none_or(|sign| pitch.signum() == sign)
        }
        CapturePose::Down => {
            yaw.abs() <= 14.0
                && (12.0..=30.0).contains(&pitch.abs())
                && vertical_sign.is_some_and(|sign| pitch.signum() == -sign)
        }
    }
}

fn pose_guidance(
    pose: CapturePose,
    yaw: f64,
    pitch: f64,
    horizontal_sign: Option<f64>,
    vertical_sign: Option<f64>,
) -> String {
    match pose {
        CapturePose::Center if yaw.abs() > 10.0 => "rotate your face back to center".to_owned(),
        CapturePose::Center => "level your chin toward center".to_owned(),
        CapturePose::Left | CapturePose::Right if yaw.abs() < 12.0 => {
            "turn farther toward the prompt".to_owned()
        }
        CapturePose::Left | CapturePose::Right if yaw.abs() > 50.0 => {
            "turn slightly back toward the camera".to_owned()
        }
        CapturePose::Left if horizontal_sign.is_some_and(|sign| yaw.signum() != sign) => {
            "turn toward the other side".to_owned()
        }
        CapturePose::Right if horizontal_sign.is_some_and(|sign| yaw.signum() != -sign) => {
            "turn toward the opposite side from LEFT".to_owned()
        }
        CapturePose::Up | CapturePose::Down if pitch.abs() < 12.0 => {
            "tilt farther toward the prompt".to_owned()
        }
        CapturePose::Up | CapturePose::Down if pitch.abs() > 30.0 => {
            "tilt slightly back toward center".to_owned()
        }
        CapturePose::Up if vertical_sign.is_some_and(|sign| pitch.signum() != sign) => {
            "tilt toward the other vertical direction".to_owned()
        }
        CapturePose::Down if vertical_sign.is_some_and(|sign| pitch.signum() != -sign) => {
            "tilt opposite from UP".to_owned()
        }
        _ => "adjust your head toward the prompt".to_owned(),
    }
}

struct FaceEmbeddingEngine {
    session: Session,
}

impl FaceEmbeddingEngine {
    fn new(model_path: &Path, cache_dir: &Path) -> Result<Self, String> {
        if !model_path.is_file() {
            return Err(format!(
                "face-ID model not found: {}\nrun from the project directory or pass --face-id-model PATH",
                model_path.display()
            ));
        }
        std::fs::create_dir_all(cache_dir).map_err(|error| {
            format!(
                "could not create Core ML cache {}: {error}",
                cache_dir.display()
            )
        })?;

        let core_ml = ort::ep::CoreML::default()
            .with_compute_units(ComputeUnits::All)
            .with_model_format(ModelFormat::MLProgram)
            .with_static_input_shapes(true)
            .with_specialization_strategy(SpecializationStrategy::FastPrediction)
            .with_model_cache_dir(cache_dir.display().to_string())
            .build();
        let mut builder = Session::builder()
            .map_err(|error| format!("could not initialize ONNX Runtime: {error}"))?
            .with_execution_providers([core_ml])
            .map_err(|error| format!("could not enable the Core ML execution provider: {error}"))?;
        let session = builder.commit_from_file(model_path).map_err(|error| {
            format!(
                "could not load face-ID model {}: {error}",
                model_path.display()
            )
        })?;

        Ok(Self { session })
    }

    fn embed(&mut self, input: Vec<f32>) -> Result<Vec<f32>, String> {
        if input.len() != 3 * SFACE_WIDTH * SFACE_HEIGHT {
            return Err(format!("invalid SFace input length: {}", input.len()));
        }
        let tensor = Tensor::from_array((
            [1usize, 3, SFACE_HEIGHT, SFACE_WIDTH],
            input.into_boxed_slice(),
        ))
        .map_err(|error| format!("could not create face-ID tensor: {error}"))?;
        let outputs = self
            .session
            .run(ort::inputs![tensor])
            .map_err(|error| format!("face-ID inference failed: {error}"))?;
        let (_, output) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|error| format!("face-ID model returned an invalid tensor: {error}"))?;
        normalize_embedding(output)
    }
}

fn run_worker(
    mut engine: FaceEmbeddingEngine,
    receiver: mpsc::Receiver<FaceIdJob>,
    mut gallery: IdentityGallery,
    results: Arc<Mutex<HashMap<u64, FaceIdentityMatch>>>,
    capture: Option<Arc<Mutex<GuidedCaptureState>>>,
) {
    let mut captured_samples = Vec::new();
    for job in receiver {
        match engine.embed(job.tensor) {
            Ok(embedding) if job.capture.is_some() => {
                let request = job.capture.expect("capture request checked");
                captured_samples.push(CapturedFaceSample {
                    pose: request.pose,
                    embedding,
                    quality: request.quality,
                });
                let Some(capture) = &capture else {
                    eprintln!("face-ID capture job has no capture session");
                    continue;
                };
                let finished = capture
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .sample_completed(request);
                if finished {
                    let target = capture
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .target
                        .clone();
                    match gallery.commit_capture(&target, &captured_samples) {
                        Ok(identity) => {
                            println!(
                                "face-id capture complete person={} name={:?} samples={} fingerprint={}",
                                identity.person_id,
                                identity.display_name(),
                                captured_samples.len(),
                                identity.fingerprint
                            );
                            capture
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner())
                                .complete(identity.clone());
                            results
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner())
                                .insert(job.track_id, identity);
                        }
                        Err(error) => {
                            capture
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner())
                                .release_pending(&format!("database error: {error}"));
                            eprintln!("face-ID capture database error: {error}");
                        }
                    }
                }
            }
            Ok(embedding) => match gallery.identify(embedding) {
                Ok(result) => {
                    let display_name = result.display_name();
                    if result.is_new {
                        if result.persisted {
                            println!(
                                "face-id track={} person={} name={:?} similarity=new best={:.3} fingerprint={}",
                                job.track_id,
                                result.person_id,
                                display_name,
                                result.similarity,
                                result.fingerprint
                            );
                        } else {
                            println!(
                                "face-id track={} person=unknown similarity=unmatched best={:.3} fingerprint={}",
                                job.track_id, result.similarity, result.fingerprint
                            );
                        }
                    } else {
                        println!(
                            "face-id track={} person={} name={:?} similarity={:.3} fingerprint={}",
                            job.track_id,
                            result.person_id,
                            display_name,
                            result.similarity,
                            result.fingerprint
                        );
                    }
                    results
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .insert(job.track_id, result);
                }
                Err(error) => eprintln!("face-id track={} database error: {error}", job.track_id),
            },
            Err(error) => {
                if job.capture.is_some() {
                    if let Some(capture) = &capture {
                        capture
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .release_pending("embedding failed; hold the pose again");
                    }
                }
                eprintln!("face-id track={} error: {error}", job.track_id);
            }
        }
    }
}

#[derive(Debug)]
struct CapturedFaceSample {
    pose: CapturePose,
    embedding: Vec<f32>,
    quality: f32,
}

#[derive(Debug)]
struct Identity {
    id: u64,
    name: Option<String>,
    centroid: Vec<f32>,
    pose_templates: Vec<Vec<f32>>,
    samples: u32,
    fingerprint: String,
}

#[derive(Debug)]
struct IdentityGallery {
    connection: Connection,
    database_path: PathBuf,
    identities: Vec<Identity>,
    writable: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FaceIdentityMatch {
    pub person_id: u64,
    pub name: Option<String>,
    pub similarity: f32,
    pub fingerprint: String,
    pub is_new: bool,
    pub persisted: bool,
}

impl FaceIdentityMatch {
    pub fn display_name(&self) -> String {
        if !self.persisted {
            return "Unknown".to_owned();
        }
        self.name
            .as_deref()
            .filter(|name| !name.trim().is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("Person {}", self.person_id))
    }
}

impl IdentityGallery {
    fn validate_capture_target(&self, target: &FaceCaptureTarget) -> Result<(), String> {
        match target {
            FaceCaptureTarget::New { name } => {
                let name = name.trim();
                if name.is_empty() {
                    return Err("capture name cannot be empty".to_owned());
                }
                if let Some(identity) = self.identities.iter().find(|identity| {
                    identity
                        .name
                        .as_deref()
                        .is_some_and(|stored| stored.eq_ignore_ascii_case(name))
                }) {
                    return Err(format!(
                        "identity name {name:?} already belongs to person {}; use --capture --person {}",
                        identity.id, identity.id
                    ));
                }
                Ok(())
            }
            FaceCaptureTarget::Existing { person_id } => self
                .identities
                .iter()
                .any(|identity| identity.id == *person_id)
                .then_some(())
                .ok_or_else(|| {
                    format!("person {person_id} does not exist in the identity gallery")
                }),
        }
    }

    fn open(
        database_path: &Path,
        model_checksum: &str,
        mode: FaceIdDatabaseMode,
    ) -> Result<Self, String> {
        match mode {
            FaceIdDatabaseMode::ReadWrite => {
                if let Some(parent) = database_path
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                {
                    std::fs::create_dir_all(parent).map_err(|error| {
                        format!(
                            "could not create face-ID database directory {}: {error}",
                            parent.display()
                        )
                    })?;
                }
                let connection = Connection::open(database_path).map_err(|error| {
                    format!(
                        "could not open face-ID database {}: {error}",
                        database_path.display()
                    )
                })?;
                Self::initialize(connection, database_path.to_owned(), model_checksum)
            }
            FaceIdDatabaseMode::ReadOnly => {
                let connection = Connection::open_with_flags(
                    database_path,
                    OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
                )
                .map_err(|error| {
                    format!(
                        "could not open face-ID database {} read-only: {error}",
                        database_path.display()
                    )
                })?;
                Self::initialize_read_only(connection, database_path.to_owned(), model_checksum)
            }
        }
    }

    #[cfg(test)]
    fn open_in_memory(model_checksum: &str) -> Result<Self, String> {
        let connection = Connection::open_in_memory()
            .map_err(|error| format!("could not create test face-ID database: {error}"))?;
        Self::initialize(connection, PathBuf::from(":memory:"), model_checksum)
    }

    fn initialize(
        mut connection: Connection,
        database_path: PathBuf,
        model_checksum: &str,
    ) -> Result<Self, String> {
        connection
            .busy_timeout(Duration::from_secs(2))
            .map_err(|error| format!("could not configure face-ID database: {error}"))?;
        connection
            .pragma_update(None, "journal_mode", "WAL")
            .map_err(|error| format!("could not enable SQLite WAL mode: {error}"))?;
        connection
            .pragma_update(None, "foreign_keys", true)
            .map_err(|error| format!("could not enable SQLite foreign keys: {error}"))?;
        connection
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS face_id_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS face_identities (
                    person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    fingerprint TEXT NOT NULL UNIQUE,
                    centroid BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL CHECK (embedding_dim > 0),
                    samples INTEGER NOT NULL CHECK (samples > 0),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS face_identity_samples (
                    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL REFERENCES face_identities(person_id) ON DELETE CASCADE,
                    pose TEXT NOT NULL CHECK (pose IN ('center', 'left', 'right', 'up', 'down')),
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL CHECK (embedding_dim > 0),
                    quality REAL NOT NULL CHECK (quality >= 0.0 AND quality <= 1.0),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );",
            )
            .map_err(|error| format!("could not initialize face-ID database schema: {error}"))?;

        migrate_schema(&mut connection)?;
        validate_or_insert_metadata(&connection, "embedding_format", EMBEDDING_FORMAT)?;
        validate_or_insert_metadata(&connection, "model_sha256", model_checksum)?;

        let identities = load_identities(&connection)?;
        Ok(Self {
            connection,
            database_path,
            identities,
            writable: true,
        })
    }

    fn initialize_read_only(
        connection: Connection,
        database_path: PathBuf,
        model_checksum: &str,
    ) -> Result<Self, String> {
        connection
            .busy_timeout(Duration::from_secs(2))
            .map_err(|error| format!("could not configure face-ID database: {error}"))?;
        validate_metadata(
            &connection,
            "schema_version",
            &DATABASE_SCHEMA_VERSION.to_string(),
        )?;
        validate_metadata(&connection, "embedding_format", EMBEDDING_FORMAT)?;
        validate_metadata(&connection, "model_sha256", model_checksum)?;
        let identities = load_identities(&connection)?;
        Ok(Self {
            connection,
            database_path,
            identities,
            writable: false,
        })
    }

    fn identify(&mut self, embedding: Vec<f32>) -> Result<FaceIdentityMatch, String> {
        let best = self
            .identities
            .iter()
            .enumerate()
            .map(|(index, identity)| {
                let similarity = std::iter::once(identity.centroid.as_slice())
                    .chain(identity.pose_templates.iter().map(Vec::as_slice))
                    .map(|template| cosine_similarity(template, &embedding))
                    .max_by(f32::total_cmp)
                    .unwrap_or(f32::NEG_INFINITY);
                (index, similarity)
            })
            .max_by(|left, right| left.1.total_cmp(&right.1));

        if let Some((index, similarity)) =
            best.filter(|(_, score)| *score >= SFACE_COSINE_THRESHOLD)
        {
            let identity = &mut self.identities[index];
            if self.writable {
                let database_display = self.database_path.display().to_string();
                let samples = identity.samples.saturating_add(1);
                // Slowly adapt to lighting/pose while keeping the original fingerprint stable.
                let mut centroid = identity.centroid.clone();
                for (centroid, sample) in centroid.iter_mut().zip(&embedding) {
                    *centroid = *centroid * 0.85 + *sample * 0.15;
                }
                let norm = centroid
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    .sqrt();
                if norm > f32::EPSILON {
                    for value in &mut centroid {
                        *value /= norm;
                    }
                }
                self.connection
                    .execute(
                        "UPDATE face_identities
                         SET centroid = ?1, embedding_dim = ?2, samples = ?3,
                             updated_at = CURRENT_TIMESTAMP
                         WHERE person_id = ?4",
                        params![
                            embedding_to_blob(&centroid),
                            centroid.len() as i64,
                            samples as i64,
                            identity.id as i64
                        ],
                    )
                    .map_err(|error| {
                        format!("could not update identity in {database_display}: {error}")
                    })?;
                identity.centroid = centroid;
                identity.samples = samples;
            }
            return Ok(FaceIdentityMatch {
                person_id: identity.id,
                name: identity.name.clone(),
                similarity,
                fingerprint: identity.fingerprint.clone(),
                is_new: false,
                persisted: true,
            });
        }

        let fingerprint = embedding_fingerprint(&embedding);
        if !self.writable {
            return Ok(FaceIdentityMatch {
                person_id: 0,
                name: None,
                similarity: best.map_or(0.0, |(_, score)| score),
                fingerprint,
                is_new: true,
                persisted: false,
            });
        }
        self.connection
            .execute(
                "INSERT INTO face_identities
                 (fingerprint, centroid, embedding_dim, samples)
                 VALUES (?1, ?2, ?3, 1)",
                params![
                    &fingerprint,
                    embedding_to_blob(&embedding),
                    embedding.len() as i64
                ],
            )
            .map_err(|error| self.database_error("insert identity", error))?;
        let person_id = self.connection.last_insert_rowid() as u64;
        let identity = Identity {
            id: person_id,
            name: None,
            fingerprint,
            centroid: embedding,
            pose_templates: Vec::new(),
            samples: 1,
        };
        let result = FaceIdentityMatch {
            person_id: identity.id,
            name: identity.name.clone(),
            similarity: best.map_or(0.0, |(_, score)| score),
            fingerprint: identity.fingerprint.clone(),
            is_new: true,
            persisted: true,
        };
        self.identities.push(identity);
        Ok(result)
    }

    fn commit_capture(
        &mut self,
        target: &FaceCaptureTarget,
        captured: &[CapturedFaceSample],
    ) -> Result<FaceIdentityMatch, String> {
        if !self.writable {
            return Err("guided capture requires a writable face-ID database".to_owned());
        }
        if captured.len() != CapturePose::ALL.len() * CAPTURE_SAMPLES_PER_POSE {
            return Err(format!(
                "guided capture has {} samples; expected {}",
                captured.len(),
                CapturePose::ALL.len() * CAPTURE_SAMPLES_PER_POSE
            ));
        }
        let centroid = mean_embedding(captured.iter().map(|sample| sample.embedding.as_slice()))?;
        let (existing_index, person_id, name, fingerprint, is_new) = match target {
            FaceCaptureTarget::New { name } => {
                let name = name.trim();
                if name.is_empty() {
                    return Err("capture name cannot be empty".to_owned());
                }
                (
                    None,
                    None,
                    Some(name.to_owned()),
                    embedding_fingerprint(&centroid),
                    true,
                )
            }
            FaceCaptureTarget::Existing { person_id } => {
                let index = self
                    .identities
                    .iter()
                    .position(|identity| identity.id == *person_id)
                    .ok_or_else(|| format!("person {person_id} does not exist"))?;
                let identity = &self.identities[index];
                (
                    Some(index),
                    Some(*person_id),
                    identity.name.clone(),
                    identity.fingerprint.clone(),
                    false,
                )
            }
        };

        let transaction = self
            .connection
            .transaction()
            .map_err(|error| format!("could not begin capture transaction: {error}"))?;
        let person_id = if let Some(person_id) = person_id {
            let samples = self.identities[existing_index.expect("existing identity")]
                .samples
                .saturating_add(captured.len() as u32);
            transaction
                .execute(
                    "UPDATE face_identities
                     SET centroid = ?1, embedding_dim = ?2, samples = ?3,
                         updated_at = CURRENT_TIMESTAMP
                     WHERE person_id = ?4",
                    params![
                        embedding_to_blob(&centroid),
                        centroid.len() as i64,
                        samples as i64,
                        person_id as i64
                    ],
                )
                .map_err(|error| format!("could not update captured identity: {error}"))?;
            person_id
        } else {
            transaction
                .execute(
                    "INSERT INTO face_identities
                     (name, fingerprint, centroid, embedding_dim, samples)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        name.as_deref(),
                        &fingerprint,
                        embedding_to_blob(&centroid),
                        centroid.len() as i64,
                        captured.len() as i64
                    ],
                )
                .map_err(|error| format!("could not insert captured identity: {error}"))?;
            transaction.last_insert_rowid() as u64
        };

        {
            let mut insert = transaction
                .prepare(
                    "INSERT INTO face_identity_samples
                     (person_id, pose, embedding, embedding_dim, quality)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                )
                .map_err(|error| format!("could not prepare captured sample insert: {error}"))?;
            for sample in captured {
                insert
                    .execute(params![
                        person_id as i64,
                        sample.pose.database_label(),
                        embedding_to_blob(&sample.embedding),
                        sample.embedding.len() as i64,
                        sample.quality
                    ])
                    .map_err(|error| format!("could not insert captured sample: {error}"))?;
            }
        }
        transaction
            .commit()
            .map_err(|error| format!("could not commit guided capture: {error}"))?;
        let pose_templates = load_pose_templates(&self.connection, person_id)?;

        if let Some(index) = existing_index {
            let identity = &mut self.identities[index];
            identity.centroid = centroid;
            identity.pose_templates = pose_templates;
            identity.samples = identity.samples.saturating_add(captured.len() as u32);
        } else {
            self.identities.push(Identity {
                id: person_id,
                name: name.clone(),
                centroid,
                pose_templates,
                samples: captured.len() as u32,
                fingerprint: fingerprint.clone(),
            });
        }
        Ok(FaceIdentityMatch {
            person_id,
            name,
            similarity: 1.0,
            fingerprint,
            is_new,
            persisted: true,
        })
    }

    fn database_error(&self, operation: &str, error: rusqlite::Error) -> String {
        format!(
            "could not {operation} in {}: {error}",
            self.database_path.display()
        )
    }
}

fn validate_metadata(connection: &Connection, key: &str, expected: &str) -> Result<(), String> {
    let actual = connection
        .query_row(
            "SELECT value FROM face_id_metadata WHERE key = ?1",
            [key],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|error| format!("could not read face-ID database metadata: {error}"))?;
    match actual {
        Some(actual) if actual == expected => Ok(()),
        Some(actual) => Err(format!(
            "face-ID database {key} mismatch: stored={actual}, expected={expected}"
        )),
        None => Err(format!("face-ID database is missing metadata key {key}")),
    }
}

fn migrate_schema(connection: &mut Connection) -> Result<(), String> {
    let stored = connection
        .query_row(
            "SELECT value FROM face_id_metadata WHERE key = 'schema_version'",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|error| format!("could not read face-ID schema version: {error}"))?;

    match stored.as_deref() {
        None => connection
            .execute(
                "INSERT INTO face_id_metadata (key, value) VALUES ('schema_version', ?1)",
                [DATABASE_SCHEMA_VERSION.to_string()],
            )
            .map(|_| ())
            .map_err(|error| format!("could not initialize face-ID schema version: {error}")),
        Some("1") => {
            let transaction = connection
                .transaction()
                .map_err(|error| format!("could not start face-ID schema migration: {error}"))?;
            transaction
                .execute_batch("ALTER TABLE face_identities ADD COLUMN name TEXT;")
                .map_err(|error| format!("could not add identity name column: {error}"))?;
            transaction
                .execute(
                    "UPDATE face_id_metadata SET value = ?1 WHERE key = 'schema_version'",
                    [DATABASE_SCHEMA_VERSION.to_string()],
                )
                .map_err(|error| format!("could not update face-ID schema version: {error}"))?;
            transaction
                .commit()
                .map_err(|error| format!("could not commit face-ID schema migration: {error}"))
        }
        Some("2") => connection
            .execute(
                "UPDATE face_id_metadata SET value = ?1 WHERE key = 'schema_version'",
                [DATABASE_SCHEMA_VERSION.to_string()],
            )
            .map(|_| ())
            .map_err(|error| format!("could not migrate face-ID schema to version 3: {error}")),
        Some("3") => Ok(()),
        Some(version) => Err(format!(
            "unsupported face-ID database schema version {version}; expected {DATABASE_SCHEMA_VERSION}"
        )),
    }
}

fn mean_embedding<'a>(embeddings: impl Iterator<Item = &'a [f32]>) -> Result<Vec<f32>, String> {
    let mut count = 0usize;
    let mut sum = Vec::<f32>::new();
    for embedding in embeddings {
        if sum.is_empty() {
            sum.resize(embedding.len(), 0.0);
        }
        if embedding.len() != sum.len() {
            return Err("captured embeddings have inconsistent dimensions".to_owned());
        }
        for (sum, value) in sum.iter_mut().zip(embedding) {
            *sum += *value;
        }
        count += 1;
    }
    if count == 0 {
        return Err("guided capture has no embeddings".to_owned());
    }
    for value in &mut sum {
        *value /= count as f32;
    }
    normalize_embedding(&sum)
}

fn validate_or_insert_metadata(
    connection: &Connection,
    key: &str,
    expected: &str,
) -> Result<(), String> {
    let actual = connection
        .query_row(
            "SELECT value FROM face_id_metadata WHERE key = ?1",
            [key],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|error| format!("could not read face-ID database metadata: {error}"))?;
    match actual {
        Some(actual) if actual != expected => Err(format!(
            "face-ID database {key} mismatch: stored={actual}, expected={expected}; use a different --face-id-db path"
        )),
        Some(_) => Ok(()),
        None => connection
            .execute(
                "INSERT INTO face_id_metadata (key, value) VALUES (?1, ?2)",
                params![key, expected],
            )
            .map(|_| ())
            .map_err(|error| format!("could not write face-ID database metadata: {error}")),
    }
}

fn load_identities(connection: &Connection) -> Result<Vec<Identity>, String> {
    let mut statement = connection
        .prepare(
            "SELECT person_id, name, fingerprint, centroid, embedding_dim, samples
             FROM face_identities ORDER BY person_id",
        )
        .map_err(|error| format!("could not prepare face-ID gallery query: {error}"))?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Vec<u8>>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)?,
            ))
        })
        .map_err(|error| format!("could not query face-ID gallery: {error}"))?;
    let mut identities = Vec::new();
    for row in rows {
        let (id, name, fingerprint, blob, dimension, samples) =
            row.map_err(|error| format!("could not read face-ID identity: {error}"))?;
        if id <= 0 || dimension <= 0 || samples <= 0 {
            return Err("face-ID database contains invalid identity metadata".to_owned());
        }
        identities.push(Identity {
            id: id as u64,
            name,
            fingerprint,
            centroid: embedding_from_blob(&blob, dimension as usize)?,
            pose_templates: Vec::new(),
            samples: samples as u32,
        });
    }
    drop(statement);
    for identity in &mut identities {
        identity.pose_templates = load_pose_templates(connection, identity.id)?;
    }
    Ok(identities)
}

fn load_pose_templates(connection: &Connection, person_id: u64) -> Result<Vec<Vec<f32>>, String> {
    let mut statement = connection
        .prepare(
            "SELECT pose, embedding, embedding_dim
             FROM face_identity_samples
             WHERE person_id = ?1
             ORDER BY pose, sample_id",
        )
        .map_err(|error| format!("could not prepare pose-template query: {error}"))?;
    let rows = statement
        .query_map([person_id as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })
        .map_err(|error| format!("could not query pose templates: {error}"))?;
    let mut grouped = std::collections::BTreeMap::<String, Vec<Vec<f32>>>::new();
    for row in rows {
        let (pose, blob, dimension) =
            row.map_err(|error| format!("could not read pose template: {error}"))?;
        if dimension <= 0 {
            return Err("pose template has an invalid embedding dimension".to_owned());
        }
        grouped
            .entry(pose)
            .or_default()
            .push(embedding_from_blob(&blob, dimension as usize)?);
    }
    grouped
        .into_values()
        .map(|samples| mean_embedding(samples.iter().map(Vec::as_slice)))
        .collect()
}

fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    embedding
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn embedding_from_blob(blob: &[u8], dimension: usize) -> Result<Vec<f32>, String> {
    let expected_bytes = dimension
        .checked_mul(size_of::<f32>())
        .ok_or("face-ID database embedding dimension is too large")?;
    if blob.len() != expected_bytes {
        return Err(format!(
            "face-ID database embedding has {} bytes for dimension {dimension}",
            blob.len()
        ));
    }
    let embedding = blob
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    normalize_embedding(&embedding)
}

fn file_sha256(path: &Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|error| format!("could not open model {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("could not hash model {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn normalize_embedding(values: &[f32]) -> Result<Vec<f32>, String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("face-ID model returned an empty or non-finite embedding".to_owned());
    }
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return Err("face-ID model returned a zero embedding".to_owned());
    }
    Ok(values.iter().map(|value| value / norm).collect())
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() {
        return f32::NEG_INFINITY;
    }
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn embedding_fingerprint(embedding: &[f32]) -> String {
    let quantized = embedding.iter().map(|value| {
        let value = (value.clamp(-1.0, 1.0) * 127.0).round() as i8;
        value as u8
    });
    let digest = Sha256::digest(quantized.collect::<Vec<_>>());
    digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Creates the SFace RGB/NCHW tensor by aligning Vision's five landmark groups
/// against the standard 112×112 SFace template. `bgra` must be a top-left-origin
/// 32-bit BGRA image.
pub fn aligned_sface_tensor_bgra(
    bgra: &[u8],
    width: usize,
    height: usize,
    bytes_per_row: usize,
    face: &FaceGeometry,
) -> Result<Vec<f32>, String> {
    if width == 0 || height == 0 || bytes_per_row < width * 4 || bgra.len() < bytes_per_row * height
    {
        return Err("invalid BGRA pixel buffer layout".to_owned());
    }
    let source = five_landmarks(face, width as f32, height as f32)?;
    let transform = similarity_transform(&source, &SFACE_TEMPLATE)?;
    let mut tensor = vec![0.0; 3 * SFACE_WIDTH * SFACE_HEIGHT];
    let plane = SFACE_WIDTH * SFACE_HEIGHT;

    for destination_y in 0..SFACE_HEIGHT {
        for destination_x in 0..SFACE_WIDTH {
            let (source_x, source_y) =
                transform.inverse(destination_x as f32 + 0.5, destination_y as f32 + 0.5);
            let [red, green, blue] = sample_bgra(
                bgra,
                width,
                height,
                bytes_per_row,
                source_x - 0.5,
                source_y - 0.5,
            );
            let offset = destination_y * SFACE_WIDTH + destination_x;
            tensor[offset] = red;
            tensor[plane + offset] = green;
            tensor[2 * plane + offset] = blue;
        }
    }
    Ok(tensor)
}

fn five_landmarks(face: &FaceGeometry, width: f32, height: f32) -> Result<[[f32; 2]; 5], String> {
    let mut eyes = [
        region_center(face, LandmarkKind::LeftEye).ok_or("left eye landmark is unavailable")?,
        region_center(face, LandmarkKind::RightEye).ok_or("right eye landmark is unavailable")?,
    ];
    eyes.sort_by(|left, right| left.x.total_cmp(&right.x));
    let nose = region_center(face, LandmarkKind::Nose)
        .or_else(|| region_center(face, LandmarkKind::NoseCrest))
        .ok_or("nose landmark is unavailable")?;
    let lips = face
        .landmarks
        .iter()
        .find(|region| region.kind == LandmarkKind::OuterLips)
        .ok_or("outer-lip landmarks are unavailable")?;
    let left_mouth = lips
        .points
        .iter()
        .min_by(|left, right| left.x.total_cmp(&right.x))
        .copied()
        .ok_or("outer-lip landmarks are empty")?;
    let right_mouth = lips
        .points
        .iter()
        .max_by(|left, right| left.x.total_cmp(&right.x))
        .copied()
        .ok_or("outer-lip landmarks are empty")?;

    Ok([
        image_pixel(eyes[0], width, height),
        image_pixel(eyes[1], width, height),
        image_pixel(nose, width, height),
        image_pixel(left_mouth, width, height),
        image_pixel(right_mouth, width, height),
    ])
}

fn region_center(face: &FaceGeometry, kind: LandmarkKind) -> Option<Point> {
    let points = &face
        .landmarks
        .iter()
        .find(|region| region.kind == kind)?
        .points;
    if points.is_empty() {
        return None;
    }
    let (x, y) = points
        .iter()
        .fold((0.0, 0.0), |(x, y), point| (x + point.x, y + point.y));
    Some(Point {
        x: x / points.len() as f64,
        y: y / points.len() as f64,
    })
}

fn image_pixel(point: Point, width: f32, height: f32) -> [f32; 2] {
    [point.x as f32 * width, (1.0 - point.y as f32) * height]
}

#[derive(Clone, Copy, Debug)]
struct SimilarityTransform {
    a: f32,
    b: f32,
    tx: f32,
    ty: f32,
}

impl SimilarityTransform {
    fn inverse(self, x: f32, y: f32) -> (f32, f32) {
        let x = x - self.tx;
        let y = y - self.ty;
        let determinant = self.a * self.a + self.b * self.b;
        (
            (self.a * x + self.b * y) / determinant,
            (-self.b * x + self.a * y) / determinant,
        )
    }
}

fn similarity_transform(
    source: &[[f32; 2]; 5],
    destination: &[[f32; 2]; 5],
) -> Result<SimilarityTransform, String> {
    let source_mean = mean_point(source);
    let destination_mean = mean_point(destination);
    let mut denominator = 0.0;
    let mut real = 0.0;
    let mut imaginary = 0.0;
    for (source, destination) in source.iter().zip(destination) {
        let sx = source[0] - source_mean[0];
        let sy = source[1] - source_mean[1];
        let dx = destination[0] - destination_mean[0];
        let dy = destination[1] - destination_mean[1];
        denominator += sx * sx + sy * sy;
        real += sx * dx + sy * dy;
        imaginary += sx * dy - sy * dx;
    }
    if denominator <= f32::EPSILON {
        return Err("face landmarks are degenerate".to_owned());
    }
    let a = real / denominator;
    let b = imaginary / denominator;
    if a * a + b * b <= f32::EPSILON {
        return Err("face alignment transform is degenerate".to_owned());
    }
    Ok(SimilarityTransform {
        a,
        b,
        tx: destination_mean[0] - a * source_mean[0] + b * source_mean[1],
        ty: destination_mean[1] - b * source_mean[0] - a * source_mean[1],
    })
}

fn mean_point(points: &[[f32; 2]; 5]) -> [f32; 2] {
    let sum = points.iter().fold([0.0, 0.0], |sum, point| {
        [sum[0] + point[0], sum[1] + point[1]]
    });
    [sum[0] / points.len() as f32, sum[1] / points.len() as f32]
}

fn sample_bgra(
    pixels: &[u8],
    width: usize,
    height: usize,
    bytes_per_row: usize,
    x: f32,
    y: f32,
) -> [f32; 3] {
    if x < 0.0 || y < 0.0 || x > (width - 1) as f32 || y > (height - 1) as f32 {
        return [0.0; 3];
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let read = |px: usize, py: usize| {
        let offset = py * bytes_per_row + px * 4;
        [
            pixels[offset + 2] as f32,
            pixels[offset + 1] as f32,
            pixels[offset] as f32,
        ]
    };
    let top_left = read(x0, y0);
    let top_right = read(x1, y0);
    let bottom_left = read(x0, y1);
    let bottom_right = read(x1, y1);
    let mut result = [0.0; 3];
    for channel in 0..3 {
        let top = top_left[channel] * (1.0 - fx) + top_right[channel] * fx;
        let bottom = bottom_left[channel] * (1.0 - fx) + bottom_right[channel] * fx;
        result[channel] = top * (1.0 - fy) + bottom * fy;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_transform_maps_source_to_destination() {
        let source = SFACE_TEMPLATE.map(|[x, y]| [x * 2.0 + 10.0, y * 2.0 - 7.0]);
        let transform = similarity_transform(&source, &SFACE_TEMPLATE).unwrap();
        for (source, destination) in source.iter().zip(SFACE_TEMPLATE) {
            let x = transform.a * source[0] - transform.b * source[1] + transform.tx;
            let y = transform.b * source[0] + transform.a * source[1] + transform.ty;
            assert!((x - destination[0]).abs() < 0.001);
            assert!((y - destination[1]).abs() < 0.001);
        }
    }

    #[test]
    fn gallery_reuses_person_id_above_threshold() {
        let mut gallery = IdentityGallery::open_in_memory("test-model").unwrap();
        let first = gallery
            .identify(normalize_embedding(&[1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let second = gallery
            .identify(normalize_embedding(&[0.99, 0.05, 0.0]).unwrap())
            .unwrap();
        assert_eq!(first.person_id, second.person_id);
        assert_eq!(first.fingerprint, second.fingerprint);
        assert!(!second.is_new);
    }

    #[test]
    fn gallery_splits_different_embeddings() {
        let mut gallery = IdentityGallery::open_in_memory("test-model").unwrap();
        let first = gallery
            .identify(normalize_embedding(&[1.0, 0.0]).unwrap())
            .unwrap();
        let second = gallery
            .identify(normalize_embedding(&[0.0, 1.0]).unwrap())
            .unwrap();
        assert_ne!(first.person_id, second.person_id);
        assert!(second.is_new);
    }

    #[test]
    fn sqlite_gallery_survives_reopen_and_rejects_another_model() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "facefeature-gallery-{}-{unique}.sqlite3",
            std::process::id()
        ));
        let embedding = normalize_embedding(&[1.0, 0.2, 0.1]).unwrap();

        let first = {
            let mut gallery =
                IdentityGallery::open(&path, "model-a", FaceIdDatabaseMode::ReadWrite).unwrap();
            let result = gallery.identify(embedding.clone()).unwrap();
            gallery
                .connection
                .execute(
                    "UPDATE face_identities SET name = 'Alice' WHERE person_id = ?1",
                    [result.person_id as i64],
                )
                .unwrap();
            result
        };
        {
            let mut gallery =
                IdentityGallery::open(&path, "model-a", FaceIdDatabaseMode::ReadOnly).unwrap();
            let matched = gallery.identify(embedding.clone()).unwrap();
            assert_eq!(matched.person_id, first.person_id);
            assert!(matched.persisted);
            assert!(!matched.is_new);

            let unknown = gallery
                .identify(normalize_embedding(&[0.0, 1.0, 0.0]).unwrap())
                .unwrap();
            assert_eq!(unknown.person_id, 0);
            assert!(!unknown.persisted);
            assert_eq!(unknown.display_name(), "Unknown");

            let samples: i64 = gallery
                .connection
                .query_row(
                    "SELECT samples FROM face_identities WHERE person_id = ?1",
                    [first.person_id as i64],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(samples, 1);
            assert_eq!(gallery.identities.len(), 1);
        }
        let second = {
            let mut gallery =
                IdentityGallery::open(&path, "model-a", FaceIdDatabaseMode::ReadWrite).unwrap();
            assert_eq!(gallery.identities.len(), 1);
            gallery.identify(embedding).unwrap()
        };
        assert_eq!(first.person_id, second.person_id);
        assert_eq!(first.fingerprint, second.fingerprint);
        assert_eq!(second.name.as_deref(), Some("Alice"));
        assert_eq!(second.display_name(), "Alice");
        assert!(!second.is_new);

        let mismatch =
            IdentityGallery::open(&path, "model-b", FaceIdDatabaseMode::ReadWrite).unwrap_err();
        assert!(mismatch.contains("model_sha256 mismatch"));

        for suffix in ["", "-shm", "-wal"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }

    #[test]
    fn schema_one_database_is_migrated_with_a_name_column() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE face_id_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                INSERT INTO face_id_metadata VALUES ('schema_version', '1');
                CREATE TABLE face_identities (
                    person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint TEXT NOT NULL UNIQUE,
                    centroid BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    samples INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );",
            )
            .unwrap();

        let gallery = IdentityGallery::initialize(
            connection,
            PathBuf::from(":migration-test:"),
            "test-model",
        )
        .unwrap();
        let name_columns: i64 = gallery
            .connection
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('face_identities') WHERE name = 'name'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let version: String = gallery
            .connection
            .query_row(
                "SELECT value FROM face_id_metadata WHERE key = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(name_columns, 1);
        let sample_table: i64 = gallery
            .connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'face_identity_samples'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(version, "3");
        assert_eq!(sample_table, 1);
    }

    #[test]
    fn guided_capture_commits_five_pose_templates_atomically() {
        let mut gallery = IdentityGallery::open_in_memory("capture-model").unwrap();
        let mut captured = Vec::new();
        for (pose_index, pose) in CapturePose::ALL.into_iter().enumerate() {
            for sample_index in 0..CAPTURE_SAMPLES_PER_POSE {
                let embedding = normalize_embedding(&[
                    1.0,
                    pose_index as f32 * 0.02,
                    sample_index as f32 * 0.01,
                ])
                .unwrap();
                captured.push(CapturedFaceSample {
                    pose,
                    embedding,
                    quality: 0.9,
                });
            }
        }

        let identity = gallery
            .commit_capture(
                &FaceCaptureTarget::New {
                    name: "Radit".to_owned(),
                },
                &captured,
            )
            .unwrap();
        assert_eq!(identity.person_id, 1);
        assert_eq!(identity.name.as_deref(), Some("Radit"));
        assert_eq!(gallery.identities[0].pose_templates.len(), 5);

        let (sample_count, pose_count): (i64, i64) = gallery
            .connection
            .query_row(
                "SELECT COUNT(*), COUNT(DISTINCT pose) FROM face_identity_samples WHERE person_id = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(sample_count, 15);
        assert_eq!(pose_count, 5);

        gallery
            .commit_capture(&FaceCaptureTarget::Existing { person_id: 1 }, &captured)
            .unwrap();
        let updated_count: i64 = gallery
            .connection
            .query_row(
                "SELECT COUNT(*) FROM face_identity_samples WHERE person_id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(updated_count, 30);
        assert_eq!(gallery.identities[0].name.as_deref(), Some("Radit"));
        assert_eq!(gallery.identities[0].pose_templates.len(), 5);
    }

    #[test]
    fn guided_pose_ranges_calibrate_opposite_directions() {
        assert!(pose_matches(CapturePose::Center, 4.0, -3.0, None, None));
        assert!(pose_matches(CapturePose::Left, -25.0, 2.0, None, None));
        assert!(pose_matches(
            CapturePose::Right,
            25.0,
            2.0,
            Some(-1.0),
            None
        ));
        assert!(pose_matches(CapturePose::Up, 2.0, 20.0, None, None));
        assert!(pose_matches(CapturePose::Down, 2.0, -20.0, None, Some(1.0)));
        assert!(!pose_matches(
            CapturePose::Right,
            -25.0,
            2.0,
            Some(-1.0),
            None
        ));
        assert!(pose_matches(CapturePose::Left, -45.0, 28.0, None, None));
        assert_eq!(
            pose_guidance(CapturePose::Left, -6.0, 0.0, None, None),
            "turn farther toward the prompt"
        );
        assert_eq!(
            pose_guidance(CapturePose::Left, -58.0, 0.0, None, None),
            "turn slightly back toward the camera"
        );
    }

    #[test]
    fn straight_capture_uses_landmark_pitch_when_vision_pitch_is_missing() {
        let face = capture_test_face(0.0, None, 0.5);
        assert!(landmark_pitch_degrees(&face).unwrap().abs() < f64::EPSILON);
        let mut state = GuidedCaptureState::new(FaceCaptureTarget::New {
            name: "Test".to_owned(),
        });
        let started = Instant::now();
        assert!(state.observe(&face, started).is_none());
        let request = state
            .observe(
                &face,
                started + CAPTURE_HOLD_DURATION + Duration::from_millis(1),
            )
            .expect("straight pose should be captured without native pitch");
        assert_eq!(request.pose, CapturePose::Center);
        assert!(request.pitch_degrees.abs() < f64::EPSILON);
        assert!(request.vertical_proxy.is_some());
        assert!(!state.sample_completed(request));
        assert_eq!(state.samples, 1);
        assert!(state.neutral_vertical.is_some());
        assert!(state.status().pitch_estimated);
    }

    fn capture_test_face(
        yaw_degrees: f64,
        pitch_degrees: Option<f64>,
        nose_y: f64,
    ) -> FaceGeometry {
        let bounding_box = crate::BoundingBox {
            x: 0.25,
            y: 0.15,
            width: 0.5,
            height: 0.65,
        };
        let landmarks = vec![
            crate::LandmarkRegion {
                kind: LandmarkKind::LeftEye,
                points: vec![Point { x: 0.4, y: 0.65 }],
            },
            crate::LandmarkRegion {
                kind: LandmarkKind::RightEye,
                points: vec![Point { x: 0.6, y: 0.65 }],
            },
            crate::LandmarkRegion {
                kind: LandmarkKind::Nose,
                points: vec![Point { x: 0.5, y: nose_y }],
            },
            crate::LandmarkRegion {
                kind: LandmarkKind::OuterLips,
                points: vec![Point { x: 0.44, y: 0.35 }, Point { x: 0.56, y: 0.35 }],
            },
        ];
        FaceGeometry {
            confidence: 0.98,
            landmark_confidence: 0.98,
            bounding_box,
            roll_radians: Some(0.0),
            yaw_radians: Some(yaw_degrees.to_radians()),
            pitch_radians: pitch_degrees.map(f64::to_radians),
            measurements: FaceGeometry::calculate_measurements(bounding_box, &landmarks),
            landmarks,
        }
    }

    #[test]
    fn bgra_sampling_outputs_rgb() {
        let pixels = [10_u8, 20, 30, 255];
        assert_eq!(sample_bgra(&pixels, 1, 1, 4, 0.0, 0.0), [30.0, 20.0, 10.0]);
    }

    #[test]
    #[ignore = "loads the 37 MB model and compiles its Core ML graph"]
    fn bundled_model_runs_with_core_ml() {
        let mut engine = FaceEmbeddingEngine::new(
            Path::new("models/face_recognition_sface_2021dec.onnx"),
            Path::new("target/face-id-coreml-test-cache"),
        )
        .unwrap();
        let embedding = engine
            .embed(vec![127.0; 3 * SFACE_WIDTH * SFACE_HEIGHT])
            .unwrap();
        assert!(embedding.len() >= 128);
        assert!((cosine_similarity(&embedding, &embedding) - 1.0).abs() < 0.001);
    }
}
