//! Local face geometry detection with platform-specific acceleration backends.

pub mod detector;
#[cfg(target_os = "macos")]
pub mod face_id;
pub mod model;
pub mod tracker;

pub use detector::{DetectorError, FaceGeometryDetector, default_detector};
pub use model::{
    BoundingBox, CoordinateSystem, Detection, FaceGeometry, GeometryMeasurements, LandmarkKind,
    LandmarkRegion, Point,
};
pub use tracker::{FaceTracker, FaceTrackerConfig, TrackedFace};
