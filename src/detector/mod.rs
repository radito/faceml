use std::error::Error;
use std::fmt;
use std::path::Path;

use crate::model::Detection;

#[cfg(target_os = "macos")]
pub mod apple_vision;

pub trait FaceGeometryDetector {
    fn name(&self) -> &'static str;
    fn detect_path(&self, image_path: &Path) -> Result<Detection, DetectorError>;
}

#[derive(Debug)]
pub enum DetectorError {
    InvalidInput(String),
    Backend(String),
    UnsupportedPlatform,
}

impl fmt::Display for DetectorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(formatter, "invalid input: {message}"),
            Self::Backend(message) => write!(formatter, "detector backend failed: {message}"),
            Self::UnsupportedPlatform => formatter.write_str(
                "no detector backend is available on this platform; the current backend requires macOS",
            ),
        }
    }
}

impl Error for DetectorError {}

pub fn default_detector() -> Result<Box<dyn FaceGeometryDetector>, DetectorError> {
    #[cfg(target_os = "macos")]
    {
        Ok(Box::new(apple_vision::AppleVisionDetector))
    }

    #[cfg(not(target_os = "macos"))]
    {
        Err(DetectorError::UnsupportedPlatform)
    }
}
