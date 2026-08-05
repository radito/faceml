use std::path::Path;

use objc2::AnyThread;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::AnyObject;
use objc2_core_video::CVPixelBuffer;
use objc2_foundation::{NSArray, NSDictionary, NSString, NSURL};
use objc2_vision::{
    VNDetectFaceLandmarksRequest, VNFaceLandmarkRegion2D, VNFaceLandmarks2D, VNImageOption,
    VNImageRequestHandler, VNRequest,
};

use super::{DetectorError, FaceGeometryDetector};
use crate::model::{
    BoundingBox, CoordinateSystem, Detection, FaceGeometry, LandmarkKind, LandmarkRegion, Point,
};

pub struct AppleVisionDetector;

impl AppleVisionDetector {
    /// Detect geometry in a live Core Video frame without copying its pixels.
    pub fn detect_pixel_buffer(
        &self,
        pixel_buffer: &CVPixelBuffer,
    ) -> Result<Detection, DetectorError> {
        autoreleasepool(|_| {
            let options = empty_options();
            let handler = unsafe {
                VNImageRequestHandler::initWithCVPixelBuffer_options(
                    VNImageRequestHandler::alloc(),
                    pixel_buffer,
                    &options,
                )
            };
            detect_with_handler(&handler)
        })
    }
}

impl FaceGeometryDetector for AppleVisionDetector {
    fn name(&self) -> &'static str {
        "apple_vision"
    }

    fn detect_path(&self, image_path: &Path) -> Result<Detection, DetectorError> {
        let image_path = image_path
            .canonicalize()
            .map_err(|error| DetectorError::InvalidInput(error.to_string()))?;
        if !image_path.is_file() {
            return Err(DetectorError::InvalidInput(format!(
                "{} is not a file",
                image_path.display()
            )));
        }

        autoreleasepool(|_| detect_image(&image_path))
    }
}

fn detect_image(image_path: &Path) -> Result<Detection, DetectorError> {
    let path = image_path.to_str().ok_or_else(|| {
        DetectorError::InvalidInput("the image path is not valid UTF-8".to_owned())
    })?;
    let path = NSString::from_str(path);

    let image_url = NSURL::fileURLWithPath(&path);
    let options = empty_options();
    let handler = unsafe {
        VNImageRequestHandler::initWithURL_options(
            VNImageRequestHandler::alloc(),
            &image_url,
            &options,
        )
    };

    detect_with_handler(&handler)
}

fn detect_with_handler(handler: &VNImageRequestHandler) -> Result<Detection, DetectorError> {
    let request = unsafe { VNDetectFaceLandmarksRequest::new() };
    let request_for_handler: Retained<VNRequest> = request.clone().into_super().into_super();
    let requests = NSArray::from_retained_slice(&[request_for_handler]);
    handler
        .performRequests_error(&requests)
        .map_err(|error| DetectorError::Backend(error.to_string()))?;

    let observations = unsafe { request.results() }
        .ok_or_else(|| DetectorError::Backend("Vision returned no results array".to_owned()))?;
    let mut faces = Vec::with_capacity(observations.count());

    for index in 0..observations.count() {
        let observation = observations.objectAtIndex(index);
        let bounds = unsafe { observation.boundingBox() };
        let bounding_box = BoundingBox {
            x: bounds.origin.x,
            y: bounds.origin.y,
            width: bounds.size.width,
            height: bounds.size.height,
        };
        let landmarks = unsafe { observation.landmarks() }
            .ok_or_else(|| DetectorError::Backend("face result has no landmarks".to_owned()))?;
        let regions = extract_regions(&landmarks, bounding_box)?;
        let measurements = FaceGeometry::calculate_measurements(bounding_box, &regions);

        faces.push(FaceGeometry {
            confidence: unsafe { observation.confidence() },
            landmark_confidence: unsafe { landmarks.confidence() },
            bounding_box,
            roll_radians: unsafe { observation.roll() }.map(|number| number.as_f64()),
            yaw_radians: unsafe { observation.yaw() }.map(|number| number.as_f64()),
            pitch_radians: unsafe { observation.pitch() }.map(|number| number.as_f64()),
            measurements,
            landmarks: regions,
        });
    }

    Ok(Detection {
        backend: "apple_vision",
        coordinate_system: CoordinateSystem::NormalizedLowerLeft,
        faces,
    })
}

fn empty_options() -> Retained<NSDictionary<VNImageOption, AnyObject>> {
    NSDictionary::<VNImageOption, AnyObject>::from_slices::<NSString>(&[], &[])
}

fn extract_regions(
    landmarks: &VNFaceLandmarks2D,
    bounds: BoundingBox,
) -> Result<Vec<LandmarkRegion>, DetectorError> {
    // SAFETY: Each returned region is retained for the duration of point extraction.
    let candidates = unsafe {
        [
            (LandmarkKind::AllPoints, landmarks.allPoints()),
            (LandmarkKind::FaceContour, landmarks.faceContour()),
            (LandmarkKind::LeftEye, landmarks.leftEye()),
            (LandmarkKind::RightEye, landmarks.rightEye()),
            (LandmarkKind::LeftEyebrow, landmarks.leftEyebrow()),
            (LandmarkKind::RightEyebrow, landmarks.rightEyebrow()),
            (LandmarkKind::Nose, landmarks.nose()),
            (LandmarkKind::NoseCrest, landmarks.noseCrest()),
            (LandmarkKind::MedianLine, landmarks.medianLine()),
            (LandmarkKind::OuterLips, landmarks.outerLips()),
            (LandmarkKind::InnerLips, landmarks.innerLips()),
            (LandmarkKind::LeftPupil, landmarks.leftPupil()),
            (LandmarkKind::RightPupil, landmarks.rightPupil()),
        ]
    };

    candidates
        .into_iter()
        .filter_map(|(kind, region)| region.map(|region| (kind, region)))
        .map(|(kind, region)| extract_region(kind, &region, bounds))
        .collect()
}

fn extract_region(
    kind: LandmarkKind,
    region: &VNFaceLandmarkRegion2D,
    bounds: BoundingBox,
) -> Result<LandmarkRegion, DetectorError> {
    let point_count = unsafe { region.pointCount() };
    if point_count == 0 {
        return Ok(LandmarkRegion {
            kind,
            points: Vec::new(),
        });
    }

    let pointer = unsafe { region.normalizedPoints() };
    if pointer.is_null() {
        return Err(DetectorError::Backend(format!(
            "Vision returned a null point buffer for {kind:?}"
        )));
    }

    let normalized_points = unsafe { std::slice::from_raw_parts(pointer, point_count) };
    let points = normalized_points
        .iter()
        .map(|point| Point {
            x: bounds.x + point.x * bounds.width,
            y: bounds.y + point.y * bounds.height,
        })
        .collect();

    Ok(LandmarkRegion { kind, points })
}
