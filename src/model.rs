use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn distance(self, other: Self) -> f64 {
        (self.x - other.x).hypot(self.y - other.y)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct BoundingBox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinateSystem {
    /// Coordinates are in `[0, 1]`, with `(0, 0)` at the image's lower-left.
    NormalizedLowerLeft,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LandmarkKind {
    AllPoints,
    FaceContour,
    LeftEye,
    RightEye,
    LeftEyebrow,
    RightEyebrow,
    Nose,
    NoseCrest,
    MedianLine,
    OuterLips,
    InnerLips,
    LeftPupil,
    RightPupil,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LandmarkRegion {
    pub kind: LandmarkKind,
    /// Points in full-image normalized coordinates.
    pub points: Vec<Point>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GeometryMeasurements {
    pub face_aspect_ratio: f64,
    pub inter_eye_distance_face_width: Option<f64>,
    pub mouth_width_face_width: Option<f64>,
    pub eye_to_mouth_distance_face_height: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FaceGeometry {
    pub confidence: f32,
    pub landmark_confidence: f32,
    pub bounding_box: BoundingBox,
    pub roll_radians: Option<f64>,
    pub yaw_radians: Option<f64>,
    pub pitch_radians: Option<f64>,
    pub measurements: GeometryMeasurements,
    pub landmarks: Vec<LandmarkRegion>,
}

impl FaceGeometry {
    pub fn calculate_measurements(
        bounding_box: BoundingBox,
        landmarks: &[LandmarkRegion],
    ) -> GeometryMeasurements {
        let left_eye = region_center(landmarks, LandmarkKind::LeftEye);
        let right_eye = region_center(landmarks, LandmarkKind::RightEye);
        let mouth = region_center(landmarks, LandmarkKind::OuterLips);

        let eye_midpoint = left_eye.zip(right_eye).map(|(left, right)| Point {
            x: (left.x + right.x) / 2.0,
            y: (left.y + right.y) / 2.0,
        });

        GeometryMeasurements {
            face_aspect_ratio: safe_ratio(bounding_box.height, bounding_box.width).unwrap_or(0.0),
            inter_eye_distance_face_width: left_eye
                .zip(right_eye)
                .and_then(|(left, right)| safe_ratio(left.distance(right), bounding_box.width)),
            mouth_width_face_width: region_diameter(landmarks, LandmarkKind::OuterLips)
                .and_then(|width| safe_ratio(width, bounding_box.width)),
            eye_to_mouth_distance_face_height: eye_midpoint
                .zip(mouth)
                .and_then(|(eyes, mouth)| safe_ratio(eyes.distance(mouth), bounding_box.height)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Detection {
    pub backend: &'static str,
    pub coordinate_system: CoordinateSystem,
    pub faces: Vec<FaceGeometry>,
}

fn region<'a>(landmarks: &'a [LandmarkRegion], kind: LandmarkKind) -> Option<&'a LandmarkRegion> {
    landmarks.iter().find(|region| region.kind == kind)
}

fn region_center(landmarks: &[LandmarkRegion], kind: LandmarkKind) -> Option<Point> {
    let points = &region(landmarks, kind)?.points;
    if points.is_empty() {
        return None;
    }

    let sum = points
        .iter()
        .fold(Point { x: 0.0, y: 0.0 }, |sum, point| Point {
            x: sum.x + point.x,
            y: sum.y + point.y,
        });
    Some(Point {
        x: sum.x / points.len() as f64,
        y: sum.y / points.len() as f64,
    })
}

fn region_diameter(landmarks: &[LandmarkRegion], kind: LandmarkKind) -> Option<f64> {
    let points = &region(landmarks, kind)?.points;
    points
        .iter()
        .enumerate()
        .flat_map(|(index, point)| {
            points[index + 1..]
                .iter()
                .map(move |other| point.distance(*other))
        })
        .reduce(f64::max)
}

fn safe_ratio(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator.abs() > f64::EPSILON).then_some(numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measurements_are_relative_to_the_face_box() {
        let bounds = BoundingBox {
            x: 0.2,
            y: 0.2,
            width: 0.4,
            height: 0.5,
        };
        let landmarks = vec![
            region_with_points(LandmarkKind::LeftEye, &[(0.3, 0.55)]),
            region_with_points(LandmarkKind::RightEye, &[(0.5, 0.55)]),
            region_with_points(LandmarkKind::OuterLips, &[(0.34, 0.35), (0.46, 0.35)]),
        ];

        let result = FaceGeometry::calculate_measurements(bounds, &landmarks);
        assert!((result.face_aspect_ratio - 1.25).abs() < 1e-12);
        assert!((result.inter_eye_distance_face_width.unwrap() - 0.5).abs() < 1e-12);
        assert!((result.mouth_width_face_width.unwrap() - 0.3).abs() < 1e-12);
        assert!((result.eye_to_mouth_distance_face_height.unwrap() - 0.4).abs() < 1e-12);
    }

    fn region_with_points(kind: LandmarkKind, points: &[(f64, f64)]) -> LandmarkRegion {
        LandmarkRegion {
            kind,
            points: points.iter().map(|&(x, y)| Point { x, y }).collect(),
        }
    }
}
