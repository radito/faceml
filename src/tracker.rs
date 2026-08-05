use crate::{BoundingBox, FaceGeometry, LandmarkKind, LandmarkRegion, Point};
use std::time::Instant;

const INVALID_MATCH_COST: f64 = 1_000_000.0;
const UNMATCHED_COST: f64 = 0.9;

#[derive(Clone, Copy, Debug)]
pub struct FaceTrackerConfig {
    /// Detection updates that a missing face remains eligible for reassociation.
    pub max_missed_frames: u32,
    /// Weight of the latest observation when smoothing geometry, in `(0, 1]`.
    pub smoothing_factor: f64,
}

impl Default for FaceTrackerConfig {
    fn default() -> Self {
        Self {
            max_missed_frames: 15,
            smoothing_factor: 0.8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrackedFace {
    pub track_id: u64,
    pub geometry: FaceGeometry,
    velocity_per_second: Point,
    smoothing_lag_seconds: f64,
}

impl TrackedFace {
    /// Predicts geometry forward to compensate for inference and presentation delay.
    pub fn predicted(&self, pipeline_delay_seconds: f64) -> Self {
        let prediction_seconds =
            (pipeline_delay_seconds.max(0.0) + self.smoothing_lag_seconds).min(0.25);
        let mut predicted = self.clone();
        translate_geometry(
            &mut predicted.geometry,
            Point {
                x: self.velocity_per_second.x * prediction_seconds,
                y: self.velocity_per_second.y * prediction_seconds,
            },
        );
        predicted
    }
}

#[derive(Debug)]
pub struct FaceTracker {
    config: FaceTrackerConfig,
    next_id: u64,
    tracks: Vec<Track>,
}

#[derive(Debug)]
struct Track {
    id: u64,
    geometry: FaceGeometry,
    last_observed_bounds: BoundingBox,
    last_observed_at: Instant,
    last_interval_seconds: f64,
    velocity: Point,
    missed_frames: u32,
}

impl Default for FaceTracker {
    fn default() -> Self {
        Self::new(FaceTrackerConfig::default())
    }
}

impl FaceTracker {
    pub fn new(config: FaceTrackerConfig) -> Self {
        assert!(
            config.smoothing_factor > 0.0 && config.smoothing_factor <= 1.0,
            "smoothing_factor must be in (0, 1]"
        );
        Self {
            config,
            next_id: 1,
            tracks: Vec::new(),
        }
    }

    /// Associates detections with existing tracks and returns faces in detection order.
    pub fn update(&mut self, detections: Vec<FaceGeometry>) -> Vec<TrackedFace> {
        self.update_at(detections, Instant::now())
    }

    /// Time-aware update used by live capture pipelines.
    pub fn update_at(
        &mut self,
        detections: Vec<FaceGeometry>,
        observed_at: Instant,
    ) -> Vec<TrackedFace> {
        let assignments = associate(&self.tracks, &detections, observed_at);
        let mut matched_tracks = vec![false; self.tracks.len()];
        let mut tracked_detections = vec![None; detections.len()];

        for (track_index, detection_index) in assignments {
            let track = &mut self.tracks[track_index];
            let observation = &detections[detection_index];
            let previous_center = center(track.last_observed_bounds);
            let observed_center = center(observation.bounding_box);
            let interval_seconds = observed_at
                .saturating_duration_since(track.last_observed_at)
                .as_secs_f64()
                .max(1.0 / 120.0);
            let observed_velocity = Point {
                x: (observed_center.x - previous_center.x) / interval_seconds,
                y: (observed_center.y - previous_center.y) / interval_seconds,
            };
            track.velocity = Point {
                x: lerp(track.velocity.x, observed_velocity.x, 0.65),
                y: lerp(track.velocity.y, observed_velocity.y, 0.65),
            };
            track.geometry =
                smooth_geometry(&track.geometry, observation, self.config.smoothing_factor);
            track.last_observed_bounds = observation.bounding_box;
            track.last_observed_at = observed_at;
            track.last_interval_seconds = interval_seconds;
            track.missed_frames = 0;
            matched_tracks[track_index] = true;
            tracked_detections[detection_index] = Some(TrackedFace {
                track_id: track.id,
                geometry: track.geometry.clone(),
                velocity_per_second: track.velocity,
                smoothing_lag_seconds: smoothing_lag_seconds(
                    self.config.smoothing_factor,
                    track.last_interval_seconds,
                ),
            });
        }

        for (index, track) in self.tracks.iter_mut().enumerate() {
            if !matched_tracks[index] {
                track.missed_frames = track.missed_frames.saturating_add(1);
            }
        }

        for (index, observation) in detections.into_iter().enumerate() {
            if tracked_detections[index].is_some() {
                continue;
            }
            let id = self.next_id;
            self.next_id = self.next_id.saturating_add(1);
            self.tracks.push(Track {
                id,
                geometry: observation.clone(),
                last_observed_bounds: observation.bounding_box,
                last_observed_at: observed_at,
                last_interval_seconds: 1.0 / 30.0,
                velocity: Point { x: 0.0, y: 0.0 },
                missed_frames: 0,
            });
            tracked_detections[index] = Some(TrackedFace {
                track_id: id,
                geometry: observation,
                velocity_per_second: Point { x: 0.0, y: 0.0 },
                smoothing_lag_seconds: 0.0,
            });
        }

        self.tracks
            .retain(|track| track.missed_frames <= self.config.max_missed_frames);

        tracked_detections.into_iter().flatten().collect()
    }
}

fn associate(
    tracks: &[Track],
    detections: &[FaceGeometry],
    observed_at: Instant,
) -> Vec<(usize, usize)> {
    if tracks.is_empty() || detections.is_empty() {
        return Vec::new();
    }

    // Extra rows and columns let the optimizer leave either a track or a detection unmatched.
    let track_count = tracks.len();
    let detection_count = detections.len();
    let size = track_count + detection_count;
    let mut costs = vec![vec![INVALID_MATCH_COST; size]; size];

    for (track_index, track) in tracks.iter().enumerate() {
        for (detection_index, detection) in detections.iter().enumerate() {
            costs[track_index][detection_index] = match_cost(track, detection, observed_at);
        }
        costs[track_index][detection_count + track_index] = UNMATCHED_COST;
    }
    for detection_index in 0..detection_count {
        let dummy_row = track_count + detection_index;
        costs[dummy_row][detection_index] = UNMATCHED_COST;
        for dummy_column in detection_count..size {
            costs[dummy_row][dummy_column] = 0.0;
        }
    }

    hungarian_minimize(&costs)
        .into_iter()
        .enumerate()
        .filter_map(|(track_index, column)| {
            (track_index < track_count
                && column < detection_count
                && costs[track_index][column] < INVALID_MATCH_COST / 2.0)
                .then_some((track_index, column))
        })
        .collect()
}

fn match_cost(track: &Track, detection: &FaceGeometry, observed_at: Instant) -> f64 {
    let predicted = predicted_bounds(track, observed_at);
    let observed = detection.bounding_box;
    let overlap = intersection_over_union(predicted, observed);
    let average_diagonal = ((diagonal(predicted) + diagonal(observed)) / 2.0).max(0.05);
    let center_distance = center(predicted).distance(center(observed));
    let relative_center_distance = center_distance / average_diagonal;

    if overlap < 0.05 && relative_center_distance > 1.25 {
        return INVALID_MATCH_COST;
    }

    let landmarks = landmark_shape_distance(&track.geometry, detection).unwrap_or(0.5);
    0.6 * (1.0 - overlap) + 0.3 * relative_center_distance.min(1.0) + 0.1 * landmarks.min(1.0)
}

fn predicted_bounds(track: &Track, observed_at: Instant) -> BoundingBox {
    let elapsed = observed_at
        .saturating_duration_since(track.last_observed_at)
        .as_secs_f64()
        .min(0.5);
    BoundingBox {
        x: track.last_observed_bounds.x + track.velocity.x * elapsed,
        y: track.last_observed_bounds.y + track.velocity.y * elapsed,
        ..track.last_observed_bounds
    }
}

fn smoothing_lag_seconds(alpha: f64, interval_seconds: f64) -> f64 {
    ((1.0 - alpha) / alpha) * interval_seconds
}

fn translate_geometry(geometry: &mut FaceGeometry, offset: Point) {
    geometry.bounding_box.x += offset.x;
    geometry.bounding_box.y += offset.y;
    for region in &mut geometry.landmarks {
        for point in &mut region.points {
            point.x += offset.x;
            point.y += offset.y;
        }
    }
}

fn landmark_shape_distance(previous: &FaceGeometry, current: &FaceGeometry) -> Option<f64> {
    let previous_points = all_points(&previous.landmarks)?;
    let current_points = all_points(&current.landmarks)?;
    if previous_points.len() != current_points.len() || previous_points.is_empty() {
        return None;
    }

    let distance = previous_points
        .iter()
        .zip(current_points)
        .map(|(left, right)| {
            normalized_point(*left, previous.bounding_box)
                .distance(normalized_point(*right, current.bounding_box))
        })
        .sum::<f64>()
        / previous_points.len() as f64;
    Some(distance / std::f64::consts::SQRT_2)
}

fn all_points(regions: &[LandmarkRegion]) -> Option<&[Point]> {
    regions
        .iter()
        .find(|region| region.kind == LandmarkKind::AllPoints)
        .map(|region| region.points.as_slice())
}

fn normalized_point(point: Point, bounds: BoundingBox) -> Point {
    Point {
        x: (point.x - bounds.x) / bounds.width.max(f64::EPSILON),
        y: (point.y - bounds.y) / bounds.height.max(f64::EPSILON),
    }
}

fn smooth_geometry(previous: &FaceGeometry, current: &FaceGeometry, alpha: f64) -> FaceGeometry {
    let bounding_box = smooth_bounds(previous.bounding_box, current.bounding_box, alpha);
    let landmarks = current
        .landmarks
        .iter()
        .map(|region| {
            let previous_region = previous
                .landmarks
                .iter()
                .find(|candidate| candidate.kind == region.kind);
            let points = match previous_region {
                Some(previous_region) if previous_region.points.len() == region.points.len() => {
                    previous_region
                        .points
                        .iter()
                        .zip(&region.points)
                        .map(|(old, new)| Point {
                            x: lerp(old.x, new.x, alpha),
                            y: lerp(old.y, new.y, alpha),
                        })
                        .collect()
                }
                _ => region.points.clone(),
            };
            LandmarkRegion {
                kind: region.kind,
                points,
            }
        })
        .collect::<Vec<_>>();

    FaceGeometry {
        confidence: current.confidence,
        landmark_confidence: current.landmark_confidence,
        bounding_box,
        roll_radians: smooth_optional(previous.roll_radians, current.roll_radians, alpha),
        yaw_radians: smooth_optional(previous.yaw_radians, current.yaw_radians, alpha),
        pitch_radians: smooth_optional(previous.pitch_radians, current.pitch_radians, alpha),
        measurements: FaceGeometry::calculate_measurements(bounding_box, &landmarks),
        landmarks,
    }
}

fn smooth_bounds(previous: BoundingBox, current: BoundingBox, alpha: f64) -> BoundingBox {
    BoundingBox {
        x: lerp(previous.x, current.x, alpha),
        y: lerp(previous.y, current.y, alpha),
        width: lerp(previous.width, current.width, alpha),
        height: lerp(previous.height, current.height, alpha),
    }
}

fn smooth_optional(previous: Option<f64>, current: Option<f64>, alpha: f64) -> Option<f64> {
    match (previous, current) {
        (Some(previous), Some(current)) => Some(lerp(previous, current, alpha)),
        (_, Some(current)) => Some(current),
        (Some(previous), None) => Some(previous),
        (None, None) => None,
    }
}

fn lerp(previous: f64, current: f64, alpha: f64) -> f64 {
    previous + (current - previous) * alpha
}

fn center(bounds: BoundingBox) -> Point {
    Point {
        x: bounds.x + bounds.width / 2.0,
        y: bounds.y + bounds.height / 2.0,
    }
}

fn diagonal(bounds: BoundingBox) -> f64 {
    bounds.width.hypot(bounds.height)
}

fn intersection_over_union(left: BoundingBox, right: BoundingBox) -> f64 {
    let intersection_width = (left.x + left.width).min(right.x + right.width) - left.x.max(right.x);
    let intersection_height =
        (left.y + left.height).min(right.y + right.height) - left.y.max(right.y);
    if intersection_width <= 0.0 || intersection_height <= 0.0 {
        return 0.0;
    }
    let intersection = intersection_width * intersection_height;
    let union = left.width * left.height + right.width * right.height - intersection;
    if union <= f64::EPSILON {
        0.0
    } else {
        intersection / union
    }
}

/// Minimum-cost assignment for a square matrix using the Hungarian algorithm.
fn hungarian_minimize(costs: &[Vec<f64>]) -> Vec<usize> {
    let size = costs.len();
    debug_assert!(costs.iter().all(|row| row.len() == size));
    if size == 0 {
        return Vec::new();
    }

    let mut row_potential = vec![0.0; size + 1];
    let mut column_potential = vec![0.0; size + 1];
    let mut row_for_column = vec![0usize; size + 1];
    let mut previous_column = vec![0usize; size + 1];

    for row in 1..=size {
        row_for_column[0] = row;
        let mut column = 0;
        let mut minimum = vec![f64::INFINITY; size + 1];
        let mut used = vec![false; size + 1];

        loop {
            used[column] = true;
            let active_row = row_for_column[column];
            let mut delta = f64::INFINITY;
            let mut next_column = 0;
            for candidate in 1..=size {
                if used[candidate] {
                    continue;
                }
                let reduced = costs[active_row - 1][candidate - 1]
                    - row_potential[active_row]
                    - column_potential[candidate];
                if reduced < minimum[candidate] {
                    minimum[candidate] = reduced;
                    previous_column[candidate] = column;
                }
                if minimum[candidate] < delta {
                    delta = minimum[candidate];
                    next_column = candidate;
                }
            }

            for candidate in 0..=size {
                if used[candidate] {
                    row_potential[row_for_column[candidate]] += delta;
                    column_potential[candidate] -= delta;
                } else {
                    minimum[candidate] -= delta;
                }
            }
            column = next_column;
            if row_for_column[column] == 0 {
                break;
            }
        }

        loop {
            let candidate = previous_column[column];
            row_for_column[column] = row_for_column[candidate];
            column = candidate;
            if column == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![0; size];
    for column in 1..=size {
        assignment[row_for_column[column] - 1] = column - 1;
    }
    assignment
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GeometryMeasurements;

    #[test]
    fn ids_follow_faces_when_detection_order_changes() {
        let mut tracker = FaceTracker::default();
        let first = tracker.update(vec![face(0.10), face(0.70)]);
        assert_eq!(ids(&first), vec![1, 2]);

        let second = tracker.update(vec![face(0.69), face(0.11)]);
        assert_eq!(ids(&second), vec![2, 1]);
    }

    #[test]
    fn a_short_disappearance_keeps_the_id() {
        let mut tracker = FaceTracker::new(FaceTrackerConfig {
            max_missed_frames: 2,
            smoothing_factor: 1.0,
        });
        assert_eq!(ids(&tracker.update(vec![face(0.2)])), vec![1]);
        assert!(tracker.update(Vec::new()).is_empty());
        assert_eq!(ids(&tracker.update(vec![face(0.21)])), vec![1]);
    }

    #[test]
    fn expired_tracks_receive_a_new_id() {
        let mut tracker = FaceTracker::new(FaceTrackerConfig {
            max_missed_frames: 1,
            smoothing_factor: 1.0,
        });
        assert_eq!(ids(&tracker.update(vec![face(0.2)])), vec![1]);
        tracker.update(Vec::new());
        tracker.update(Vec::new());
        assert_eq!(ids(&tracker.update(vec![face(0.2)])), vec![2]);
    }

    #[test]
    fn smoothing_reduces_single_frame_box_jitter() {
        let mut tracker = FaceTracker::new(FaceTrackerConfig {
            max_missed_frames: 1,
            smoothing_factor: 0.5,
        });
        tracker.update(vec![face(0.2)]);
        let tracked = tracker.update(vec![face(0.3)]);
        assert!((tracked[0].geometry.bounding_box.x - 0.25).abs() < 1e-12);
    }

    #[test]
    fn prediction_moves_geometry_forward_by_measured_velocity() {
        let mut tracker = FaceTracker::new(FaceTrackerConfig {
            max_missed_frames: 1,
            smoothing_factor: 1.0,
        });
        let start = Instant::now();
        tracker.update_at(vec![face(0.2)], start);
        let tracked = tracker.update_at(
            vec![face(0.3)],
            start + std::time::Duration::from_millis(100),
        );
        let predicted = tracked[0].predicted(0.1);

        // Velocity is deliberately damped to 65% of the raw 1.0 normalized-unit/second motion.
        assert!((predicted.geometry.bounding_box.x - 0.365).abs() < 1e-12);
    }

    #[test]
    fn hungarian_assignment_finds_the_global_minimum() {
        let assignment = hungarian_minimize(&[
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ]);
        assert_eq!(assignment, vec![1, 0, 2]);
    }

    fn ids(faces: &[TrackedFace]) -> Vec<u64> {
        faces.iter().map(|face| face.track_id).collect()
    }

    fn face(x: f64) -> FaceGeometry {
        let bounding_box = BoundingBox {
            x,
            y: 0.2,
            width: 0.2,
            height: 0.3,
        };
        FaceGeometry {
            confidence: 1.0,
            landmark_confidence: 1.0,
            bounding_box,
            roll_radians: None,
            yaw_radians: None,
            pitch_radians: None,
            measurements: GeometryMeasurements {
                face_aspect_ratio: 1.5,
                inter_eye_distance_face_width: None,
                mouth_width_face_width: None,
                eye_to_mouth_distance_face_height: None,
            },
            landmarks: Vec::new(),
        }
    }
}
