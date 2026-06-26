use bevy::prelude::*;

use crate::mlp::{Genome, MLP_INPUTS, mlp_movement};
use crate::sense::{PerceivedKind, PerceivedObject, Sense, Vision};

pub fn think_with_vision(
    vision: &Vision,
    genome: &Genome,
    origin: Vec2,
    forward: Vec2,
    world: &crate::sense::PerceptionWorld<'_>,
) -> Vec2 {
    let sensed = vision.sense(origin, forward, world);
    let features = encode_perception_features(&sensed, vision.range.max(1.0));
    mlp_movement(features, genome).vector
}

fn encode_perception_features(
    sensed_objects: &[PerceivedObject],
    vision_range: f32,
) -> [f32; MLP_INPUTS] {
    let nearest_plant = sensed_objects
        .iter()
        .filter(|object| object.kind == PerceivedKind::Plant)
        .min_by(|left, right| {
            left.distance
                .partial_cmp(&right.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

    debug!("Nearest plant: {:?}", nearest_plant);

    let plant_features = nearest_plant
        .map(|plant| {
            let angle_normalized = (plant.angle_radians / std::f32::consts::PI).clamp(-1.0, 1.0);
            [
                angle_normalized.sin(),
                angle_normalized.cos(),
                (plant.distance / vision_range).clamp(0.0, 1.0),
            ]
        })
        .unwrap_or([0.0; 3]);

    debug!("Plant features: {:?}", plant_features);
    let mut features = [0.0f32; MLP_INPUTS];
    features[0] = plant_features[0];
    features[1] = plant_features[1];
    features[2] = plant_features[2];
    features
}
