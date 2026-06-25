use bevy::prelude::*;

use crate::sense::{PerceivedKind, PerceivedObject, Sense, Vision};
use crate::mlp::{mlp_acceleration, Genome, MLP_INPUTS};

const ACCELERATION_SCALE: f32 = 450.0;

pub fn think_with_vision(
    vision: &Vision,
    genome: &Genome,
    origin: Vec2,
    forward: Vec2,
    world: &crate::sense::PerceptionWorld<'_>,
) -> Vec2 {
    let sensed = vision.sense(origin, forward, world);
    let sensed_offsets: Vec<PerceivedObject> = sensed
        .into_iter()
        .map(|mut object| {
            object.position -= origin;
            object
        })
        .collect();
    let features = encode_perception_features(&sensed_offsets, vision.range.max(1.0));
    mlp_acceleration(features, genome) * ACCELERATION_SCALE
}

fn encode_perception_features(
    sensed_objects: &[PerceivedObject],
    vision_range: f32,
) -> [f32; MLP_INPUTS] {
    let nearest_plant = sensed_objects
        .iter()
        .filter(|object| object.kind == PerceivedKind::Plant)
        .min_by(|left, right| {
            left.position
                .length_squared()
                .partial_cmp(&right.position.length_squared())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

    let nearest_animal = sensed_objects
        .iter()
        .filter(|object| object.kind == PerceivedKind::Animal)
        .min_by(|left, right| {
            left.position
                .length_squared()
                .partial_cmp(&right.position.length_squared())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

    let plant_count = sensed_objects
        .iter()
        .filter(|object| object.kind == PerceivedKind::Plant)
        .count() as f32;
    let animal_count = sensed_objects
        .iter()
        .filter(|object| object.kind == PerceivedKind::Animal)
        .count() as f32;

    let plant_features = nearest_plant
        .map(|plant| {
            let distance = plant.position.length();
            let dir = plant.position.normalize_or_zero();
            [
                dir.x,
                dir.y,
                (distance / vision_range).clamp(0.0, 1.0),
                (plant.energy / 200.0).clamp(0.0, 1.0),
                (plant.radius / 50.0).clamp(0.0, 1.0),
            ]
        })
        .unwrap_or([0.0; 5]);

    let animal_features = nearest_animal
        .map(|animal| {
            let distance = animal.position.length();
            let dir = animal.position.normalize_or_zero();
            [
                dir.x,
                dir.y,
                (distance / vision_range).clamp(0.0, 1.0),
                (animal.energy / 200.0).clamp(0.0, 1.0),
                (animal.radius / 50.0).clamp(0.0, 1.0),
            ]
        })
        .unwrap_or([0.0; 5]);

    let mut features = [0.0f32; MLP_INPUTS];
    features[0] = plant_features[0];
    features[1] = plant_features[1];
    features[2] = plant_features[2];
    features[3] = plant_features[3];
    features[4] = plant_features[4];
    features[5] = animal_features[0];
    features[6] = animal_features[1];
    features[7] = (plant_count / 8.0).clamp(0.0, 1.0);
    features[8] = (animal_count / 8.0).clamp(0.0, 1.0);
    features[9] = 1.0 - animal_features[2];
    features
}
