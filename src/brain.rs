use bevy::prelude::*;

use crate::sense::{PerceivedKind, PerceivedObject, Sense, Vision};
use crate::mlp::{mlp_steering, Genome, MLP_INPUTS, SteeringOutput};

const ACCELERATION_SCALE: f32 = 900.0;

pub fn think_with_vision(
    vision: &Vision,
    genome: &Genome,
    origin: Vec2,
    forward: Vec2,
    world: &crate::sense::PerceptionWorld<'_>,
) -> SteeringOutput {
    let sensed = vision.sense(origin, forward, world);
    let features = encode_perception_features(&sensed, vision.range.max(1.0));
    mlp_steering(features, genome)
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

    let nearest_animal = sensed_objects
        .iter()
        .filter(|object| object.kind == PerceivedKind::Animal)
        .min_by(|left, right| {
            left.distance
                .partial_cmp(&right.distance)
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
            let angle_normalized = (plant.angle_radians / std::f32::consts::PI).clamp(-1.0, 1.0);
            [
                angle_normalized.sin(),
                angle_normalized.cos(),
                (plant.distance / vision_range).clamp(0.0, 1.0),
                (plant.energy / 200.0).clamp(0.0, 1.0),
                (plant.radius / 50.0).clamp(0.0, 1.0),
            ]
        })
        .unwrap_or([0.0; 5]);

    let animal_features = nearest_animal
        .map(|animal| {
            let angle_normalized = (animal.angle_radians / std::f32::consts::PI).clamp(-1.0, 1.0);
            [
                angle_normalized.sin(),
                angle_normalized.cos(),
                (animal.distance / vision_range).clamp(0.0, 1.0),
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

pub fn steering_to_acceleration(
    steering: SteeringOutput,
    current_forward: Vec2,
) -> Vec2 {
    if steering.magnitude <= 0.0 {
        return Vec2::ZERO;
    }
    let current_angle = current_forward.to_angle();
    let desired_angle = current_angle + steering.angle_radians;
    Vec2::from_angle(desired_angle) * steering.magnitude * ACCELERATION_SCALE
}
