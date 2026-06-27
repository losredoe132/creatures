use bevy::prelude::*;

use crate::mlp::{Genome, MLP_INPUTS, mlp_movement};
use crate::sense::{PerceivedAnimal, PerceivedPlant, Sense, Vision};

pub fn think_with_vision(
    vision: &Vision,
    genome: &Genome,
    origin: Vec2,
    forward: Vec2,
    world: &crate::sense::PerceptionWorld<'_>,
) -> Vec2 {
    let sensed = vision.sense(origin, forward, world);
    let features =
        encode_perception_features(&sensed.plants, &sensed.animals, vision.range.max(1.0));
    mlp_movement(features, genome).vector
}

fn encode_perception_features(
    perceived_plants: &[PerceivedPlant],
    perceived_animals: &[PerceivedAnimal],
    vision_range: f32,
) -> [f32; MLP_INPUTS] {
    let plant_features = encode_plant_features(perceived_plants, vision_range);
    let animal_features = encode_animal_features(perceived_animals, vision_range);

    debug!("Plant features: {:?}", plant_features);
    let mut features = [0.0f32; MLP_INPUTS];
    features[0] = plant_features[0];
    features[1] = plant_features[1];
    features[2] = plant_features[2];
    features[3] = plant_features[3];
    features[4] = animal_features[0];
    features[5] = animal_features[1];
    features[6] = animal_features[2];
    features[7] = animal_features[3];
    features[8] = animal_features[4];
    features
}

fn encode_plant_features(perceived_plants: &[PerceivedPlant], vision_range: f32) -> [f32; 4] {
    let nearest_plant = perceived_plants.iter().min_by(|left, right| {
        left.distance
            .partial_cmp(&right.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    debug!("Nearest plant: {:?}", nearest_plant);

    nearest_plant
        .map(|plant| {
            let normalized_relative =
                (plant.relative_position / vision_range).clamp(Vec2::splat(-1.0), Vec2::splat(1.0));
            [
                normalized_relative.x,
                normalized_relative.y,
                (plant.distance / vision_range).clamp(0.0, 1.0),
                (plant.energy / 100.0).clamp(0.0, 1.0),
            ]
        })
        .unwrap_or([0.0; 4])
}

fn encode_animal_features(perceived_animals: &[PerceivedAnimal], vision_range: f32) -> [f32; 5] {
    let nearest_animal = perceived_animals.iter().min_by(|left, right| {
        left.distance
            .partial_cmp(&right.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    debug!("Nearest animal: {:?}", nearest_animal);

    nearest_animal
        .map(|animal| {
            let normalized_relative = (animal.relative_position / vision_range)
                .clamp(Vec2::splat(-1.0), Vec2::splat(1.0));
            [
                normalized_relative.x,
                normalized_relative.y,
                (animal.distance / vision_range).clamp(0.0, 1.0),
                (animal.energy / 100.0).clamp(0.0, 1.0),
                animal.diet as u8 as f32, // Encode diet as a float (0.0, 1.0, 2.0)
            ]
        })
        .unwrap_or([0.0; 5])
}
