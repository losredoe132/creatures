use bevy::prelude::*;

use crate::creature::Diet;
use crate::mlp::{Genome, MLP_INPUTS, mlp_movement};
use crate::sense::{PerceivedAnimal, PerceivedCarcass, PerceivedPlant, Sense, Vision};

pub fn compute_features(
    vision_range: f32,
    origin: Vec2,
    forward: Vec2,
    self_energy: f32,
    world: &crate::sense::PerceptionWorld<'_>,
) -> [f32; MLP_INPUTS] {
    let vision = Vision {
        range: vision_range,
    };
    let sensed = vision.sense(origin, forward, world);
    encode_perception_features(
        &sensed.plants,
        &sensed.animals,
        &sensed.carcasses,
        vision_range.max(1.0),
        forward,
        self_energy,
    )
}

pub fn think_with_vision(
    vision: &Vision,
    genome: &Genome,
    origin: Vec2,
    forward: Vec2,
    self_energy: f32,
    world: &crate::sense::PerceptionWorld<'_>,
) -> Vec2 {
    let sensed = vision.sense(origin, forward, world);
    let features = encode_perception_features(
        &sensed.plants,
        &sensed.animals,
        &sensed.carcasses,
        vision.range.max(1.0),
        forward,
        self_energy,
    );

    mlp_movement(features, genome).vector
}

fn encode_perception_features(
    perceived_plants: &[PerceivedPlant],
    perceived_animals: &[PerceivedAnimal],
    perceived_carcasses: &[PerceivedCarcass],
    vision_range: f32,
    self_velocity: Vec2,
    self_energy: f32,
) -> [f32; MLP_INPUTS] {
    let plant_features = encode_plant_features(perceived_plants, vision_range);

    let perceived_animals_herbivors: Vec<PerceivedAnimal> = perceived_animals
        .iter()
        .filter(|animal| animal.diet == Diet::Herbivore)
        .cloned()
        .collect();
    let perceived_animals_carnivors: Vec<PerceivedAnimal> = perceived_animals
        .iter()
        .filter(|animal| animal.diet == Diet::Carnivore)
        .cloned()
        .collect();

    let animal_features_herbivors =
        encode_animal_features(&perceived_animals_herbivors, vision_range);
    let animal_features_carnivors =
        encode_animal_features(&perceived_animals_carnivors, vision_range);

    let animal_features = encode_animal_features(perceived_animals, vision_range);

    let mut features = [0.0f32; MLP_INPUTS];
    features[0] = plant_features[0];
    features[1] = plant_features[1];
    features[2] = plant_features[3];
    features[3] = animal_features_herbivors[0];
    features[4] = animal_features_herbivors[1];
    features[5] = animal_features_herbivors[4];
    features[6] = animal_features_carnivors[0];
    features[7] = animal_features_carnivors[1];
    features[8] = animal_features_carnivors[4];
    //features[5] = animal_features_carnivors[2];animanimal_featuresal_features
    //features[6] = animal_features_carnivors[3];
    // features[8] = animal_features_carnivors[4];
    // features[9] = animal_features_herbivores[0];
    // features[10] = animal_features_herbivores[1];
    // features[11] = animal_features_herbivores[2];
    // features[12] = animal_features_herbivores[3];
    // features[13] = animal_features_herbivores[4];
    // features[14] = animal_features_omnivores[0];
    // features[15] = animal_features_omnivores[1];
    // features[16] = animal_features_omnivores[2];
    // features[17] = animal_features_omnivores[3];
    // features[18] = animal_features_omnivores[4];
    // features[19] = self_awareness_features[0];
    // features[20] = self_awareness_features[1];
    // features[21] = self_awareness_features[2];
    // features[22] = carcass_features[0];
    // features[23] = carcass_features[1];
    // features[24] = carcass_features[2];
    // features[25] = carcass_features[3];
    features
}

fn encode_self_awareness_features(self_velocity: Vec2, self_energy: f32) -> [f32; 3] {
    [
        self_velocity.x.tanh().clamp(-1.0, 1.0),
        self_velocity.y.tanh().clamp(-1.0, 1.0),
        (self_energy / 100.0).clamp(0.0, 1.0),
    ]
}

fn encode_plant_features(perceived_plants: &[PerceivedPlant], vision_range: f32) -> [f32; 4] {
    let nearest_plant = perceived_plants.iter().min_by(|left, right| {
        left.distance
            .partial_cmp(&right.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    nearest_plant
        .map(|plant| {
            let normalized_relative = (plant.relative_position / plant.distance)
                .clamp(Vec2::splat(-1.0), Vec2::splat(1.0));
            [
                normalized_relative.x,
                normalized_relative.y,
                (plant.distance / vision_range).clamp(0.0, 1.0),
                (plant.energy / 100.0).clamp(0.0, 1.0),
            ]
        })
        .unwrap_or([0.0; 4])
}

fn encode_carcass_features(
    perceived_carcasses: &[PerceivedCarcass],
    vision_range: f32,
) -> [f32; 4] {
    let nearest = perceived_carcasses.iter().min_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    nearest
        .map(|c| {
            let normalized =
                (c.relative_position / vision_range).clamp(Vec2::splat(-1.0), Vec2::splat(1.0));
            [
                normalized.x,
                normalized.y,
                (c.distance / vision_range).clamp(0.0, 1.0),
                (c.energy / 100.0).clamp(0.0, 1.0),
            ]
        })
        .unwrap_or([0.0; 4])
}

fn encode_animal_features(perceived_animals: &[PerceivedAnimal], vision_range: f32) -> [f32; 7] {
    let nearest_animal = perceived_animals.iter().min_by(|left, right| {
        left.distance
            .partial_cmp(&right.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    nearest_animal
        .map(|animal| {
            let normalized_relative = (animal.relative_position / animal.distance)
                .clamp(Vec2::splat(-1.0), Vec2::splat(1.0));
            [
                normalized_relative.x,
                normalized_relative.y,
                animal.velocity[0].tanh().clamp(-1.0, 1.0),
                animal.velocity[1].tanh().clamp(-1.0, 1.0),
                (animal.distance / vision_range).clamp(0.0, 1.0),
                (animal.energy / 100.0).clamp(0.0, 1.0),
                animal.diet as u8 as f32, // Encode diet as a float (0.0, 1.0, 2.0)
            ]
        })
        .unwrap_or([0.0; 7])
}
