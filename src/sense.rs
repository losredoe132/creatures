use crate::creature::Diet;
use bevy::prelude::*;

pub trait Sense {
    type Output;

    fn sense(&self, origin: Vec2, forward: Vec2, world: &PerceptionWorld<'_>) -> Self::Output;
}

#[derive(Debug, Clone, Copy)]
pub struct Vision {
    pub range: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct PerceivedAnimal {
    pub diet: Diet,
    pub relative_position: Vec2,
    pub velocity: Vec2,
    pub distance: f32,
    pub energy: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct PerceivedPlant {
    pub relative_position: Vec2,
    pub distance: f32,
    pub energy: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct PlantSnapshot {
    pub position: Vec2,
    pub energy: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct AnimalSnapshot {
    pub diet: Diet,
    pub position: Vec2,
    pub velocity: Vec2,
    pub energy: f32,
}

pub struct PerceptionWorld<'a> {
    pub plants: &'a [PlantSnapshot],
    pub animals: &'a [AnimalSnapshot],
}

#[derive(Debug, Clone)]
pub struct PerceivedVision {
    pub plants: Vec<PerceivedPlant>,
    pub animals: Vec<PerceivedAnimal>,
}

impl Sense for Vision {
    type Output = PerceivedVision;

    fn sense(&self, origin: Vec2, forward: Vec2, world: &PerceptionWorld<'_>) -> Self::Output {
        let plants: Vec<PerceivedPlant> = world
            .plants
            .iter()
            .filter(|plant| {
                let offset = plant.position - origin;
                within_perceptive_field(offset, forward, self.range)
            })
            .map(|plant| {
                let relative_position = plant.position - origin;

                PerceivedPlant {
                    relative_position,
                    distance: relative_position.length(),
                    energy: plant.energy,
                }
            })
            .collect();

        let animals: Vec<PerceivedAnimal> = world
            .animals
            .iter()
            .filter(|animal| {
                let offset = animal.position - origin;
                within_perceptive_field(offset, forward, self.range)
            })
            .map(|animal| {
                let relative_position = animal.position - origin;

                PerceivedAnimal {
                    diet: animal.diet,
                    relative_position,
                    velocity: animal.velocity,
                    distance: relative_position.length(),
                    energy: animal.energy,
                }
            })
            .collect();

        
        PerceivedVision { plants, animals }
    }
}

fn within_perceptive_field(offset: Vec2, _forward: Vec2, range: f32) -> bool {
    let distance = offset.length();
    if distance == 0.0 || distance > range {
        return false;
    } else {
        return true;
    }
}
