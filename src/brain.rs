use bevy::prelude::*;

use crate::sense::{PerceivedKind, PerceivedObject, Sense, Vision};

pub trait BrainModel {
    fn impulse(&self, sensed_objects: &[PerceivedObject]) -> Vec2;
}

#[derive(Debug, Clone, Copy)]
pub struct Brain {
    pub steering_strength: f32,
}

impl Default for Brain {
    fn default() -> Self {
        Self {
            steering_strength: 40.0,
        }
    }
}

impl BrainModel for Brain {
    fn impulse(&self, sensed_objects: &[PerceivedObject]) -> Vec2 {
        let nearest_plant = sensed_objects
            .iter()
            .filter(|object| object.kind == PerceivedKind::Plant)
            .min_by(|left, right| {
                left.position
                    .length_squared()
                    .partial_cmp(&right.position.length_squared())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        nearest_plant
            .map(|plant| plant.position.normalize_or_zero() * self.steering_strength)
            .unwrap_or(Vec2::ZERO)
    }
}

pub fn think_with_vision(
    vision: &Vision,
    brain: &Brain,
    origin: Vec2,
    forward: Vec2,
    world: &crate::sense::PerceptionWorld<'_>,
) -> Vec2 {
    let sensed = vision.sense(origin, forward, world);
    brain.impulse(&sensed)
}
