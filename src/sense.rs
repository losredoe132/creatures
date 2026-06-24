use bevy::prelude::*;

pub trait Sense {
    type Output;

    fn sense(&self, origin: Vec2, forward: Vec2, world: &PerceptionWorld<'_>) -> Self::Output;
}

#[derive(Debug, Clone, Copy)]
pub struct Vision {
    pub range: f32,
    pub field_of_view_radians: f32,
}

impl Default for Vision {
    fn default() -> Self {
        Self {
            range: 200.0,
            field_of_view_radians: std::f32::consts::FRAC_PI_2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerceivedKind {
    Plant,
    Animal,
}

#[derive(Debug, Clone, Copy)]
pub struct PerceivedObject {
    pub kind: PerceivedKind,
    pub position: Vec2,
    pub radius: f32,
    pub energy: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct PlantSnapshot {
    pub position: Vec2,
    pub energy: f32,
    pub radius: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct AnimalSnapshot {
    pub position: Vec2,
    pub energy: f32,
    pub radius: f32,
}

pub struct PerceptionWorld<'a> {
    pub plants: &'a [PlantSnapshot],
    pub animals: &'a [AnimalSnapshot],
}

impl Sense for Vision {
    type Output = Vec<PerceivedObject>;

    fn sense(&self, origin: Vec2, forward: Vec2, world: &PerceptionWorld<'_>) -> Self::Output {
        let forward = forward.normalize_or_zero();
        let half_fov = self.field_of_view_radians * 0.5;

        world
            .plants
            .iter()
            .filter_map(|plant| {
                let offset = plant.position - origin;
                within_vision_cone(offset, forward, self.range, half_fov).then_some(PerceivedObject {
                    kind: PerceivedKind::Plant,
                    position: plant.position,
                    radius: plant.radius,
                    energy: plant.energy,
                })
            })
            .chain(world.animals.iter().filter_map(|animal| {
                let offset = animal.position - origin;
                within_vision_cone(offset, forward, self.range, half_fov).then_some(PerceivedObject {
                    kind: PerceivedKind::Animal,
                    position: animal.position,
                    radius: animal.radius,
                    energy: animal.energy,
                })
            }))
            .collect()
    }
}

fn within_vision_cone(offset: Vec2, forward: Vec2, range: f32, half_fov: f32) -> bool {
    let distance = offset.length();
    if distance == 0.0 || distance > range {
        return false;
    }

    if forward == Vec2::ZERO {
        return true;
    }

    let angle = forward.angle_to(offset.normalize());
    angle.abs() <= half_fov
}
