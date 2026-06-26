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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerceivedKind {
    Plant,
    Animal,
}

#[derive(Debug, Clone, Copy)]
pub struct PerceivedObject {
    pub kind: PerceivedKind,
    pub angle_radians: f32,
    pub distance: f32,
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
        world
            .plants
            .iter()
            .filter(|plant| {
                let offset = plant.position - origin;
                within_perceptive_field(offset, forward, self.range)
            })
            .map(|plant| PerceivedObject {
                kind: PerceivedKind::Plant,
                angle_radians: forward.angle_to((plant.position - origin).normalize_or_zero()),
                distance: (plant.position - origin).length(),
                radius: plant.radius,
                energy: plant.energy,
            })
            // .chain(world.animals.iter().filter_map(|animal| {
            //     let offset = animal.position - origin;
            //     within_vision_cone(offset, forward, self.range, half_fov).then_some(PerceivedObject {
            //         kind: PerceivedKind::Animal,
            //         angle_radians: forward.angle_to(offset.normalize_or_zero()),
            //         distance: offset.length(),
            //         radius: animal.radius,
            //         energy: animal.energy,
            //     })
            // }))
            .collect()
    }
}

fn within_perceptive_field(offset: Vec2, forward: Vec2, range: f32) -> bool {
    let distance = offset.length();
    if distance == 0.0 || distance > range {
        return false;
    } else {
        return true;
    }
}
