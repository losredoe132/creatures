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
    pub relative_position: Vec2,
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
        let sensed_objects: Vec<PerceivedObject> = world
            .plants
            .iter()
            .filter(|plant| {
                let offset = plant.position - origin;
                within_perceptive_field(offset, forward, self.range)
            })
            .map(|plant| {
                let relative_position = plant.position - origin;

                PerceivedObject {
                    kind: PerceivedKind::Plant,
                    relative_position,
                    distance: relative_position.length(),
                    radius: plant.radius,
                    energy: plant.energy,
                }
            })
            .chain(
                world
                    .animals
                    .iter()
                    .filter(|animal| {
                        let offset = animal.position - origin;
                        within_perceptive_field(offset, forward, self.range)
                    })
                    .map(|animal| {
                        let relative_position = animal.position - origin;

                        PerceivedObject {
                            kind: PerceivedKind::Animal,
                            relative_position,
                            distance: relative_position.length(),
                            radius: animal.radius,
                            energy: animal.energy,
                        }
                    }),
            )
            .collect();
        debug!("Sensed objects: {:?}", sensed_objects);
        sensed_objects
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
