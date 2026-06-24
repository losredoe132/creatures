use bevy::prelude::*;

use crate::brain::Brain;
use crate::sense::Vision;

pub trait EnergyPosition {
    fn position(&self) -> Vec2;
    fn set_position(&mut self, position: Vec2);
    fn energy(&self) -> f32;
    fn set_energy(&mut self, energy: f32);
}

pub trait Movable: EnergyPosition {
    fn velocity(&self) -> Vec2;
    fn set_velocity(&mut self, velocity: Vec2);
    fn apply_impulse(&mut self, impulse: Vec2) {
        let new_velocity = self.velocity() + impulse;
        self.set_velocity(new_velocity);
    }
}

#[derive(Component)]
pub struct Plant {
    pub position: Vec2,
    pub energy: f32,
    pub size: f32,
    pub color: Color,
}

impl EnergyPosition for Plant {
    fn position(&self) -> Vec2 {
        self.position
    }

    fn set_position(&mut self, position: Vec2) {
        self.position = position;
    }

    fn energy(&self) -> f32 {
        self.energy
    }

    fn set_energy(&mut self, energy: f32) {
        self.energy = energy;
    }
}

#[derive(Component)]
pub struct Animal {
    pub position: Vec2,
    pub velocity: Vec2,
    pub energy: f32,
    pub size: f32,
    pub color: Color,
    pub vision: Vision,
    pub brain: Brain,
}

impl EnergyPosition for Animal {
    fn position(&self) -> Vec2 {
        self.position
    }

    fn set_position(&mut self, position: Vec2) {
        self.position = position;
    }

    fn energy(&self) -> f32 {
        self.energy
    }

    fn set_energy(&mut self, energy: f32) {
        self.energy = energy;
    }
}

impl Movable for Animal {
    fn velocity(&self) -> Vec2 {
        self.velocity
    }

    fn set_velocity(&mut self, velocity: Vec2) {
        self.velocity = velocity;
    }
}
