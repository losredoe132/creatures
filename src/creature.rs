use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::config::SimulationConfig;
use crate::mlp::Genome;
use crate::sense::Vision;
use crate::simulation::GlobalFrameCounter;
use crate::utils::size_from_energy;

pub trait EnergyPosition {
    fn set_position(&mut self, position: Vec2);
    fn energy(&self) -> f32;
    fn set_energy(&mut self, energy: f32);
}

pub trait Movable: EnergyPosition {
    fn velocity(&self) -> Vec2;
    fn set_velocity(&mut self, velocity: Vec2);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Diet {
    Herbivore,
    Omnivore,
    Carnivore,
    Scavenger,
}

impl Diet {
    pub fn random(rng: &mut impl Rng) -> Self {
        match rng.gen_range(0..4) {
            0 => Self::Herbivore,
            1 => Self::Omnivore,
            2 => Self::Carnivore,
            _ => Self::Scavenger,
        }
    }

    pub fn can_eat_plants(self) -> bool {
        matches!(self, Self::Herbivore | Self::Omnivore)
    }

    pub fn can_eat_animals(self) -> bool {
        matches!(self, Self::Omnivore | Self::Carnivore)
    }

    pub fn can_eat_carcasses(self) -> bool {
        matches!(self, Self::Scavenger | Self::Omnivore)
    }

    pub fn metabolism_ratio(self, config: &SimulationConfig) -> f32 {
        match self {
            Self::Herbivore => config.tuning.herbivore_metabolism_ratio.max(0.0),
            Self::Omnivore => config.tuning.omnivore_metabolism_ratio.max(0.0),
            Self::Carnivore => config.tuning.carnivore_metabolism_ratio.max(0.0),
            Self::Scavenger => config.tuning.scavenger_metabolism_ratio.max(0.0),
        }
    }

    pub fn color(self) -> Color {
        match self {
            Self::Herbivore => Color::srgb(0.0, 0.8, 0.0),
            Self::Omnivore => Color::srgb(0.5, 0.5, 0.0),
            Self::Carnivore => Color::srgb(1.0, 0.0, 0.0),
            Self::Scavenger => Color::srgb(0.6, 0.4, 0.1),
        }
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

#[derive(Component, Debug)]
pub struct Animal {
    pub id: u64,
    pub parent_id: Option<u64>,
    pub diet: Diet,
    pub position: Vec2,
    pub velocity: Vec2,
    pub energy: f32,
    pub initial_energy: f32,
    pub size: f32,
    pub color: Color,
    pub vision: Vision,
    pub genome: Genome,
    pub spawn_at: u64,
    pub despawn_at: Option<u64>,
    pub family: u32,
}

#[derive(Component)]
pub struct Carcass {
    pub position: Vec2,
    pub energy: f32,
    pub size: f32,
}

impl Animal {
    pub fn new(
        id: u64,
        parent_id: Option<u64>,
        diet: Diet,
        position: Vec2,
        velocity: Vec2,
        genome: Genome,
        frame_count: &Res<GlobalFrameCounter>,
        config: &SimulationConfig,
        family: u32,
    ) -> Self {
        Self {
            id,
            parent_id,
            diet,
            position,
            velocity,
            energy: config.spawn_config.animal_spawn_energy,
            initial_energy: config.spawn_config.animal_spawn_energy,
            size: size_from_energy(config.spawn_config.animal_spawn_energy, &config),
            color: diet.color(),
            vision: Vision {
                range: config.tuning.vision_range.max(0.0),
            },
            genome: genome,
            spawn_at: frame_count.0 as u64,
            despawn_at: None,
            family: family,
        }
    }
}

impl EnergyPosition for Animal {
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
