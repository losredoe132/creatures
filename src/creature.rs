use bevy::prelude::*;
use rand::Rng;

use crate::config::SimulationConfig;
use crate::mlp::Genome;
use crate::sense::Vision;
use crate::utils::size_from_energy;

pub trait EnergyPosition {
    fn position(&self) -> Vec2;
    fn set_position(&mut self, position: Vec2);
    fn energy(&self) -> f32;
    fn set_energy(&mut self, energy: f32);
}

pub trait Movable: EnergyPosition {
    fn velocity(&self) -> Vec2;
    fn set_velocity(&mut self, velocity: Vec2);
    fn apply_acceleration(&mut self, acceleration: Vec2, delta_secs: f32) {
        let new_velocity = self.velocity() + acceleration * delta_secs;
        self.set_velocity(new_velocity);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diet {
    Herbivore,
    Omnivore,
    Carnivore,
}

impl Diet {
    pub fn random(rng: &mut impl Rng) -> Self {
        match rng.gen_range(0..3) {
            0 => Self::Herbivore,
            1 => Self::Omnivore,
            _ => Self::Carnivore,
        }
    }

    pub fn can_eat_plants(self) -> bool {
        matches!(self, Self::Herbivore | Self::Omnivore)
    }

    pub fn can_eat_animals(self) -> bool {
        matches!(self, Self::Omnivore | Self::Carnivore)
    }

    pub fn metabolism_ratio(self, config: &SimulationConfig) -> f32 {
        match self {
            Self::Herbivore => config.tuning.herbivore_metabolism_ratio.max(0.0),
            Self::Omnivore => 1.0,
            Self::Carnivore => config.tuning.carnivore_metabolism_ratio.max(0.0),
        }
    }

    pub fn color(self) -> Color {
        match self {
            Self::Herbivore => Color::srgb(0.0, 0.8, 0.0),
            Self::Omnivore => Color::srgb(0.5, 0.5, 0.0),
            Self::Carnivore => Color::srgb(1.0, 0.0, 0.0),
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
    pub diet: Diet,
    pub position: Vec2,
    pub velocity: Vec2,
    pub energy: f32,
    pub size: f32,
    pub color: Color,
    pub vision: Vision,
    pub genome: Genome,
    pub spawn_at: f32,
    pub despawn_at: Option<f32>,
}

impl Animal {
    pub fn new(
        diet: Diet,
        position: Vec2,
        velocity: Vec2,
        genome: Genome,
        time: &Res<Time>,

        config: &SimulationConfig,
    ) -> Self {
        Self {
            diet,
            position,
            velocity,
            energy: config.spawn_config.animal_spawn_energy,
            size: size_from_energy(config.spawn_config.animal_spawn_energy, &config),
            color: diet.color(),
            vision: Vision {
                range: config.tuning.vision_range.max(0.0),
                field_of_view_radians: config
                    .tuning
                    .vision_fov_radians
                    .clamp(0.0, std::f32::consts::PI * 2.0),
            },
            genome: genome,
            spawn_at: time.elapsed_secs(),
            despawn_at: None,
        }
    }
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
