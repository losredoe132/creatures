use bevy::prelude::*;

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
    pub genome: Genome,
    pub spawn_at: f32,
    pub despawn_at: Option<f32>,
}

impl Animal {
    pub fn new(
        position: Vec2,
        velocity: Vec2,
        genome: Genome,
        time: &Res<Time>,

        config: &SimulationConfig,
    ) -> Self {
        Self {
            position,
            velocity,
            energy: config.spawn_config.animal_spawn_energy,
            size: size_from_energy(config.spawn_config.animal_spawn_energy, &config),
            color: Color::srgb(0.8, 0.2, 0.4),
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
