use bevy::prelude::Resource;
use std::env;

#[derive(Resource, Debug, Clone, Copy)]
pub struct WorldBounds {
    pub half_width: f32,
    pub half_height: f32,
}
#[derive(Resource, Debug, Clone, Copy)]

pub struct SpawnConfig {
    pub n_plants: usize,
}

#[derive(Resource, Debug, Clone, Copy)]

pub struct SimulationConfig {
    pub world_bounds: WorldBounds,
    pub spawn_config: SpawnConfig,
}

impl SimulationConfig {
    pub fn from_env() -> Self {
        Self {
            world_bounds: WorldBounds::from_env(),
            spawn_config: SpawnConfig {
                n_plants: read_env_usize("N_PLANTS", 10),
            },
        }
    }
}

impl WorldBounds {
    pub fn from_env() -> Self {
        Self {
            half_width: read_env_f32("WORLD_HALF_WIDTH", 400.0),
            half_height: read_env_f32("WORLD_HALF_HEIGHT", 250.0),
        }
    }
}

fn read_env_f32(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(default)
}

fn read_env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}
