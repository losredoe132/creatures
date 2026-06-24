use bevy::prelude::Resource;
use std::env;

#[derive(Resource, Debug, Clone, Copy)]
pub struct WorldBounds {
    pub half_width: f32,
    pub half_height: f32,
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
