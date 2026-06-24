use bevy::prelude::{IVec2, Resource, Vec2};
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
pub struct GridConfig {
    pub resolution_x: usize,
    pub resolution_y: usize,
}

#[derive(Resource, Debug, Clone, Copy)]
pub struct TuningConfig {
    pub plant_consume_per_cell: f32,
    pub plant_growth_per_sec: f32,
    pub plant_max_energy: f32,
    pub plant_spawn_rate_per_sec: f32,
    pub animal_crowding_penalty: f32,
    pub plant_crowding_penalty: f32,
    pub animal_base_energy_drain_per_sec: f32,
    pub animal_speed_energy_drain_per_sec: f32,
    pub animal_speed_exponent: f32,
    pub animal_speed_reference: f32,
    pub animal_max_speed: f32,
    pub speed_sigmoid_steepness: f32,
}

#[derive(Resource, Debug, Clone, Copy)]

pub struct SimulationConfig {
    pub world_bounds: WorldBounds,
    pub spawn_config: SpawnConfig,
    pub grid_config: GridConfig,
    pub tuning: TuningConfig,
}

impl SimulationConfig {
    pub fn from_env() -> Self {
        Self {
            world_bounds: WorldBounds::from_env(),
            spawn_config: SpawnConfig {
                n_plants: read_env_usize("N_PLANTS", 10),
            },
            grid_config: GridConfig::from_env(),
            tuning: TuningConfig::from_env(),
        }
    }
}

impl TuningConfig {
    pub fn from_env() -> Self {
        Self {
            plant_consume_per_cell: read_env_f32("PLANT_CONSUME_PER_CELL", 20.0),
            plant_growth_per_sec: read_env_f32("PLANT_GROWTH_PER_SEC", 0.05),
            plant_max_energy: read_env_f32("PLANT_MAX_ENERGY", 120.0),
            plant_spawn_rate_per_sec: read_env_f32("PLANT_SPAWN_RATE_PER_SEC", 0.1),
            animal_crowding_penalty: read_env_f32("ANIMAL_CROWDING_PENALTY", 1.0),
            plant_crowding_penalty: read_env_f32("PLANT_CROWDING_PENALTY", 0.5),
            animal_base_energy_drain_per_sec: read_env_f32(
                "ANIMAL_BASE_ENERGY_DRAIN_PER_SEC",
                0.15,
            ),
            animal_speed_energy_drain_per_sec: read_env_f32(
                "ANIMAL_SPEED_ENERGY_DRAIN_PER_SEC",
                1.2,
            ),
            animal_speed_exponent: read_env_f32("ANIMAL_SPEED_EXPONENT", 0.9),
            animal_speed_reference: read_env_f32("ANIMAL_SPEED_REFERENCE", 220.0),
            animal_max_speed: read_env_f32("ANIMAL_MAX_SPEED", 220.0),
            speed_sigmoid_steepness: read_env_f32("SPEED_SIGMOID_STEEPNESS", 4.0),
        }
    }
}

impl GridConfig {
    pub fn from_env() -> Self {
        Self {
            resolution_x: read_env_positive_usize("GRID_RESOLUTION_X", 4),
            resolution_y: read_env_positive_usize("GRID_RESOLUTION_Y", 3),
        }
    }

    pub fn dimensions(&self, world_bounds: &WorldBounds) -> IVec2 {
        let _ = world_bounds;
        IVec2::new(self.resolution_x.max(1) as i32, self.resolution_y.max(1) as i32)
    }

    pub fn cell_size(&self, world_bounds: &WorldBounds) -> Vec2 {
        let dims = self.dimensions(world_bounds);
        Vec2::new(
            (world_bounds.half_width * 2.0) / dims.x as f32,
            (world_bounds.half_height * 2.0) / dims.y as f32,
        )
    }

    pub fn world_to_cell(&self, position: Vec2, world_bounds: &WorldBounds) -> IVec2 {
        let dims = self.dimensions(world_bounds);
        let cell_size = self.cell_size(world_bounds);
        let col = ((position.x + world_bounds.half_width) / cell_size.x).floor() as i32;
        let row = ((position.y + world_bounds.half_height) / cell_size.y).floor() as i32;

        IVec2::new(col.clamp(0, dims.x - 1), row.clamp(0, dims.y - 1))
    }

    pub fn cell_center(&self, cell: IVec2, world_bounds: &WorldBounds) -> Vec2 {
        let cell_size = self.cell_size(world_bounds);
        Vec2::new(
            -world_bounds.half_width + (cell.x as f32 + 0.5) * cell_size.x,
            -world_bounds.half_height + (cell.y as f32 + 0.5) * cell_size.y,
        )
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

fn read_env_positive_usize(name: &str, default: usize) -> usize {
    match env::var(name).ok().and_then(|value| value.parse::<usize>().ok()) {
        Some(value) if value > 0 => value,
        _ => default,
    }
}
