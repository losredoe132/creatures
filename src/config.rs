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
    pub animal_spawn_energy: f32,
}

#[derive(Resource, Debug, Clone, Copy)]
pub struct TuningConfig {
    pub plant_consume_per_collision: f32,
    pub herbivore_metabolism_ratio: f32,
    pub carnivore_metabolism_ratio: f32,
    pub omnivore_metabolism_ratio: f32,
    pub plant_growth_per_sec: f32,
    pub plant_max_energy: f32,
    pub plant_spawn_rate_per_sec: f32,
    pub animal_spawn_rate_per_sec: f32,
    pub plant_base_size: f32,
    pub plant_size_per_sqrt_energy: f32,
    pub animal_base_size: f32,
    pub animal_size_per_sqrt_energy: f32,
    pub animal_base_energy_drain_per_sec: f32,
    pub animal_speed_energy_drain_per_sec: f32,
    pub animal_speed_exponent: f32,
    pub animal_speed_reference: f32,
    pub animal_max_speed: f32,
    pub speed_sigmoid_steepness: f32,
    pub animal_friction: f32,
    pub vision_range: f32,
    pub vision_fov_radians: f32,
    pub reproduction_energy_multiplier: f32,
    pub offspring_energy_jitter: f32,
    pub genome_mutation_strength: f32,
    pub reproduction_position_jitter: f32,
}

#[derive(Resource, Debug, Clone, Copy)]

pub struct SimulationConfig {
    pub world_bounds: WorldBounds,
    pub spawn_config: SpawnConfig,
    pub tuning: TuningConfig,
}

impl SimulationConfig {
    pub fn from_env() -> Self {
        Self {
            world_bounds: WorldBounds::from_env(),
            spawn_config: SpawnConfig {
                n_plants: read_env_usize("N_PLANTS", 10),
                animal_spawn_energy: read_env_f32("ANIMAL_SPAWN_ENERGY", 10.0),
            },
            tuning: TuningConfig::from_env(),
        }
    }
}

impl TuningConfig {
    pub fn from_env() -> Self {
        Self {
            plant_consume_per_collision: read_env_f32(
                "PLANT_CONSUME_PER_COLLISION",
                read_env_f32("PLANT_CONSUME_PER_CELL", 20.0),
            ),
            herbivore_metabolism_ratio: read_env_f32("HERBIVORE_METABOLISM_RATIO", 1.4),
            carnivore_metabolism_ratio: read_env_f32("CARNIVORE_METABOLISM_RATIO", 1.1),
            omnivore_metabolism_ratio: read_env_f32("OMNIVORE_METABOLISM_RATIO", 0.6),
            plant_growth_per_sec: read_env_f32("PLANT_GROWTH_PER_SEC", 0.05),
            plant_max_energy: read_env_f32("PLANT_MAX_ENERGY", 120.0),
            plant_spawn_rate_per_sec: read_env_f32("PLANT_SPAWN_RATE_PER_SEC", 0.1),
            animal_spawn_rate_per_sec: read_env_f32("ANIMAL_SPAWN_RATE_PER_SEC", 0.02),
            plant_base_size: read_env_f32("PLANT_BASE_SIZE", 4.0),
            plant_size_per_sqrt_energy: read_env_f32("PLANT_SIZE_PER_SQRT_ENERGY", 1.3),
            animal_base_size: read_env_f32("ANIMAL_BASE_SIZE", 6.0),
            animal_size_per_sqrt_energy: read_env_f32("ANIMAL_SIZE_PER_SQRT_ENERGY", 1.4),
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
            animal_friction: read_env_f32("ANIMAL_FRICTION", 2.5),
            vision_range: read_env_f32("VISION_RANGE", 200.0),
            vision_fov_radians: read_env_f32("VISION_FOV_RADIANS", std::f32::consts::FRAC_PI_4),
            reproduction_energy_multiplier: read_env_f32("REPRODUCTION_ENERGY_MULTIPLIER", 3.0),
            offspring_energy_jitter: read_env_f32("OFFSPRING_ENERGY_JITTER", 0.1),
            genome_mutation_strength: read_env_f32("GENOME_MUTATION_STRENGTH", 0.05),
            reproduction_position_jitter: read_env_f32("REPRODUCTION_POSITION_JITTER", 18.0),
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
