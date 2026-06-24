use bevy::prelude::*;
use rand::Rng;
use std::collections::{HashMap, HashSet};

use crate::brain::{think_with_vision, Brain};
use crate::config::{SimulationConfig, WorldBounds};
use crate::creature::{Animal, EnergyPosition, Movable, Plant};
use crate::sense::{AnimalSnapshot, PerceptionWorld, PlantSnapshot, Vision};
use crate::utils::limit_speed_sigmoid;

pub struct SimulationPlugin;

#[derive(Resource, Default)]
pub struct SharedGridCells {
    pub cells: Vec<GridCellOccupancy>,
}

#[derive(Resource, Default)]
struct PlantSpawnClock {
    time_until_next: f32,
    initialized: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct GridCellOccupancy {
    pub cell: IVec2,
    pub animal_count: usize,
    pub plant_count: usize,
}

impl GridCellOccupancy {
    pub fn total(&self) -> usize {
        self.animal_count + self.plant_count
    }
}

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(SharedGridCells::default())
            .insert_resource(PlantSpawnClock::default())
            .add_systems(Startup, setup_world)
            .add_systems(
                Update,
                (
                    think_animals,
                    move_animals,
                    grow_plants,
                    random_spawn_plants,
                    detect_shared_grid_cells,
                    apply_shared_cell_rules,
                )
                    .chain(),
            );
    }
}

fn setup_world(mut commands: Commands, config: Res<SimulationConfig>) {
    let animal_energy = 100.0;
    let animal = Animal {
        position: Vec2::new(0.0, 0.0),
        velocity: Vec2::new(100.0, 1.0),
        energy: animal_energy,
        size: animal_size_from_energy(animal_energy, &config),
        color: Color::srgb(0.8, 0.2, 0.4),
        vision: Vision::default(),
        brain: Brain::default(),
    };


    let mut rng = rand::thread_rng();
    for _ in 0..config.spawn_config.n_plants {
        spawn_random_plant(&mut commands, &config, &mut rng, "startup");
    }

    commands.spawn(animal);
}

fn random_spawn_plants(
    mut commands: Commands,
    time: Res<Time>,
    config: Res<SimulationConfig>,
    mut spawn_clock: ResMut<PlantSpawnClock>,
) {
    let rate = config.tuning.plant_spawn_rate_per_sec;
    if rate <= 0.0 {
        spawn_clock.initialized = false;
        return;
    }

    let mut rng = rand::thread_rng();
    if !spawn_clock.initialized {
        spawn_clock.time_until_next = sample_spawn_delay(rate, &mut rng);
        spawn_clock.initialized = true;
    }

    spawn_clock.time_until_next -= time.delta_secs();
    while spawn_clock.time_until_next <= 0.0 {
        spawn_random_plant(&mut commands, &config, &mut rng, "random");
        spawn_clock.time_until_next += sample_spawn_delay(rate, &mut rng);
    }
}

fn spawn_random_plant(
    commands: &mut Commands,
    config: &SimulationConfig,
    rng: &mut impl Rng,
    source: &str,
) {
    let energy = 60.0;
    let plant = Plant {
        position: Vec2::new(
            rng.gen_range(-config.world_bounds.half_width..config.world_bounds.half_width),
            rng.gen_range(-config.world_bounds.half_height..config.world_bounds.half_height),
        ),
        energy,
        size: plant_size_from_energy(energy, config),
        color: Color::srgb(0.3, 0.6, 0.2),
    };
    info!(
        "plant_spawn source={} x={:.2} y={:.2}",
        source, plant.position.x, plant.position.y
    );
    commands.spawn(plant);
}

fn sample_spawn_delay(rate_per_sec: f32, rng: &mut impl Rng) -> f32 {
    let uniform = rng.gen_range(f32::EPSILON..1.0);
    -uniform.ln() / rate_per_sec
}

fn think_animals(
    mut animals: Query<&mut Animal>,
    plants: Query<&Plant>,
    time: Res<Time>,
    config: Res<SimulationConfig>,
) {
    let plants_snapshot: Vec<PlantSnapshot> = plants
        .iter()
        .map(|plant| PlantSnapshot {
            position: plant.position,
            energy: plant.energy,
            radius: plant.size,
        })
        .collect();

    let animals_snapshot: Vec<AnimalSnapshot> = animals
        .iter()
        .map(|animal| AnimalSnapshot {
            position: animal.position,
            energy: animal.energy,
            radius: animal.size,
        })
        .collect();

    let world = PerceptionWorld {
        plants: &plants_snapshot,
        animals: &animals_snapshot,
    };

    for mut animal in &mut animals {
        let acceleration = think_with_vision(
            &animal.vision,
            &animal.brain,
            animal.position,
            animal.velocity,
            &world,
        );
            animal.apply_acceleration(acceleration, time.delta_secs());
        let limited_velocity = limit_speed_sigmoid(
            animal.velocity(),
            config.tuning.animal_max_speed,
            config.tuning.speed_sigmoid_steepness,
        );
        animal.set_velocity(limited_velocity);
    }
}

fn move_animals(
    mut query: Query<(&mut Animal, &mut Transform)>,
    time: Res<Time>,
    config: Res<SimulationConfig>,
) {
    for (mut animal, mut transform) in &mut query {
        transform.translation += animal.velocity().extend(0.0) * time.delta_secs();
        ensure_torodial_world(&mut transform.translation, &config.world_bounds);
        animal.set_position(transform.translation.xy());

        let speed_ratio = (animal.velocity().length() / config.tuning.animal_speed_reference).max(0.0);
        let speed_drain =
            config.tuning.animal_speed_energy_drain_per_sec
                * ((config.tuning.animal_speed_exponent * speed_ratio).exp() - 1.0);
        let energy_drain =
            (config.tuning.animal_base_energy_drain_per_sec + speed_drain) * time.delta_secs();
        let updated_energy = (animal.energy() - energy_drain).max(0.0);
        animal.set_energy(updated_energy);
        animal.size = animal_size_from_energy(animal.energy, &config);
    }
}

fn grow_plants(
    mut plants: Query<&mut Plant>,
    time: Res<Time>,
    config: Res<SimulationConfig>,
) {
    let growth = config.tuning.plant_growth_per_sec * time.delta_secs();
    if growth <= 0.0 {
        return;
    }

    for mut plant in &mut plants {
        plant.energy = (plant.energy + growth).min(config.tuning.plant_max_energy);
        plant.size = plant_size_from_energy(plant.energy, &config);
    }
}

fn detect_shared_grid_cells(
    animals: Query<&Animal>,
    plants: Query<&Plant>,
    config: Res<SimulationConfig>,
    mut shared_cells: ResMut<SharedGridCells>,
) {
    let mut counts: HashMap<IVec2, (usize, usize)> = HashMap::new();

    for animal in &animals {
        let cell = config
            .grid_config
            .world_to_cell(animal.position, &config.world_bounds);
        let entry = counts.entry(cell).or_insert((0, 0));
        entry.0 += 1;
    }

    for plant in &plants {
        let cell = config
            .grid_config
            .world_to_cell(plant.position, &config.world_bounds);
        let entry = counts.entry(cell).or_insert((0, 0));
        entry.1 += 1;
    }

    let mut occupied = counts
        .into_iter()
        .filter_map(|(cell, (animal_count, plant_count))| {
            let occupancy = GridCellOccupancy {
                cell,
                animal_count,
                plant_count,
            };
            (occupancy.total() > 1).then_some(occupancy)
        })
        .collect::<Vec<_>>();
    occupied.sort_by_key(|occupancy| (occupancy.cell.x, occupancy.cell.y));
    shared_cells.cells = occupied;
}

fn apply_shared_cell_rules(
    mut commands: Commands,
    shared_cells: Res<SharedGridCells>,
    config: Res<SimulationConfig>,
    mut entities: ParamSet<(
        Query<(Entity, &Animal)>,
        Query<&mut Animal>,
        Query<(Entity, &Plant)>,
        Query<&mut Plant>,
    )>,
) {
    if shared_cells.cells.is_empty() {
        return;
    }

    let mut animals_by_cell: HashMap<IVec2, Vec<Entity>> = HashMap::new();
    for (entity, animal) in &entities.p0() {
        let cell = config
            .grid_config
            .world_to_cell(animal.position, &config.world_bounds);
        animals_by_cell.entry(cell).or_default().push(entity);
    }

    let mut plants_by_cell: HashMap<IVec2, Vec<Entity>> = HashMap::new();
    for (entity, plant) in &entities.p2() {
        let cell = config
            .grid_config
            .world_to_cell(plant.position, &config.world_bounds);
        plants_by_cell.entry(cell).or_default().push(entity);
    }

    let mut to_despawn: HashSet<Entity> = HashSet::new();

    for occupancy in &shared_cells.cells {
        let cell = occupancy.cell;
        let animals = animals_by_cell
            .get(&cell)
            .cloned()
            .unwrap_or_default();
        let plants = plants_by_cell
            .get(&cell)
            .cloned()
            .unwrap_or_default();

        if !animals.is_empty() && !plants.is_empty() {
            let mut consumed_energy = 0.0;
            let mut depleted_plants = 0usize;
            for plant_entity in &plants {
                if let Ok(mut plant) = entities.p3().get_mut(*plant_entity) {
                    let taken = plant.energy.min(config.tuning.plant_consume_per_cell);
                    plant.energy -= taken;
                    plant.size = plant_size_from_energy(plant.energy, &config);
                    consumed_energy += taken;
                    if plant.energy <= 0.0 {
                        depleted_plants += 1;
                        to_despawn.insert(*plant_entity);
                    }
                }
            }

            let gain_per_animal = consumed_energy / animals.len() as f32;
            for animal_entity in &animals {
                if let Ok(mut animal) = entities.p1().get_mut(*animal_entity) {
                    animal.energy += gain_per_animal;
                    animal.size = animal_size_from_energy(animal.energy, &config);
                }
            }

            info!(
                "cell_event type=feeding cell=({}, {}) animals={} plants={} consumed_energy={:.2} gain_per_animal={:.2} depleted_plants={}",
                cell.x,
                cell.y,
                animals.len(),
                plants.len(),
                consumed_energy,
                gain_per_animal,
                depleted_plants
            );
        }

        if animals.len() > 1 {
            for animal_entity in &animals {
                if let Ok(mut animal) = entities.p1().get_mut(*animal_entity) {
                    animal.energy = (animal.energy - config.tuning.animal_crowding_penalty).max(0.0);
                    animal.size = animal_size_from_energy(animal.energy, &config);
                }
            }

            info!(
                "cell_event type=animal_crowding cell=({}, {}) animals={} penalty_per_animal={:.2}",
                cell.x,
                cell.y,
                animals.len(),
                config.tuning.animal_crowding_penalty
            );
        }

        if plants.len() > 1 {
            let mut depleted_plants = 0usize;
            for plant_entity in &plants {
                if let Ok(mut plant) = entities.p3().get_mut(*plant_entity) {
                    plant.energy = (plant.energy - config.tuning.plant_crowding_penalty).max(0.0);
                    plant.size = plant_size_from_energy(plant.energy, &config);
                    if plant.energy <= 0.0 {
                        depleted_plants += 1;
                        to_despawn.insert(*plant_entity);
                    }
                }
            }

            info!(
                "cell_event type=plant_crowding cell=({}, {}) plants={} penalty_per_plant={:.2} depleted_plants={}",
                cell.x,
                cell.y,
                plants.len(),
                config.tuning.plant_crowding_penalty,
                depleted_plants
            );
        }
    }

    if !to_despawn.is_empty() {
        info!("cell_event type=despawn_plants count={}", to_despawn.len());
    }
    for entity in to_despawn {
        commands.entity(entity).despawn();
    }
}

fn plant_size_from_energy(energy: f32, config: &SimulationConfig) -> f32 {
    let clamped_energy = energy.max(0.0);
    let size = config.tuning.plant_base_size
        + config.tuning.plant_size_per_sqrt_energy * clamped_energy.sqrt();
    size.max(1.0)
}

fn animal_size_from_energy(energy: f32, config: &SimulationConfig) -> f32 {
    let clamped_energy = energy.max(0.0);
    let size = config.tuning.animal_base_size
        + config.tuning.animal_size_per_sqrt_energy * clamped_energy.sqrt();
    size.max(1.0)
}

fn ensure_torodial_world(translation: &mut Vec3, world_bounds: &WorldBounds) {
    if translation.x < -world_bounds.half_width {
        translation.x = world_bounds.half_width;
    } else if translation.x > world_bounds.half_width {
        translation.x = -world_bounds.half_width;
    }

    if translation.y < -world_bounds.half_height {
        translation.y = world_bounds.half_height;
    } else if translation.y > world_bounds.half_height {
        translation.y = -world_bounds.half_height;
    }
}
