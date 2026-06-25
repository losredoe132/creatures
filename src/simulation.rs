use bevy::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::fs::{OpenOptions, create_dir_all};
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::brain::{think_with_vision, steering_to_acceleration};
use crate::config::{SimulationConfig, WorldBounds};
use crate::creature::{Animal, EnergyPosition, Movable, Plant};
use crate::mlp::Genome;
use crate::sense::{AnimalSnapshot, PerceptionWorld, PlantSnapshot, Vision};
use crate::utils::limit_speed_sigmoid;

pub struct SimulationPlugin;

#[derive(Resource)]
struct SimulationLog {
    start_timestamp_secs: u64,
    file: Option<std::fs::File>,
}

#[derive(Resource, Default)]
struct PlantSpawnClock {
    time_until_next: f32,
    initialized: bool,
}

#[derive(Resource, Default)]
struct AnimalSpawnClock {
    time_until_next: f32,
    initialized: bool,
}

#[derive(Resource)]
struct SimulationRng(StdRng);

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(PlantSpawnClock::default())
            .insert_resource(AnimalSpawnClock::default())
            .add_systems(Startup, (initialize_simulation_log, setup_world).chain())
            .add_systems(
                Update,
                (
                    think_animals,
                    move_animals,
                    grow_plants,
                    random_spawn_plants,
                    random_spawn_animals,
                    feed_animals_on_plant_collision,
                    despawn_starved_animals,
                    reproduce_animals,
                )
                    .chain(),
            );
    }
}

fn initialize_simulation_log(mut commands: Commands) {
    let start_timestamp_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let seed = std::env::var("RANDOM_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or_else(|| rand::random::<u64>());

    let file = match create_dir_all("logs") {
        Ok(()) => {
            let path = format!("logs/simulation_{}.log", start_timestamp_secs);
            OpenOptions::new().create(true).append(true).open(path).ok()
        }
        Err(_) => None,
    };

    let mut log = SimulationLog {
        start_timestamp_secs,
        file,
    };
    write_simulation_log(
        &mut log,
        &format!("simulation_start start_ts={} seed={}", start_timestamp_secs, seed),
    );
    commands.insert_resource(log);
    commands.insert_resource(SimulationRng(StdRng::seed_from_u64(seed)));
}

fn write_simulation_log(log: &mut SimulationLog, message: &str) {
    info!("{}", message);
    if let Some(file) = &mut log.file {
        let _ = writeln!(
            file,
            "[simulation_start_ts={}] {}",
            log.start_timestamp_secs,
            message
        );
        let _ = file.flush();
    }
}

fn setup_world(
    mut commands: Commands,
    time: Res<Time>,
    mut log: ResMut<SimulationLog>,
    config: Res<SimulationConfig>,
    mut rng: ResMut<SimulationRng>,
) {
    let animal_energy = config.spawn_config.animal_spawn_energy;
    let animal = Animal {
        position: Vec2::new(0.0, 0.0),
        velocity: Vec2::new(0.0, 0.0),
        energy: animal_energy,
        size: animal_size_from_energy(animal_energy, &config),
        color: Color::srgb(0.8, 0.2, 0.4),
        vision: Vision {
            range: config.tuning.vision_range.max(0.0),
            field_of_view_radians: config.tuning.vision_fov_radians.clamp(0.0, std::f32::consts::PI),
        },
        genome: Genome::random(&mut rng.0),
        spawn_at: time.elapsed_secs(),
        despawn_at: None,
    };


    for _ in 0..config.spawn_config.n_plants {
        spawn_random_plant(&mut commands, &config, &mut rng.0, "startup", &mut log);
    }

    commands.spawn(animal);
}

fn random_spawn_plants(
    mut commands: Commands,
    time: Res<Time>,
    config: Res<SimulationConfig>,
    mut log: ResMut<SimulationLog>,
    mut spawn_clock: ResMut<PlantSpawnClock>,
    mut rng: ResMut<SimulationRng>,
) {
    let rate = config.tuning.plant_spawn_rate_per_sec;
    if rate <= 0.0 {
        spawn_clock.initialized = false;
        return;
    }

    if !spawn_clock.initialized {
        spawn_clock.time_until_next = sample_spawn_delay(rate, &mut rng.0);
        spawn_clock.initialized = true;
    }

    spawn_clock.time_until_next -= time.delta_secs();
    while spawn_clock.time_until_next <= 0.0 {
        spawn_random_plant(&mut commands, &config, &mut rng.0, "random", &mut log);
        spawn_clock.time_until_next += sample_spawn_delay(rate, &mut rng.0);
    }
}

fn random_spawn_animals(
    mut commands: Commands,
    time: Res<Time>,
    config: Res<SimulationConfig>,
    mut log: ResMut<SimulationLog>,
    mut spawn_clock: ResMut<AnimalSpawnClock>,
    mut rng: ResMut<SimulationRng>,
) {
    let rate = config.tuning.animal_spawn_rate_per_sec;
    if rate <= 0.0 {
        spawn_clock.initialized = false;
        return;
    }

    if !spawn_clock.initialized {
        spawn_clock.time_until_next = sample_spawn_delay(rate, &mut rng.0);
        spawn_clock.initialized = true;
    }

    spawn_clock.time_until_next -= time.delta_secs();
    while spawn_clock.time_until_next <= 0.0 {
        spawn_random_animal(
            &mut commands,
            &config,
            &mut rng.0,
            "random",
            time.elapsed_secs(),
            &mut log,
        );
        spawn_clock.time_until_next += sample_spawn_delay(rate, &mut rng.0);
    }
}

fn spawn_random_plant(
    commands: &mut Commands,
    config: &SimulationConfig,
    rng: &mut impl Rng,
    source: &str,
    log: &mut SimulationLog,
) {
    let energy = 10.0;
    let plant = Plant {
        position: Vec2::new(
            rng.gen_range(-config.world_bounds.half_width..config.world_bounds.half_width),
            rng.gen_range(-config.world_bounds.half_height..config.world_bounds.half_height),
        ),
        energy,
        size: plant_size_from_energy(energy, config),
        color: Color::srgb(0.3, 0.6, 0.2),
    };
    write_simulation_log(
        log,
        &format!(
            "plant_spawn source={} x={:.2} y={:.2}",
            source, plant.position.x, plant.position.y
        ),
    );
    commands.spawn(plant);
}

fn spawn_random_animal(
    commands: &mut Commands,
    config: &SimulationConfig,
    rng: &mut impl Rng,
    source: &str,
    spawn_at: f32,
    log: &mut SimulationLog,
) {
    let energy = config.spawn_config.animal_spawn_energy;
    let animal = Animal {
        position: Vec2::new(
            rng.gen_range(-config.world_bounds.half_width..config.world_bounds.half_width),
            rng.gen_range(-config.world_bounds.half_height..config.world_bounds.half_height),
        ),
        velocity: Vec2::new(0.0, 0.0),
        energy,
        size: animal_size_from_energy(energy, config),
        color: Color::srgb(0.8, 0.2, 0.4),
        vision: Vision {
            range: config.tuning.vision_range.max(0.0),
            field_of_view_radians: config.tuning.vision_fov_radians.clamp(0.0, std::f32::consts::PI),
        },
        genome: Genome::random(rng),
        spawn_at,
        despawn_at: None,
    };
    write_simulation_log(
        log,
        &format!(
            "animal_spawn source={} x={:.2} y={:.2}",
            source, animal.position.x, animal.position.y
        ),
    );
    commands.spawn(animal);
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
        let steering = think_with_vision(
            &animal.vision,
            &animal.genome,
            animal.position,
            animal.velocity,
            &world,
        );

        let velocity = animal.velocity();
        let forward = if velocity.length_squared() > f32::EPSILON {
            velocity.normalize()
        } else {
            Vec2::X
        };

        let acceleration = steering_to_acceleration(steering, forward);
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
        let friction_factor = (1.0 - config.tuning.animal_friction * time.delta_secs()).max(0.0);
        let velocity_after_friction = animal.velocity() * friction_factor;
        animal.set_velocity(velocity_after_friction);
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

fn reproduce_animals(
    mut commands: Commands,
    mut animals: Query<&mut Animal>,
    time: Res<Time>,
    mut log: ResMut<SimulationLog>,
    config: Res<SimulationConfig>,
    mut rng: ResMut<SimulationRng>,
) {
    let spawn_energy = config.spawn_config.animal_spawn_energy.max(0.1);
    let threshold = spawn_energy * config.tuning.reproduction_energy_multiplier.max(1.0);
    let jitter = config.tuning.offspring_energy_jitter.max(0.0);
    let mutation_strength = config.tuning.genome_mutation_strength.max(0.0);
    let position_jitter = config.tuning.reproduction_position_jitter.max(0.0);

    let mut offspring = Vec::new();
    let mut reproduction_count = 0usize;

    for mut parent in &mut animals {
        if parent.energy < threshold {
            continue;
        }

        reproduction_count += 1;
        let mut offspring_energy_total = 0.0;

        for _ in 0..2 {
            let energy_factor = 1.0 + rng.0.gen_range(-jitter..jitter);
            let child_energy = (spawn_energy * energy_factor).max(0.1);
            offspring_energy_total += child_energy;

            let mut child_position = parent.position
                + Vec2::new(
                    rng.0.gen_range(-position_jitter..position_jitter),
                    rng.0.gen_range(-position_jitter..position_jitter),
                );
            let mut child_translation = child_position.extend(0.0);
            ensure_torodial_world(&mut child_translation, &config.world_bounds);
            child_position = child_translation.xy();

            let child_velocity = parent.velocity
                + Vec2::new(rng.0.gen_range(-5.0..5.0), rng.0.gen_range(-5.0..5.0));

            offspring.push(Animal {
                position: child_position,
                velocity: child_velocity,
                energy: child_energy,
                size: animal_size_from_energy(child_energy, &config),
                color: parent.color,
                vision: parent.vision,
                genome: parent.genome.mutated(&mut rng.0, mutation_strength),
                spawn_at: time.elapsed_secs(),
                despawn_at: None,
            });
        }

        parent.energy = (parent.energy - offspring_energy_total).max(0.0);
        parent.size = animal_size_from_energy(parent.energy, &config);
    }

    for child in offspring {
        commands.spawn(child);
    }

    if reproduction_count > 0 {
        write_simulation_log(
            &mut log,
            &format!(
                "animal_reproduction parents={} offspring={}",
                reproduction_count,
                reproduction_count * 2
            ),
        );
    }
}

fn despawn_starved_animals(
    mut commands: Commands,
    time: Res<Time>,
    mut animals: Query<(Entity, &mut Animal)>,
    mut log: ResMut<SimulationLog>,
) {
    let mut despawn_count = 0usize;
    for (entity, mut animal) in &mut animals {
        if animal.energy <= 0.0 {
            animal.despawn_at = Some(time.elapsed_secs());
            write_simulation_log(
                &mut log,
                &format!(
                    "animal_despawn reason=starvation spawn_at={:.3} despawn_at={:.3} genome={:?}",
                    animal.spawn_at,
                    animal.despawn_at.unwrap_or_default(),
                    animal.genome.genes
                ),
            );
            commands.entity(entity).despawn();
            despawn_count += 1;
        }
    }

    if despawn_count > 0 {
        write_simulation_log(
            &mut log,
            &format!("animal_starvation_despawn count={}", despawn_count),
        );
    }
}

fn feed_animals_on_plant_collision(
    mut commands: Commands,
    config: Res<SimulationConfig>,
    mut log: ResMut<SimulationLog>,
    mut entities: ParamSet<(
        Query<(Entity, &Animal)>,
        Query<&mut Animal>,
        Query<(Entity, &Plant)>,
        Query<&mut Plant>,
    )>,
) {
    let animals_snapshot = entities
        .p0()
        .iter()
        .map(|(entity, animal)| (entity, animal.position, animal.size))
        .collect::<Vec<_>>();
    let plants_snapshot = entities
        .p2()
        .iter()
        .map(|(entity, plant)| (entity, plant.position, plant.size, plant.energy))
        .collect::<Vec<_>>();

    if animals_snapshot.is_empty() || plants_snapshot.is_empty() {
        return;
    }

    let consume_per_collision = config.tuning.plant_consume_per_collision.max(0.0);
    if consume_per_collision <= 0.0 {
        return;
    }

    let mut animal_gain_by_entity: HashMap<Entity, f32> = HashMap::new();
    let mut plant_taken_by_entity: HashMap<Entity, f32> = HashMap::new();
    let mut collision_count = 0usize;

    for (animal_entity, animal_position, animal_radius) in &animals_snapshot {
        for (plant_entity, plant_position, plant_radius, plant_energy) in &plants_snapshot {
            let combined_radius = *animal_radius + *plant_radius;
            let overlaps = animal_position.distance_squared(*plant_position)
                <= combined_radius * combined_radius;
            if !overlaps {
                continue;
            }

            let already_taken = plant_taken_by_entity.get(plant_entity).copied().unwrap_or(0.0);
            let remaining_energy = (*plant_energy - already_taken).max(0.0);
            if remaining_energy <= 0.0 {
                continue;
            }

            let taken = consume_per_collision.min(remaining_energy);
            if taken <= 0.0 {
                continue;
            }

            collision_count += 1;
            *animal_gain_by_entity.entry(*animal_entity).or_insert(0.0) += taken;
            *plant_taken_by_entity.entry(*plant_entity).or_insert(0.0) += taken;
        }
    }

    if collision_count == 0 {
        return;
    }

    for (animal_entity, gained_energy) in &animal_gain_by_entity {
        if let Ok(mut animal) = entities.p1().get_mut(*animal_entity) {
            animal.energy += *gained_energy;
            animal.size = animal_size_from_energy(animal.energy, &config);
        }
    }

    let mut to_despawn = Vec::new();
    let mut consumed_energy = 0.0;
    let mut depleted_plants = 0usize;
    for (plant_entity, taken_energy) in &plant_taken_by_entity {
        if let Ok(mut plant) = entities.p3().get_mut(*plant_entity) {
            plant.energy = (plant.energy - *taken_energy).max(0.0);
            plant.size = plant_size_from_energy(plant.energy, &config);
            consumed_energy += *taken_energy;
            if plant.energy <= 0.0 {
                depleted_plants += 1;
                to_despawn.push(*plant_entity);
            }
        }
    }

    if !to_despawn.is_empty() {
        write_simulation_log(
            &mut log,
            &format!("collision_event type=despawn_plants count={}", to_despawn.len()),
        );
    }

    write_simulation_log(
        &mut log,
        &format!(
            "collision_event type=feeding collisions={} fed_animals={} touched_plants={} consumed_energy={:.2} depleted_plants={}",
            collision_count,
            animal_gain_by_entity.len(),
            plant_taken_by_entity.len(),
            consumed_energy,
            depleted_plants
        ),
    );

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
