use bevy::prelude::*;
use rand::Rng;
use rand::RngCore;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::{BTreeMap, HashMap};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::brain::think_with_vision;
use crate::config::{SimulationConfig, WorldBounds};
use crate::creature::{Animal, Carcass, Diet, EnergyPosition, Movable, Plant};
use crate::logging::{ConsoleBackend, SimulationLogger, TextFileBackend};
use crate::mlp::{GENOME_LEN, Genome, MLP_OUTPUTS};
use crate::sense::{AnimalSnapshot, CarcassSnapshot, PerceptionWorld, PlantSnapshot};
use crate::utils::size_from_energy;
use crate::zoo::{Zoo, ZooAnimal};

#[derive(Message, Clone)]
pub struct ManualZooSpawnEvent;

pub struct SimulationPlugin;

#[derive(Resource, Default)]
pub struct GlobalFrameCounter(pub u64);

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

#[derive(Resource, Default, Debug, PartialEq, Eq)]
pub struct AnimalPopulation {
    pub carnivores: usize,
    pub herbivores: usize,
    pub omnivores: usize,
    pub scavengers: usize,
}

#[derive(Resource, Default)]
pub struct PopulationSizeTracker {
    pub plants: usize,
    pub animals: AnimalPopulation,
    pub families: BTreeMap<u32, usize>,
    pub initialized: bool,
}

#[derive(Resource)]
struct SimulationRng(StdRng);

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<ManualZooSpawnEvent>()
            .insert_resource(GlobalFrameCounter::default())
            .insert_resource(PlantSpawnClock::default())
            .insert_resource(AnimalSpawnClock::default())
            .insert_resource(PopulationSizeTracker::default())
            .add_systems(Startup, (initialize_simulation_log, setup_world).chain())
            .add_systems(
                First,
                (advance_global_frame_counter, sync_logger_with_frame_counter)
                    .chain()
                    .run_if(not_paused),
            )
            .add_systems(
                Update,
                (
                    think_animals,
                    move_animals,
                    grow_plants,
                    random_spawn_plants,
                    random_spawn_animals,
                    handle_object_collision,
                    despawn_starved_animals,
                    log_removed_animals,
                    reproduce_animals,
                )
                    .chain()
                    .run_if(not_paused),
            )
            .add_systems(Update, handle_manual_zoo_spawn)
            .add_systems(PostUpdate, log_population_size_changes);
        app.add_systems(Last, despawn_animals_on_shutdown);
    }
}

fn not_paused(time: Res<Time<Virtual>>) -> bool {
    !time.is_paused()
}

fn sync_logger_with_frame_counter(
    frame_counter: Res<GlobalFrameCounter>,
    mut log: ResMut<SimulationLogger>,
) {
    log.set_frame(frame_counter.0);
}

fn log_population_size_changes(
    plants: Query<&Plant>,
    animals: Query<&Animal>,
    mut tracker: ResMut<PopulationSizeTracker>,
    mut log: ResMut<SimulationLogger>,
) {
    let plant_count = plants.iter().count();
    let mut population = AnimalPopulation::default();
    let mut family_counts: BTreeMap<u32, usize> = BTreeMap::new();

    for animal in &animals {
        match animal.diet {
            Diet::Carnivore => population.carnivores += 1,
            Diet::Herbivore => population.herbivores += 1,
            Diet::Omnivore => population.omnivores += 1,
            Diet::Scavenger => population.scavengers += 1,
        }
        *family_counts.entry(animal.family).or_insert(0) += 1;
    }

    if !tracker.initialized {
        tracker.plants = plant_count;
        tracker.animals = population;
        tracker.families = family_counts;
        tracker.initialized = true;
        return;
    }

    if tracker.plants != plant_count
        || tracker.animals != population
        || tracker.families != family_counts
    {
        let family_report = if family_counts.is_empty() {
            "none".to_string()
        } else {
            family_counts
                .iter()
                .map(|(family, count)| format!("{}:{}", family, count))
                .collect::<Vec<String>>()
                .join("|")
        };

        log.info(&format!(
            "population_size plants={} animals={{carnivores:{} herbivores:{} omnivores:{} scavengers:{}}} families={}",
            plant_count,
            population.carnivores,
            population.herbivores,
            population.omnivores,
            population.scavengers,
            family_report
        ));
        tracker.plants = plant_count;
        tracker.animals = population;
        tracker.families = family_counts;
    }
}

fn advance_global_frame_counter(mut frame_counter: ResMut<GlobalFrameCounter>) {
    frame_counter.0 = frame_counter.0.saturating_add(1);
}

fn despawn_animal(
    commands: &mut Commands,
    log: &mut SimulationLogger,
    zoo: &mut Zoo,
    entity: Entity,
    animal: &mut Animal,
    reason: &str,
    despawn_frame: u64,
    config: &SimulationConfig,
) {
    animal.despawn_at = Some(despawn_frame);
    let lifetime_duration = animal.despawn_at.unwrap_or_default() - animal.spawn_at;
    zoo.consider_and_persist(
        ZooAnimal {
            lifetime_frames: lifetime_duration,
            diet: animal.diet,
            genome: animal.genome.clone(),
            family: animal.family,
            generation: animal.generation,
        },
        log,
    );
    log.info(&format!(
        "animal_despawn reason={} lifetime_frames={} animal={:?}",
        reason, lifetime_duration, animal
    ));

    let carcass_energy = animal.initial_energy * 0.5;
    commands.spawn(Carcass {
        position: animal.position,
        energy: carcass_energy,
        size: size_from_energy(carcass_energy, config),
    });

    commands.entity(entity).despawn();
}

fn despawn_animals_on_shutdown(
    mut exit_events: MessageReader<AppExit>,
    mut commands: Commands,
    frame_count: Res<GlobalFrameCounter>,
    config: Res<SimulationConfig>,
    mut animals: Query<(Entity, &mut Animal)>,
    mut log: ResMut<SimulationLogger>,
    mut zoo: ResMut<Zoo>,
) {
    if exit_events.read().next().is_none() {
        return;
    }

    let mut despawn_count = 0usize;
    for (entity, mut animal) in &mut animals {
        despawn_animal(
            &mut commands,
            &mut log,
            &mut zoo,
            entity,
            &mut animal,
            "shutdown",
            frame_count.0,
            &config,
        );
        despawn_count += 1;
    }

    if despawn_count > 0 {
        log.info(&format!(
            "shutdown_cleanup despawned_animals={}",
            despawn_count
        ));
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

    let mut logger = SimulationLogger::new(start_timestamp_secs);
    logger.add_backend(ConsoleBackend);
    if let Some(backend) =
        TextFileBackend::new(&format!("logs/simulation_{}.log", start_timestamp_secs))
    {
        logger.add_backend(backend);
    }
    logger.log(&format!(
        "simulation_start start_ts={} seed={}",
        start_timestamp_secs, seed
    ));

    let zoo = Zoo::load_default(&mut logger);

    commands.insert_resource(logger);
    commands.insert_resource(SimulationRng(StdRng::seed_from_u64(seed)));
    commands.insert_resource(zoo);
}

fn get_optimal_herbivore_genome() -> Vec<f32> {
    let mut genome = vec![0.0; GENOME_LEN];
    genome[0] = 4.0;
    genome[MLP_OUTPUTS + 1] = 4.0;

    // Run from carnivores
    genome[4 * 2] = -3.9; // genome[8]
    genome[5 * 2 + 1] = -4.0; // genome[11]

    // Run from omnivores
    genome[14 * 2] = -4.0; // genome[28]
    genome[15 * 2 + 1] = -3.9; // genome[31]

    genome
}

fn get_optimal_carnivore_genome() -> Vec<f32> {
    let mut genome = vec![0.0; GENOME_LEN];

    // Run from carnivores
    genome[4 * 2] = -0.3; // genome[8]
    genome[5 * 2 + 1] = -0.3; // genome[11]

    // Run to herbivores
    genome[9 * 2] = 1.0; // genome[18]
    genome[10 * 2 + 1] = 1.0; // genome[21]

    // Run from omnivores
    genome[14 * 2] = 1.0; // genome[28]
    genome[15 * 2 + 1] = 1.0; // genome[31]

    genome
}

fn get_optimal_scavenger_genome() -> Vec<f32> {
    let mut genome = vec![0.0; GENOME_LEN];

    // Move towards carcasses
    genome[22 * 2] = 1.0; // genome[44]
    genome[23 * 2 + 1] = 1.0; // genome[47]

    // Run from carnivores
    genome[4 * 2] = -1.0; // genome[8]
    genome[5 * 2 + 1] = -1.0; // genome[11]

    genome
}

fn get_optimal_omnivore_genome() -> Vec<f32> {
    let mut genome = vec![0.0; GENOME_LEN];
    genome[0] = 4.0;
    genome[MLP_OUTPUTS + 1] = 4.0;

    // Run from carnivores
    genome[4 * 2] = -1.0; // genome[8]
    genome[5 * 2 + 1] = -1.0; // genome[11]

    // Run to herbivores
    genome[9 * 2] = 1.0; // genome[18]
    genome[10 * 2 + 1] = 1.0; // genome[21]

    // Run from omnivores
    genome[14 * 2] = 1.0; // genome[28]
    genome[15 * 2 + 1] = 1.0; // genome[31]

    genome
}

fn setup_world(
    mut commands: Commands,
    frame_count: Res<GlobalFrameCounter>,
    mut log: ResMut<SimulationLogger>,
    config: Res<SimulationConfig>,
    mut rng: ResMut<SimulationRng>,
) {
    // let animal_herbivor = Animal::new(
    //     rng.0.next_u64(),
    //     None,
    //     Diet::Herbivore,
    //     Vec2::new(100.0, 100.0),
    //     Vec2::new(0.0, 0.0),
    //     Genome {
    //         genes: get_optimal_herbivore_genome(),
    //     },
    //     &frame_count,
    //     &config,
    //     0,
    //     0,
    // );

    // commands.spawn(animal_herbivor);

    // let animal_carnivor = Animal::new(
    //     rng.0.next_u64(),
    //     None,
    //     Diet::Carnivore,
    //     Vec2::new(0.0, 0.0),
    //     Vec2::new(0.0, 100.0),
    //     Genome {
    //         genes: get_optimal_carnivore_genome(),
    //     },
    //     &frame_count,
    //     &config,
    //     1,
    //     0,
    // );

    // commands.spawn(animal_carnivor);

    // let animal_omnivor = Animal::new(
    //     rng.0.next_u64(),
    //     None,
    //     Diet::Omnivore,
    //     Vec2::new(-100.0, 100.0),
    //     Vec2::new(100.0, 0.0),
    //     Genome {
    //         genes: get_optimal_omnivore_genome(),
    //     },
    //     &frame_count,
    //     &config,
    //     2,
    //     0,
    // );

    // commands.spawn(animal_omnivor);

    // let animal_scavenger = Animal::new(
    //     rng.0.next_u64(),
    //     None,
    //     Diet::Scavenger,
    //     Vec2::new(100.0, -100.0),
    //     Vec2::new(0.0, 0.0),
    //     Genome {
    //         genes: get_optimal_scavenger_genome(),
    //     },
    //     &frame_count,
    //     &config,
    //     3,
    //     0,
    // );

    // commands.spawn(animal_scavenger);

    for _ in 0..config.spawn_config.n_plants {
        spawn_random_plant(&mut commands, &config, &mut rng.0, "startup", &mut *log);
    }
}

fn random_spawn_plants(
    mut commands: Commands,
    time: Res<Time>,
    config: Res<SimulationConfig>,
    mut log: ResMut<SimulationLogger>,
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
        spawn_random_plant(&mut commands, &config, &mut rng.0, "random", &mut *log);
        spawn_clock.time_until_next += sample_spawn_delay(rate, &mut rng.0);
    }
}

fn random_spawn_animals(
    mut commands: Commands,
    time: Res<Time>,
    frame_count: Res<GlobalFrameCounter>,
    config: Res<SimulationConfig>,
    animals: Query<&Animal>,
    mut log: ResMut<SimulationLogger>,
    mut spawn_clock: ResMut<AnimalSpawnClock>,
    mut rng: ResMut<SimulationRng>,
    mut zoo: ResMut<Zoo>,
) {
    if animals.iter().next().is_none() {
        spawn_random_animal(
            &mut commands,
            &config,
            &mut rng.0,
            "population_recovery",
            &frame_count,
            &mut *log,
            &mut zoo,
        );
    }

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
            &frame_count,
            &mut *log,
            &mut zoo,
        );
        spawn_clock.time_until_next += sample_spawn_delay(rate, &mut rng.0);
    }
}

fn spawn_random_plant(
    commands: &mut Commands,
    config: &SimulationConfig,
    rng: &mut impl Rng,
    source: &str,
    log: &mut SimulationLogger,
) {
    let energy = rng
        .gen_range(5.0..=config.tuning.plant_max_energy / 10.0)
        .max(5.0);
    let plant = Plant {
        position: Vec2::new(
            rng.gen_range(-config.world_bounds.half_width..config.world_bounds.half_width),
            rng.gen_range(-config.world_bounds.half_height..config.world_bounds.half_height),
        ),
        energy,
        size: size_from_energy(energy, config),
        color: Color::srgb(0.3, 0.6, 0.2),
    };
    log.debug(&format!(
        "plant_spawn source={} x={:.2} y={:.2}",
        source, plant.position.x, plant.position.y
    ));
    commands.spawn(plant);
}

fn spawn_plant_nearby(
    commands: &mut Commands,
    config: &SimulationConfig,
    rng: &mut impl Rng,
    parent_position: Vec2,
    log: &mut SimulationLogger,
) {
    let spawn_radius = (config.tuning.plant_base_size * 3.0).max(2.0);
    let mut spawn_translation = (parent_position
        + Vec2::new(
            rng.gen_range(-spawn_radius..spawn_radius),
            rng.gen_range(-spawn_radius..spawn_radius),
        ))
    .extend(0.0);
    ensure_torodial_world(&mut spawn_translation, &config.world_bounds);

    let energy = rng
        .gen_range(5.0..=config.tuning.plant_max_energy / 10.0)
        .max(5.0);
    let plant = Plant {
        position: spawn_translation.xy(),
        energy,
        size: size_from_energy(energy, config),
        color: Color::srgb(0.3, 0.6, 0.2),
    };
    log.debug(&format!(
        "plant_spawn source=growth x={:.2} y={:.2}",
        plant.position.x, plant.position.y
    ));
    commands.spawn(plant);
}

fn spawn_random_animal(
    commands: &mut Commands,
    config: &SimulationConfig,
    rng: &mut impl Rng,
    source: &str,
    frame_count: &Res<GlobalFrameCounter>,
    log: &mut SimulationLogger,
    zoo: &mut Zoo,
) {
    let sampled = zoo.maybe_sample(rng, config.tuning.zoo_spawn_probability);
    let (diet, genome, family, generation, spawn_source) = if let Some(top) = sampled {
        (
            top.diet,
            top.genome.clone(),
            top.family,
            top.generation,
            "zoo",
        )
    } else {
        (
            Diet::random(rng),
            Genome::random(rng),
            rng.next_u32(),
            0,
            source,
        )
    };

    let animal = Animal::new(
        rng.next_u64(),
        None,
        diet,
        Vec2::new(
            rng.gen_range(-config.world_bounds.half_width..config.world_bounds.half_width),
            rng.gen_range(-config.world_bounds.half_height..config.world_bounds.half_height),
        ),
        Vec2::new(rng.gen_range(0.0..100.0), rng.gen_range(0.0..100.0)),
        genome,
        frame_count,
        config,
        family,
        generation,
    );
    log.debug(&format!(
        "animal_spawn source={} x={:.2} y={:.2}, diet ={:?},family={}",
        spawn_source, animal.position.x, animal.position.y, animal.diet, animal.family
    ));
    commands.spawn(animal);
}

fn handle_manual_zoo_spawn(
    mut events: MessageReader<ManualZooSpawnEvent>,
    mut commands: Commands,
    frame_count: Res<GlobalFrameCounter>,
    config: Res<SimulationConfig>,
    mut log: ResMut<SimulationLogger>,
    mut rng: ResMut<SimulationRng>,
) {
    for _ in events.read() {
        let animal = Animal::new(
            rng.0.next_u64(),
            None,
            Diet::random(&mut rng.0),
            Vec2::new(
                rng.0
                    .gen_range(-config.world_bounds.half_width..config.world_bounds.half_width),
                rng.0
                    .gen_range(-config.world_bounds.half_height..config.world_bounds.half_height),
            ),
            Vec2::ZERO,
            Genome::random(&mut rng.0),
            &frame_count,
            &config,
            rng.0.next_u32(),
            0,
        );
        log.info(&format!(
            "animal_spawn source=manual_random x={:.2} y={:.2} diet={:?} family={}",
            animal.position.x, animal.position.y, animal.diet, animal.family
        ));
        commands.spawn(animal);
    }
}

fn sample_spawn_delay(rate_per_sec: f32, rng: &mut impl Rng) -> f32 {
    let uniform = rng.gen_range(f32::EPSILON..1.0);
    -uniform.ln() / rate_per_sec
}

fn think_animals(
    mut animals: Query<&mut Animal>,
    plants: Query<&Plant>,
    carcasses: Query<&Carcass>,
    config: Res<SimulationConfig>,
) {
    let plants_snapshot: Vec<PlantSnapshot> = plants
        .iter()
        .map(|plant| PlantSnapshot {
            position: plant.position,
            energy: plant.energy,
        })
        .collect();

    let animals_snapshot: Vec<AnimalSnapshot> = animals
        .iter()
        .map(|animal| AnimalSnapshot {
            diet: animal.diet,
            position: animal.position,
            velocity: animal.velocity,
            energy: animal.energy,
        })
        .collect();

    let carcasses_snapshot: Vec<CarcassSnapshot> = carcasses
        .iter()
        .map(|carcass| CarcassSnapshot {
            position: carcass.position,
            energy: carcass.energy,
        })
        .collect();

    let world = PerceptionWorld {
        plants: &plants_snapshot,
        animals: &animals_snapshot,
        carcasses: &carcasses_snapshot,
    };

    for mut animal in &mut animals {
        let movement = think_with_vision(
            &animal.vision,
            &animal.genome,
            animal.position,
            animal.velocity,
            animal.energy,
            &world,
        );

        let new_velocity = movement * config.tuning.animal_max_speed.max(0.0);
        animal.set_velocity(new_velocity);
    }
}

fn calculate_energy_drain(animal: &Animal, config: &SimulationConfig, delta_secs: f32) -> f32 {
    let speed_ratio = (animal.velocity().length() / config.tuning.animal_max_speed).max(0.0);
    let speed_drain = config.tuning.animal_speed_energy_drain_per_sec * speed_ratio;
    let brain_drain = calculate_brain_drain(&animal.genome, config);
    (config.tuning.animal_base_energy_drain_per_sec + speed_drain + brain_drain) * delta_secs
}

fn calculate_brain_drain(genome: &Genome, config: &SimulationConfig) -> f32 {
    let gene_count = genome.genes.len();
    if gene_count == 0 {
        return 0.0;
    }

    let mean_abs_gene = genome.genes.iter().map(|gene| gene.abs()).sum::<f32>() / gene_count as f32;
    mean_abs_gene * config.tuning.animal_brain_energy_drain_factor.max(0.0)
}

fn move_animals(
    mut query: Query<(&mut Animal, &mut Transform)>,
    time: Res<Time>,
    config: Res<SimulationConfig>,
    mut log: ResMut<SimulationLogger>,
) {
    for (mut animal, mut transform) in &mut query {
        let previous_translation = transform.translation;
        let velocity = animal.velocity();

        if !velocity.is_finite() || !animal.position.is_finite() || !animal.energy.is_finite() {
            log.warn(&format!(
                "animal_state_invalid reason=non_finite_before_move x={:.3} y={:.3} vx={:.3} vy={:.3} energy={:.3}",
                animal.position.x,
                animal.position.y,
                velocity.x,
                velocity.y,
                animal.energy
            ));
            animal.velocity = Vec2::ZERO;
            if !animal.position.is_finite() {
                animal.position = Vec2::ZERO;
            }
            if !animal.energy.is_finite() {
                animal.energy = 0.0;
            }
            transform.translation = animal.position.extend(0.0);
        }

        transform.translation += animal.velocity().extend(0.0) * time.delta_secs();
        ensure_torodial_world(&mut transform.translation, &config.world_bounds);

        if !transform.translation.is_finite() {
            log.warn(&format!(
                "animal_state_invalid reason=non_finite_translation_after_move prev_x={:.3} prev_y={:.3} vx={:.3} vy={:.3}",
                previous_translation.x,
                previous_translation.y,
                animal.velocity.x,
                animal.velocity.y,
            ));
            transform.translation = previous_translation;
            animal.velocity = Vec2::ZERO;
        }

        animal.set_position(transform.translation.xy());

        let energy_drain = calculate_energy_drain(&animal, &config, time.delta_secs());
        let updated_energy = (animal.energy() - energy_drain).max(0.0);
        animal.set_energy(updated_energy);
        if !animal.energy.is_finite() {
            log.warn("animal_state_invalid reason=non_finite_energy_after_drain");
            animal.energy = 0.0;
        }
        animal.size = size_from_energy(animal.energy, &config);
    }
}

fn grow_plants(
    mut commands: Commands,
    mut plants: Query<&mut Plant>,
    time: Res<Time>,
    config: Res<SimulationConfig>,
    mut rng: ResMut<SimulationRng>,
    mut log: ResMut<SimulationLogger>,
) {
    let growth = config.tuning.plant_growth_per_sec * time.delta_secs();
    if growth <= 0.0 {
        return;
    }

    let mut spawn_positions = Vec::new();
    for mut plant in &mut plants {
        let was_below_max = plant.energy < config.tuning.plant_max_energy;
        plant.energy = (plant.energy + growth).min(config.tuning.plant_max_energy);
        plant.size = size_from_energy(plant.energy, &config);

        if was_below_max && plant.energy >= config.tuning.plant_max_energy {
            spawn_positions.push(plant.position);
        }
    }

    for parent_position in spawn_positions {
        spawn_plant_nearby(
            &mut commands,
            &config,
            &mut rng.0,
            parent_position,
            &mut *log,
        );
    }
}

fn reproduce_animals(
    mut commands: Commands,
    mut animals: Query<&mut Animal>,
    frame_count: Res<GlobalFrameCounter>,
    mut log: ResMut<SimulationLogger>,
    config: Res<SimulationConfig>,
    mut rng: ResMut<SimulationRng>,
) {
    let spawn_energy = config.spawn_config.animal_spawn_energy;
    let threshold = spawn_energy * config.tuning.reproduction_energy_multiplier;
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
            let child_energy = spawn_energy;
            offspring_energy_total += child_energy;

            let mut child_position = parent.position
                + Vec2::new(
                    rng.0.gen_range(-position_jitter..position_jitter),
                    rng.0.gen_range(-position_jitter..position_jitter),
                );
            let mut child_translation = child_position.extend(0.0);
            ensure_torodial_world(&mut child_translation, &config.world_bounds);
            child_position = child_translation.xy();

            let child_velocity = Vec2::new(0.0, 0.0);

            offspring.push(Animal::new(
                rng.0.next_u64(),
                Some(parent.id),
                parent.diet,
                child_position,
                child_velocity,
                parent.genome.mutated(&mut rng.0, mutation_strength),
                &frame_count,
                &config,
                parent.family,
                parent.generation.saturating_add(1),
            ));
        }

        parent.energy = (parent.energy - offspring_energy_total - 2.0 * spawn_energy).max(0.0);
        parent.size = size_from_energy(parent.energy, &config);
    }

    for child in offspring {
        commands.spawn(child);
    }

    if reproduction_count > 0 {
        log.debug(&format!(
            "animal_reproduction parents={} offspring={}",
            reproduction_count,
            reproduction_count * 2
        ));
    }
}

fn despawn_starved_animals(
    mut commands: Commands,
    frame_count: Res<GlobalFrameCounter>,
    config: Res<SimulationConfig>,
    mut animals: Query<(Entity, &mut Animal)>,
    mut log: ResMut<SimulationLogger>,
    mut zoo: ResMut<Zoo>,
) {
    let mut despawn_count = 0usize;
    for (entity, mut animal) in &mut animals {
        if animal.energy <= 0.0 {
            despawn_animal(
                &mut commands,
                &mut log,
                &mut zoo,
                entity,
                &mut animal,
                "starvation",
                frame_count.0,
                &config,
            );
            despawn_count += 1;
        }
    }

    if despawn_count > 0 {
        log.debug(&format!(
            "animal_starvation_despawn count={}",
            despawn_count
        ));
    }
}

fn log_removed_animals(
    mut removed_animals: RemovedComponents<Animal>,
    mut log: ResMut<SimulationLogger>,
) {
    let mut removed_count = 0usize;
    for _ in removed_animals.read() {
        removed_count += 1;
    }

    if removed_count > 0 {
        log.debug(&format!("animal_removed count={}", removed_count));
    }
}

fn handle_object_collision(
    mut commands: Commands,
    config: Res<SimulationConfig>,
    mut log: ResMut<SimulationLogger>,
    frame_count: Res<GlobalFrameCounter>,
    mut zoo: ResMut<Zoo>,
    mut entities: ParamSet<(
        Query<(Entity, &Animal)>,
        Query<&mut Animal>,
        Query<(Entity, &Plant)>,
        Query<&mut Plant>,
        Query<(Entity, &Carcass)>,
        Query<&mut Carcass>,
    )>,
) {
    #[derive(Clone, Copy)]
    struct AnimalFoodSnapshot {
        entity: Entity,
        id: u64,
        parent_id: Option<u64>,
        position: Vec2,
        radius: f32,
        energy: f32,
        diet: Diet,
    }

    #[derive(Clone, Copy)]
    struct PlantFoodSnapshot {
        entity: Entity,
        position: Vec2,
        radius: f32,
        energy: f32,
    }

    let animals_snapshot = entities
        .p0()
        .iter()
        .map(|(entity, animal)| AnimalFoodSnapshot {
            entity,
            id: animal.id,
            parent_id: animal.parent_id,
            position: animal.position,
            radius: animal.size,
            energy: animal.energy,
            diet: animal.diet,
        })
        .collect::<Vec<_>>();
    let plants_snapshot = entities
        .p2()
        .iter()
        .map(|(entity, plant)| PlantFoodSnapshot {
            entity,
            position: plant.position,
            radius: plant.size,
            energy: plant.energy,
        })
        .collect::<Vec<_>>();

    #[derive(Clone, Copy)]
    struct CarcassFoodSnapshot {
        entity: Entity,
        position: Vec2,
        radius: f32,
        energy: f32,
    }

    let carcasses_snapshot = entities
        .p4()
        .iter()
        .map(|(entity, carcass)| CarcassFoodSnapshot {
            entity,
            position: carcass.position,
            radius: carcass.size,
            energy: carcass.energy,
        })
        .collect::<Vec<_>>();

    if animals_snapshot.is_empty() {
        return;
    }

    let consume_per_collision = config.tuning.plant_consume_per_collision.max(0.0);
    assert!(
        consume_per_collision > 0.0,
        "plant_consume_per_collision must be greater than 0.0"
    );

    let mut animal_gain_by_entity: HashMap<Entity, f32> = HashMap::new();
    let mut plant_taken_by_entity: HashMap<Entity, f32> = HashMap::new();
    let mut prey_taken_by_entity: HashMap<Entity, f32> = HashMap::new();
    let mut carcass_taken_by_entity: HashMap<Entity, f32> = HashMap::new();

    if !plants_snapshot.is_empty() {
        for predator in &animals_snapshot {
            if !predator.diet.can_eat_plants() {
                continue;
            }

            let metabolism_ratio = predator.diet.metabolism_ratio(&config);

            for plant in &plants_snapshot {
                let combined_radius = predator.radius + plant.radius;
                let overlaps = predator.position.distance_squared(plant.position)
                    <= combined_radius * combined_radius;
                if !overlaps {
                    continue;
                }

                let already_taken = plant_taken_by_entity
                    .get(&plant.entity)
                    .copied()
                    .unwrap_or(0.0);
                let remaining_energy = (plant.energy - already_taken).max(0.0);
                if remaining_energy <= 0.0 {
                    continue;
                }

                let taken = consume_per_collision.min(remaining_energy);
                if taken <= 0.0 {
                    continue;
                }

                *animal_gain_by_entity.entry(predator.entity).or_insert(0.0) +=
                    taken * metabolism_ratio;
                *plant_taken_by_entity.entry(plant.entity).or_insert(0.0) += taken;
            }
        }
    }

    if !carcasses_snapshot.is_empty() {
        for scavenger in &animals_snapshot {
            if !scavenger.diet.can_eat_carcasses() {
                continue;
            }

            let metabolism_ratio = scavenger.diet.metabolism_ratio(&config);

            for carcass in &carcasses_snapshot {
                let combined_radius = scavenger.radius + carcass.radius;
                let overlaps = scavenger.position.distance_squared(carcass.position)
                    <= combined_radius * combined_radius;
                if !overlaps {
                    continue;
                }

                let already_taken = carcass_taken_by_entity
                    .get(&carcass.entity)
                    .copied()
                    .unwrap_or(0.0);
                let remaining_energy = (carcass.energy - already_taken).max(0.0);
                if remaining_energy <= 0.0 {
                    continue;
                }

                let taken = consume_per_collision.min(remaining_energy);
                if taken <= 0.0 {
                    continue;
                }

                *animal_gain_by_entity.entry(scavenger.entity).or_insert(0.0) +=
                    taken * metabolism_ratio;
                *carcass_taken_by_entity.entry(carcass.entity).or_insert(0.0) += taken;
            }
        }
    }

    for predator in &animals_snapshot {
        if !predator.diet.can_eat_animals() {
            continue;
        }

        let metabolism_ratio = predator.diet.metabolism_ratio(&config);

        for prey in &animals_snapshot {
            if predator.entity == prey.entity {
                continue;
            }

            if prey.parent_id == Some(predator.id) {
                continue;
            }

            // if matches!(prey.diet, Diet::Carnivore) && predator.family == prey.family {
            //     continue;
            // }

            let can_predate = match predator.diet {
                Diet::Herbivore | Diet::Scavenger => false,
                Diet::Omnivore => {
                    matches!(
                        prey.diet,
                        Diet::Herbivore | Diet::Omnivore | Diet::Carnivore | Diet::Scavenger
                    )
                }
                Diet::Carnivore => {
                    matches!(
                        prey.diet,
                        Diet::Herbivore | Diet::Omnivore | Diet::Carnivore | Diet::Scavenger
                    )
                }
            };
            if !can_predate {
                continue;
            }

            // If both animals can predate each other, choose a single winner for this
            // frame so they do not both consume and immediately cancel out.
            let prey_can_counter_predate = match prey.diet {
                Diet::Herbivore | Diet::Scavenger => false,
                Diet::Omnivore => {
                    matches!(
                        predator.diet,
                        Diet::Herbivore | Diet::Omnivore | Diet::Carnivore | Diet::Scavenger
                    )
                }
                Diet::Carnivore => {
                    matches!(
                        predator.diet,
                        Diet::Herbivore | Diet::Omnivore | Diet::Carnivore | Diet::Scavenger
                    )
                }
            };
            if prey_can_counter_predate {
                let predator_wins_duel = if predator.energy > prey.energy {
                    true
                } else if predator.energy < prey.energy {
                    false
                } else {
                    predator.entity.to_bits() > prey.entity.to_bits()
                };
                if !predator_wins_duel {
                    continue;
                }
            }

            let combined_radius = predator.radius + prey.radius;
            let overlaps = predator.position.distance_squared(prey.position)
                <= combined_radius * combined_radius;
            if !overlaps {
                continue;
            }

            let already_taken = prey_taken_by_entity
                .get(&prey.entity)
                .copied()
                .unwrap_or(0.0);
            let remaining_energy = (prey.energy - already_taken).max(0.0);
            if remaining_energy <= 0.0 {
                continue;
            }

            let taken = consume_per_collision.min(remaining_energy);
            if taken <= 0.0 {
                continue;
            }

            *animal_gain_by_entity.entry(predator.entity).or_insert(0.0) +=
                taken * metabolism_ratio;
            *prey_taken_by_entity.entry(prey.entity).or_insert(0.0) += taken;
        }
    }

    for (animal_entity, gained_energy) in &animal_gain_by_entity {
        if let Ok(mut animal) = entities.p1().get_mut(*animal_entity) {
            animal.energy += *gained_energy;
            animal.size = size_from_energy(animal.energy, &config);
        }
    }

    for (plant_entity, taken_energy) in &plant_taken_by_entity {
        if let Ok(mut plant) = entities.p3().get_mut(*plant_entity) {
            plant.energy = (plant.energy - *taken_energy).max(0.0);
            plant.size = size_from_energy(plant.energy, &config);
            if plant.energy <= 0.0 {
                commands.entity(*plant_entity).despawn();
            }
        }
    }

    for (prey_entity, taken_energy) in &prey_taken_by_entity {
        if let Ok(mut prey) = entities.p1().get_mut(*prey_entity) {
            prey.energy = (prey.energy - *taken_energy).max(0.0);
            prey.size = size_from_energy(prey.energy, &config);
            if prey.energy <= 0.0 {
                despawn_animal(
                    &mut commands,
                    &mut *log,
                    &mut zoo,
                    *prey_entity,
                    &mut prey,
                    "collision",
                    frame_count.0,
                    &config,
                );
            }
        }
    }

    for (carcass_entity, taken_energy) in &carcass_taken_by_entity {
        if let Ok(mut carcass) = entities.p5().get_mut(*carcass_entity) {
            carcass.energy = (carcass.energy - *taken_energy).max(0.0);
            carcass.size = size_from_energy(carcass.energy, &config);
            if carcass.energy <= 0.0 {
                commands.entity(*carcass_entity).despawn();
            }
        }
    }
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
