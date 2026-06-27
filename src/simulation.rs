use bevy::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::brain::think_with_vision;
use crate::config::{SimulationConfig, WorldBounds};
use crate::creature::{Animal, Diet, EnergyPosition, Movable, Plant};
use crate::logging::{ConsoleBackend, SimulationLogger, TextFileBackend};
use crate::mlp::Genome;
use crate::sense::{AnimalSnapshot, PerceptionWorld, PlantSnapshot};
use crate::utils::size_from_energy;

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

#[derive(Resource, Default, Debug)]
struct AnimalPopulation {
    carnivores: usize,
    herbivores: usize,
    omnivores: usize,
}

#[derive(Resource, Default)]
struct PopulationSizeTracker {
    plants: usize,
    animals: AnimalPopulation,
    initialized: bool,
}

#[derive(Resource)]
struct SimulationRng(StdRng);

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(GlobalFrameCounter::default())
            .insert_resource(PlantSpawnClock::default())
            .insert_resource(AnimalSpawnClock::default())
            .insert_resource(PopulationSizeTracker::default())
            .add_systems(Startup, (initialize_simulation_log, setup_world).chain())
            .add_systems(First, advance_global_frame_counter)
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
                    .chain(),
            )
            .add_systems(PostUpdate, log_population_size_changes);
        app.add_systems(Last, despawn_animals_on_shutdown);
    }
}

fn log_population_size_changes(
    plants: Query<&Plant>,
    animals: Query<&Animal>,
    mut tracker: ResMut<PopulationSizeTracker>,
    mut log: ResMut<SimulationLogger>,
) {
    let plant_count = plants.iter().count();
    let animal_count = animals.iter().count();

    if !tracker.initialized {
        tracker.plants = plant_count;
        tracker.animals.carnivores = 0; // Assuming all animals are carnivores initially
        tracker.animals.herbivores = 0;
        tracker.animals.omnivores = 0;
        tracker.initialized = true;
        return;
    }

    if tracker.animals.carnivores + tracker.animals.herbivores + tracker.animals.omnivores
        != animal_count
    {
        log.info(&format!(
            "population_size plants={} animals={:?}",
            plant_count, tracker.animals
        ));
        tracker.plants = plant_count;
        tracker.animals = AnimalPopulation {
            carnivores: animals
                .iter()
                .filter(|a| matches!(a.diet, Diet::Carnivore))
                .count(),
            herbivores: animals
                .iter()
                .filter(|a| matches!(a.diet, Diet::Herbivore))
                .count(),
            omnivores: animals
                .iter()
                .filter(|a| matches!(a.diet, Diet::Omnivore))
                .count(),
        };
    }
}

fn advance_global_frame_counter(mut frame_counter: ResMut<GlobalFrameCounter>) {
    frame_counter.0 = frame_counter.0.saturating_add(1);
}

fn despawn_animal(
    commands: &mut Commands,
    log: &mut SimulationLogger,
    entity: Entity,
    animal: &mut Animal,
    reason: &str,
    despawn_frame: u64,
) {
    animal.despawn_at = Some(despawn_frame);
    let lifetime_duration = animal.despawn_at.unwrap_or_default() - animal.spawn_at;
    log.info(&format!(
        "animal_despawn,reason={},spawn_at_frame={},despawn_at_frame={},lifetime_frames={},genome={:?},diet={:?}",
        reason,
        animal.spawn_at,
        animal.despawn_at.unwrap_or_default(),
        lifetime_duration,
        animal.genome.genes,
        animal.diet
    ));
    commands.entity(entity).despawn();
}

fn despawn_animals_on_shutdown(
    mut exit_events: MessageReader<AppExit>,
    mut commands: Commands,
    frame_count: Res<GlobalFrameCounter>,
    mut animals: Query<(Entity, &mut Animal)>,
    mut log: ResMut<SimulationLogger>,
) {
    if exit_events.read().next().is_none() {
        return;
    }

    let mut despawn_count = 0usize;
    for (entity, mut animal) in &mut animals {
        despawn_animal(
            &mut commands,
            &mut log,
            entity,
            &mut animal,
            "shutdown",
            frame_count.0,
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

    commands.insert_resource(logger);
    commands.insert_resource(SimulationRng(StdRng::seed_from_u64(seed)));
}

fn setup_world(
    mut commands: Commands,
    frame_count: Res<GlobalFrameCounter>,
    mut log: ResMut<SimulationLogger>,
    config: Res<SimulationConfig>,
    mut rng: ResMut<SimulationRng>,
) {
    let animal = Animal::new(
        Diet::random(&mut rng.0),
        Vec2::new(0.0, 0.0),
        Vec2::new(0.0, 0.0),
        Genome::random(&mut rng.0),
        &frame_count,
        &config,
    );

    for _ in 0..config.spawn_config.n_plants {
        spawn_random_plant(&mut commands, &config, &mut rng.0, "startup", &mut *log);
    }

    commands.spawn(animal);
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
    mut log: ResMut<SimulationLogger>,
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
            &frame_count,
            &mut *log,
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
    let energy = 10.0;
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

fn spawn_random_animal(
    commands: &mut Commands,
    config: &SimulationConfig,
    rng: &mut impl Rng,
    source: &str,
    frame_count: &Res<GlobalFrameCounter>,
    log: &mut SimulationLogger,
) {
    let animal = Animal::new(
        Diet::random(rng),
        Vec2::new(
            rng.gen_range(-config.world_bounds.half_width..config.world_bounds.half_width),
            rng.gen_range(-config.world_bounds.half_height..config.world_bounds.half_height),
        ),
        Vec2::new(rng.gen_range(0.0..100.0), rng.gen_range(0.0..100.0)),
        Genome::random(rng),
        frame_count,
        config,
    );
    log.debug(&format!(
        "animal_spawn source={} x={:.2} y={:.2}, diet ={:?}",
        source, animal.position.x, animal.position.y, animal.diet
    ));
    commands.spawn(animal);
}

fn sample_spawn_delay(rate_per_sec: f32, rng: &mut impl Rng) -> f32 {
    let uniform = rng.gen_range(f32::EPSILON..1.0);
    -uniform.ln() / rate_per_sec
}

fn think_animals(
    mut animals: Query<&mut Animal>,
    plants: Query<&Plant>,
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

    let world = PerceptionWorld {
        plants: &plants_snapshot,
        animals: &animals_snapshot,
    };

    for mut animal in &mut animals {
        let movement = think_with_vision(
            &animal.vision,
            &animal.genome,
            animal.position,
            animal.velocity,
            &world,
        );
        animal.set_velocity(movement * config.tuning.animal_max_speed.max(0.0));
    }
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

        let speed_ratio =
            (animal.velocity().length() / config.tuning.animal_speed_reference).max(0.0);
        let speed_drain = config.tuning.animal_speed_energy_drain_per_sec
            * ((config.tuning.animal_speed_exponent * speed_ratio).exp() - 1.0);
        let energy_drain =
            (config.tuning.animal_base_energy_drain_per_sec + speed_drain) * time.delta_secs();
        let updated_energy = (animal.energy() - energy_drain).max(0.0);
        animal.set_energy(updated_energy);
        if !animal.energy.is_finite() {
            log.warn("animal_state_invalid reason=non_finite_energy_after_drain");
            animal.energy = 0.0;
        }
        animal.size = size_from_energy(animal.energy, &config);
    }
}

fn grow_plants(mut plants: Query<&mut Plant>, time: Res<Time>, config: Res<SimulationConfig>) {
    let growth = config.tuning.plant_growth_per_sec * time.delta_secs();
    if growth <= 0.0 {
        return;
    }

    for mut plant in &mut plants {
        plant.energy = (plant.energy + growth).min(config.tuning.plant_max_energy);
        plant.size = size_from_energy(plant.energy, &config);
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

            let child_velocity =
                parent.velocity + Vec2::new(rng.0.gen_range(-5.0..5.0), rng.0.gen_range(-5.0..5.0));

            offspring.push(Animal::new(
                parent.diet,
                child_position,
                child_velocity,
                parent.genome.mutated(&mut rng.0, mutation_strength),
                &frame_count,
                &config,
            ));
        }

        parent.energy = (parent.energy - offspring_energy_total).max(0.0);
        parent.size = size_from_energy(parent.energy, &config);
    }

    for child in offspring {
        commands.spawn(child);
    }

    if reproduction_count > 0 {
        log.info(&format!(
            "animal_reproduction parents={} offspring={}",
            reproduction_count,
            reproduction_count * 2
        ));
    }
}

fn despawn_starved_animals(
    mut commands: Commands,
    frame_count: Res<GlobalFrameCounter>,
    mut animals: Query<(Entity, &mut Animal)>,
    mut log: ResMut<SimulationLogger>,
) {
    let mut despawn_count = 0usize;
    for (entity, mut animal) in &mut animals {
        if animal.energy <= 0.0 {
            despawn_animal(
                &mut commands,
                &mut log,
                entity,
                &mut animal,
                "starvation",
                frame_count.0,
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
    mut entities: ParamSet<(
        Query<(Entity, &Animal)>,
        Query<&mut Animal>,
        Query<(Entity, &Plant)>,
        Query<&mut Plant>,
    )>,
) {
    #[derive(Clone, Copy)]
    struct AnimalFoodSnapshot {
        entity: Entity,
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

    if animals_snapshot.is_empty() {
        return;
    }

    let consume_per_collision = config.tuning.plant_consume_per_collision.max(0.0);
    if consume_per_collision <= 0.0 {
        return;
    }

    let mut animal_gain_by_entity: HashMap<Entity, f32> = HashMap::new();
    let mut plant_taken_by_entity: HashMap<Entity, f32> = HashMap::new();
    let mut prey_taken_by_entity: HashMap<Entity, f32> = HashMap::new();
    let mut collision_count = 0usize;
    let mut plant_collision_count = 0usize;
    let mut animal_collision_count = 0usize;

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

                collision_count += 1;
                plant_collision_count += 1;
                *animal_gain_by_entity.entry(predator.entity).or_insert(0.0) +=
                    taken * metabolism_ratio;
                *plant_taken_by_entity.entry(plant.entity).or_insert(0.0) += taken;
            }
        }
    }

    for predator in &animals_snapshot {
        if !predator.diet.can_eat_animals() {
            continue;
        }

        let prey_diet_filter = match predator.diet {
            Diet::Omnivore => Some(Diet::Herbivore),
            Diet::Carnivore => None,
            Diet::Herbivore => continue,
        };

        let metabolism_ratio = predator.diet.metabolism_ratio(&config);

        for prey in &animals_snapshot {
            if predator.entity == prey.entity {
                continue;
            }

            if matches!(prey.diet, Diet::Carnivore) {
                continue;
            }

            if let Some(required_diet) = prey_diet_filter {
                if prey.diet != required_diet {
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

            collision_count += 1;
            animal_collision_count += 1;
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

    let mut to_despawn = Vec::new();
    let mut consumed_energy = 0.0;
    let mut depleted_plants = 0usize;
    for (plant_entity, taken_energy) in &plant_taken_by_entity {
        if let Ok(mut plant) = entities.p3().get_mut(*plant_entity) {
            plant.energy = (plant.energy - *taken_energy).max(0.0);
            plant.size = size_from_energy(plant.energy, &config);
            consumed_energy += *taken_energy;
            if plant.energy <= 0.0 {
                depleted_plants += 1;
                to_despawn.push(*plant_entity);
            }
        }
    }

    let mut depleted_prey = 0usize;
    for (prey_entity, taken_energy) in &prey_taken_by_entity {
        if let Ok(mut prey) = entities.p1().get_mut(*prey_entity) {
            prey.energy = (prey.energy - *taken_energy).max(0.0);
            prey.size = size_from_energy(prey.energy, &config);
            consumed_energy += *taken_energy;
            if prey.energy <= 0.0 {
                depleted_prey += 1;
                to_despawn.push(*prey_entity);
            }
        }
    }

    if !to_despawn.is_empty() {
        log.debug(&format!(
            "collision_event type=despawn_plants count={}",
            to_despawn.len()
        ));
    }

    log.debug(&format!(
        "collision_event type=feeding collisions={} plant_collisions={} animal_collisions={} fed_animals={} touched_plants={} touched_prey={} consumed_energy={:.2} depleted_plants={} depleted_prey={}",
        collision_count,
        plant_collision_count,
        animal_collision_count,
        animal_gain_by_entity.len(),
        plant_taken_by_entity.len(),
        prey_taken_by_entity.len(),
        consumed_energy,
        depleted_plants
        ,depleted_prey
    ));

    for entity in to_despawn {
        commands.entity(entity).despawn();
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
