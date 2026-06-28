use bevy::prelude::Resource;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::creature::Diet;
use crate::logging::SimulationLogger;
use crate::mlp::Genome;

const ZOO_CAPACITY: usize = 10;
const DEFAULT_ZOO_PATH: &str = "logs/zoo.json";

fn sort_and_truncate_by_lifetime(entries: &mut Vec<ZooAnimal>) {
    entries.sort_by(|left, right| right.lifetime_frames.cmp(&left.lifetime_frames));
    entries.truncate(ZOO_CAPACITY);
}

fn retain_top_per_diet(entries: &mut Vec<ZooAnimal>) {
    let mut herbivores = Vec::new();
    let mut omnivores = Vec::new();
    let mut carnivores = Vec::new();
    let mut scavengers = Vec::new();

    for entry in std::mem::take(entries) {
        match entry.diet {
            Diet::Herbivore => herbivores.push(entry),
            Diet::Omnivore => omnivores.push(entry),
            Diet::Carnivore => carnivores.push(entry),
            Diet::Scavenger => scavengers.push(entry),
        }
    }

    sort_and_truncate_by_lifetime(&mut herbivores);
    sort_and_truncate_by_lifetime(&mut omnivores);
    sort_and_truncate_by_lifetime(&mut carnivores);
    sort_and_truncate_by_lifetime(&mut scavengers);

    entries.extend(herbivores);
    entries.extend(omnivores);
    entries.extend(carnivores);
    entries.extend(scavengers);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZooAnimal {
    pub lifetime_frames: u64,
    pub diet: Diet,
    pub genome: Genome,
    pub family: u32,
}

#[derive(Resource, Debug)]
pub struct Zoo {
    entries: Vec<ZooAnimal>,
    file_path: PathBuf,
}

impl Zoo {
    pub fn load_default(log: &mut SimulationLogger) -> Self {
        Self::load_from_path(DEFAULT_ZOO_PATH, log)
    }

    pub fn load_from_path(path: impl AsRef<Path>, log: &mut SimulationLogger) -> Self {
        let file_path = path.as_ref().to_path_buf();
        log.info(&format!("zoo_load_started path={}", file_path.display()));

        let entries = match fs::read_to_string(&file_path) {
            Ok(raw) => match serde_json::from_str::<Vec<ZooAnimal>>(&raw) {
                Ok(mut parsed) => {
                    retain_top_per_diet(&mut parsed);
                    log.info(&format!(
                        "zoo_load_completed path={} entries={}",
                        file_path.display(),
                        parsed.len()
                    ));
                    parsed
                }
                Err(err) => {
                    log.warn(&format!(
                        "zoo_load_failed path={} error=invalid_json details={}",
                        file_path.display(),
                        err
                    ));
                    Vec::new()
                }
            },
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                log.info(&format!(
                    "zoo_load_missing path={} entries=0",
                    file_path.display()
                ));
                Vec::new()
            }
            Err(err) => {
                log.warn(&format!(
                    "zoo_load_failed path={} error={} entries=0",
                    file_path.display(),
                    err
                ));
                Vec::new()
            }
        };

        Self { entries, file_path }
    }

    pub fn maybe_sample<'a>(
        &'a self,
        rng: &mut impl Rng,
        probability: f32,
    ) -> Option<&'a ZooAnimal> {
        if self.entries.is_empty() {
            return None;
        }

        let chance = probability.clamp(0.0, 1.0) as f64;
        if !rng.gen_bool(chance) {
            return None;
        }

        let index = rng.gen_range(0..self.entries.len());
        self.entries.get(index)
    }

    pub fn consider_and_persist(&mut self, candidate: ZooAnimal, log: &mut SimulationLogger) {
        self.entries.push(candidate);
        retain_top_per_diet(&mut self.entries);
        if let Err(err) = self.persist() {
            log.warn(&format!(
                "zoo_save_failed path={} error={}",
                self.file_path.display(),
                err
            ));
            return;
        }

        log.info(&format!(
            "zoo_save_completed path={} entries={}",
            self.file_path.display(),
            self.entries.len()
        ));
    }

    fn persist(&self) -> std::io::Result<()> {
        if let Some(parent) = self.file_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let content =
            serde_json::to_string_pretty(&self.entries).unwrap_or_else(|_| "[]".to_string());

        fs::write(&self.file_path, content)
    }
}
