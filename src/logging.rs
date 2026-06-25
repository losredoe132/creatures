use bevy::prelude::*;
use std::fs::{File, OpenOptions, create_dir_all};
use std::io::Write as IoWrite;

/// A sink that receives log messages from [`SimulationLogger`].
pub trait LogBackend: Send + Sync {
    fn write(&mut self, start_timestamp_secs: u64, message: &str);
}

// ---------------------------------------------------------------------------
// Console backend
// ---------------------------------------------------------------------------

/// Emits every message through Bevy's `info!` macro.
pub struct ConsoleBackend;

impl LogBackend for ConsoleBackend {
    fn write(&mut self, _start_timestamp_secs: u64, message: &str) {
        info!("{}", message);
    }
}

// ---------------------------------------------------------------------------
// Plain-text file backend
// ---------------------------------------------------------------------------

/// Appends every message as a line to a `.log` file, prefixed with the
/// simulation start timestamp.
pub struct TextFileBackend {
    file: File,
}

impl TextFileBackend {
    /// Opens (or creates) the file at `path`. Returns `None` if the file or
    /// its parent directory cannot be created.
    pub fn new(path: &str) -> Option<Self> {
        let path_obj = std::path::Path::new(path);
        if let Some(parent) = path_obj.parent() {
            if !parent.as_os_str().is_empty() {
                create_dir_all(parent).ok()?;
            }
        }
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()
            .map(|file| Self { file })
    }
}

impl LogBackend for TextFileBackend {
    fn write(&mut self, start_timestamp_secs: u64, message: &str) {
        let _ = writeln!(
            self.file,
            "[simulation_start_ts={}] {}",
            start_timestamp_secs, message
        );
        let _ = self.file.flush();
    }
}

// ---------------------------------------------------------------------------
// CSV backend (stub — expand as needed)
// ---------------------------------------------------------------------------

/// Placeholder for structured CSV logging. Wire it up by implementing
/// `write` to parse the key=value fields from `message` and append a row.
#[allow(dead_code)]
pub struct CsvBackend {
    file: File,
}

#[allow(dead_code)]
impl CsvBackend {
    pub fn new(path: &str) -> Option<Self> {
        let path_obj = std::path::Path::new(path);
        if let Some(parent) = path_obj.parent() {
            if !parent.as_os_str().is_empty() {
                create_dir_all(parent).ok()?;
            }
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()?;

        // Write header only if the file was just created (empty).
        if file.metadata().ok().map(|m| m.len()).unwrap_or(1) == 0 {
            let _ = writeln!(file, "simulation_start_ts,event,fields");
            let _ = file.flush();
        }

        Some(Self { file })
    }
}

impl LogBackend for CsvBackend {
    fn write(&mut self, start_timestamp_secs: u64, message: &str) {
        // TODO: parse structured key=value fields from `message` and emit a
        //       proper CSV row.  For now, escape commas and write verbatim.
        let escaped = message.replace(',', ";");
        let _ = writeln!(self.file, "{},{}", start_timestamp_secs, escaped);
        let _ = self.file.flush();
    }
}

// ---------------------------------------------------------------------------
// Central logger resource
// ---------------------------------------------------------------------------

/// Bevy resource that fans log messages out to every registered backend.
///
/// Usage:
/// ```rust
/// logger.add_backend(ConsoleBackend);
/// logger.add_backend(TextFileBackend::new("logs/sim.log").unwrap());
/// // later, in any system:
/// logger.log("my_event key=value");
/// ```
#[derive(Resource)]
pub struct SimulationLogger {
    pub start_timestamp_secs: u64,
    backends: Vec<Box<dyn LogBackend>>,
}

impl SimulationLogger {
    pub fn new(start_timestamp_secs: u64) -> Self {
        Self {
            start_timestamp_secs,
            backends: Vec::new(),
        }
    }

    pub fn add_backend(&mut self, backend: impl LogBackend + 'static) {
        self.backends.push(Box::new(backend));
    }

    pub fn log(&mut self, message: &str) {
        for backend in &mut self.backends {
            backend.write(self.start_timestamp_secs, message);
        }
    }
}
