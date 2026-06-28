use bevy::prelude::Vec2;
use nalgebra as na;
use rand::Rng;

pub const MLP_INPUTS: usize = 25;
pub const MLP_HIDDEN_1: usize = 12;
pub const MLP_HIDDEN_2: usize = 9;
pub const MLP_OUTPUTS: usize = 2;

pub const GENOME_LEN: usize = MLP_INPUTS * MLP_HIDDEN_1
    + MLP_HIDDEN_1
    + MLP_HIDDEN_1 * MLP_HIDDEN_2
    + MLP_HIDDEN_2
    + MLP_HIDDEN_2 * MLP_OUTPUTS
    + MLP_OUTPUTS;

#[derive(Debug, Clone)]
pub struct Genome {
    pub genes: Vec<f32>,
}

#[allow(dead_code)]
impl Genome {
    pub fn random(rng: &mut impl Rng) -> Self {
        let genes = (0..GENOME_LEN).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Self { genes }
    }

    pub fn mutated(&self, rng: &mut impl Rng, strength: f32) -> Self {
        if strength <= 0.0 {
            return self.clone();
        }

        let genes = self
            .genes
            .iter()
            .map(|gene| *gene + rng.gen_range(-strength..strength))
            .collect();
        Self { genes }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MovementOutput {
    pub vector: Vec2,
}

pub fn mlp_movement(features: [f32; MLP_INPUTS], genome: &Genome) -> MovementOutput {
    assert_eq!(genome.genes.len(), GENOME_LEN, "Genome length mismatch");

    // x: 1xMLP_INPUTS
    let x: na::RowSVector<f32, MLP_INPUTS> = na::RowSVector::from_row_slice(&features);

    // W1: MLP_INPUTS x MLP_HIDDEN_1
    let w1: na::SMatrix<f32, MLP_INPUTS, MLP_HIDDEN_1> =
        na::SMatrix::from_row_slice(&genome.genes[0..(MLP_INPUTS * MLP_HIDDEN_1)]);

    // b1: 1xMLP_HIDDEN_1
    let b1_start = MLP_INPUTS * MLP_HIDDEN_1;
    let b1: na::RowSVector<f32, MLP_HIDDEN_1> =
        na::RowSVector::from_row_slice(&genome.genes[b1_start..b1_start + MLP_HIDDEN_1]);

    // hidden_1: 1xMLP_HIDDEN_1
    let hidden_1 = x * w1 + b1;
    let hidden_1_activated = hidden_1.map(|v| v.tanh());

    // W2: MLP_HIDDEN_1 x MLP_HIDDEN_2
    let w2_start = b1_start + MLP_HIDDEN_1;
    let w2: na::SMatrix<f32, MLP_HIDDEN_1, MLP_HIDDEN_2> = na::SMatrix::from_row_slice(
        &genome.genes[w2_start..w2_start + MLP_HIDDEN_1 * MLP_HIDDEN_2],
    );

    // b2: 1xMLP_HIDDEN_2
    let b2_start = w2_start + MLP_HIDDEN_1 * MLP_HIDDEN_2;
    let b2: na::RowSVector<f32, MLP_HIDDEN_2> =
        na::RowSVector::from_row_slice(&genome.genes[b2_start..b2_start + MLP_HIDDEN_2]);

    // hidden_2: 1xMLP_HIDDEN_2
    let hidden_2 = hidden_1_activated * w2 + b2;
    let hidden_2_activated = hidden_2.map(|v| v.tanh());

    // W3: MLP_HIDDEN_2 x MLP_OUTPUTS
    let w3_start = b2_start + MLP_HIDDEN_2;
    let w3: na::SMatrix<f32, MLP_HIDDEN_2, MLP_OUTPUTS> =
        na::SMatrix::from_row_slice(&genome.genes[w3_start..w3_start + MLP_HIDDEN_2 * MLP_OUTPUTS]);

    // b3: 1xMLP_OUTPUTS
    let b3_start = w3_start + MLP_HIDDEN_2 * MLP_OUTPUTS;
    let b3: na::RowSVector<f32, MLP_OUTPUTS> =
        na::RowSVector::from_row_slice(&genome.genes[b3_start..b3_start + MLP_OUTPUTS]);

    // y: 1xMLP_OUTPUTS
    let y = hidden_2_activated * w3 + b3;

    MovementOutput {
        vector: Vec2::new(y[0], y[1]),
    }
}
