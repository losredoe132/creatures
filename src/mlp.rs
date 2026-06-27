use bevy::prelude::Vec2;
use nalgebra as na;
use rand::Rng;

pub const MLP_INPUTS: usize = 19;
pub const MLP_HIDDEN: usize = 9;
pub const MLP_OUTPUTS: usize = 2;

pub const GENOME_LEN: usize =
    MLP_INPUTS * MLP_HIDDEN + MLP_HIDDEN + MLP_HIDDEN * MLP_OUTPUTS + MLP_OUTPUTS;

#[derive(Debug, Clone)]
pub struct Genome {
    pub genes: Vec<f32>,
}

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

    // x: 1x3
    let x: na::RowSVector<f32, MLP_INPUTS> = na::RowSVector::from_row_slice(&features);

    // W1: 9x9 (row-major slice)
    let w1: na::SMatrix<f32, MLP_INPUTS, MLP_HIDDEN> =
        na::SMatrix::from_row_slice(&genome.genes[0..(MLP_INPUTS * MLP_HIDDEN)]);

    // b1: 1x9
    let b1_start = MLP_INPUTS * MLP_HIDDEN;
    let b1: na::RowSVector<f32, MLP_HIDDEN> =
        na::RowSVector::from_row_slice(&genome.genes[b1_start..b1_start + MLP_HIDDEN]);

    // hidden: 1x9
    let hidden = (x * w1 + b1).map(|v| v.max(0.0)); // ReLU activation

    // W2: 9x2 (row-major slice)
    let w2_start = b1_start + MLP_HIDDEN;
    let w2: na::SMatrix<f32, MLP_HIDDEN, MLP_OUTPUTS> =
        na::SMatrix::from_row_slice(&genome.genes[w2_start..w2_start + MLP_HIDDEN * MLP_OUTPUTS]);

    // b2: 1x2
    let b2_start = w2_start + MLP_HIDDEN * MLP_OUTPUTS;
    let b2: na::RowSVector<f32, MLP_OUTPUTS> =
        na::RowSVector::from_row_slice(&genome.genes[b2_start..b2_start + MLP_OUTPUTS]);

    // y: 1x2
    let y = hidden * w2 + b2*0.1;

    MovementOutput {
        vector: Vec2::new(y[0], y[1]),
    }
}
