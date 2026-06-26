#[path = "../src/mlp.rs"]
mod mlp;

use mlp::{GENOME_LEN, Genome, MLP_INPUTS, mlp_movement};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[test]
fn mlp_movement_random_finite_inputs_produce_finite_output() {
    let mut rng = StdRng::seed_from_u64(0xDEC0DEDu64);

    for _ in 0..100 {
        let mut features = [0.0f32; MLP_INPUTS];
        for feature in &mut features {
            *feature = rng.gen_range(-10_000.0..10_000.0);
        }

        let genes = (0..GENOME_LEN)
            .map(|_| rng.gen_range(-10_000.0..10_000.0))
            .collect();
        let genome = Genome { genes };

        let movement = mlp_movement(features, &genome);
        assert!(
            movement.vector.x.is_finite() && movement.vector.y.is_finite(),
            "non-finite output for features={:?}, genes={:?}, output={:?}",
            features,
            genome.genes,
            movement.vector
        );
    }
}
