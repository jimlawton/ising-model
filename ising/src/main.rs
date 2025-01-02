use std::collections::HashMap;
use std::fmt::Display;

use rand::Rng; // Need the Rng trait.

const NUM_SWEEPS: usize = 15000;

/// Calculate the mean of an integer array.
fn int_mean(array: &[i32]) -> f64 {
    let sum: i32 = array.iter().sum();
    sum as f64 / (array.len() as f64)
}

/// Calculate the mean of a float array.
fn float_mean(array: &[f64]) -> f64 {
    let sum: f64 = array.iter().sum();
    sum / (array.len() as f64)
}

/// Calculate the index of the previous element in a circular array.
fn decrement_index(index: usize, size: usize) -> usize {
    (index + size - 1) % size
}

/// Calculate the index of the next element in a circular array.
fn increment_index(index: usize, size: usize) -> usize {
    (index + 1) % size
}

pub struct IsingModel {
    pub size: usize,
    pub temperature: f64,
    pub h: f64,
    lattice: Vec<Vec<i32>>,
    beta: f64,
}

impl IsingModel {
    /// Constructor.
    // NOTE: assumes a square lattice matrix.
    pub fn new(size: &usize, temperature: &f64, h: &f64) -> Self {
        if *size == 0 {
            panic!("ERROR: size must be greater than 0!");
        }
        if *temperature == 0.0 {
            panic!("ERROR: temperature must be greater than 0 K!");
        }
        let mut lattice: Vec<Vec<i32>> = vec![vec![1; *size]; *size];
        for row in &mut lattice {
            for col in row {
                *col = if rand::random::<bool>() { 1 } else { -1 }
            }
        }
        let beta = 1.0 / temperature;
        IsingModel {
            size: *size,
            temperature: *temperature,
            h: *h,
            lattice,
            beta,
        }
    }

    /// Calculate the change in energy if the spin at (i, j) is flipped.
    pub fn compute_delta_energy(&self, i: usize, j: usize) -> f64 {
        let spin = self.lattice[i][j];
        let i_minus_1 = decrement_index(i, self.size);
        let j_minus_1 = decrement_index(j, self.size);
        let i_plus_1 = increment_index(i, self.size);
        let j_plus_1 = increment_index(j, self.size);
        let neighbors: i32 = self.lattice[i_plus_1][j]
            + self.lattice[i][j_plus_1]
            + self.lattice[i_minus_1][j]
            + self.lattice[i][j_minus_1];
        (2 * spin * neighbors) as f64 + 2.0 * self.h * (spin as f64)
    }

    /// Calculate the magnetization of the lattice.
    pub fn magnetization(&self) -> i32 {
        self.lattice.iter().flatten().sum()
    }

    /// Calculate the energy of the lattice.
    pub fn energy(&self) -> f64 {
        let mut energy = 0.0;
        for i in 0..self.size {
            for j in 0..self.size {
                let spin = self.lattice[i][j];
                let neighbors =
                    self.lattice[(i + 1) % self.size][j] + self.lattice[i][(j + 1) % self.size];
                energy -= (spin * neighbors) as f64 + self.h * (spin as f64);
            }
        }
        energy
    }

    // The Metropolis algorithm here is used to minimise the energy, as it accepts any change lowering energy,
    // and accepts those increasing energy randomly with probability of acceptance depending on a function that
    // is defined as `e^(-delta_E / (k_B * T))`, where k_B is Boltzmann's constant.

    /// Perform a single Metropolis step.
    pub fn step(&mut self) {
        for _ in 0..self.size.pow(2) {
            let i: usize = rand::thread_rng().gen_range(0..self.size);
            let j: usize = rand::thread_rng().gen_range(0..self.size);
            let delta_e = self.compute_delta_energy(i, j);
            if delta_e < 0.0 || (-delta_e * self.beta).exp() > rand::random::<f64>() {
                self.lattice[i][j] *= -1;
            }
        }
    }

    /// Sweep over the lattice such that all points are flipped.
    pub fn sweep(&mut self) {
        for i in 0..self.size {
            for j in 0..self.size {
                let delta_e = self.compute_delta_energy(i, j);
                if delta_e < 0.0 || (-delta_e * self.beta).exp() > rand::random::<f64>() {
                    self.lattice[i][j] *= -1;
                }
            }
        }
    }

    /// Run the simulated model for a number of sweeps, and store average magnetization and energy.
    pub fn simulate(&mut self, num_sweeps: usize) -> (f64, f64, f64, f64) {
        let mut magnetizations = Vec::<i32>::new();
        let mut mag_squares = Vec::<f64>::new();
        let mut energies = Vec::<f64>::new();
        let mut energy_squares = Vec::<f64>::new();
        for _ in 0..num_sweeps {
            self.sweep();
            let mag = self.magnetization();
            let energy = self.energy();
            magnetizations.append(&mut vec![mag]);
            mag_squares.append(&mut vec![mag.pow(2) as f64]);
            energies.append(&mut vec![energy]);
            energy_squares.append(&mut vec![energy.powf(2.0)]);
        }
        let mean_mag = int_mean(&magnetizations);
        let mean_mag_sq = float_mean(&mag_squares);
        let mean_energy = float_mean(&energies);
        let mean_energy_sq = float_mean(&energy_squares);
        let susceptibility = (mean_mag_sq - mean_mag.powf(2.0)) * self.beta;
        let heat_capacity = (mean_energy_sq - mean_energy.powf(2.0)) * self.beta.powf(2.0)
            / (self.size.pow(2) as f64);

        (mean_mag, mean_energy, susceptibility, heat_capacity)
    }
}

impl Display for IsingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Size: {}", self.size)?;
        writeln!(f, "Temperature: {} J", self.temperature)?;
        writeln!(f, "h: {}", self.h)?;
        writeln!(f, "Lattice:")?;
        for i in 0..self.size {
            for j in 0..self.size {
                let spin = self.lattice[i][j];
                let symbol = if spin == 1 { "+" } else { "-" };
                write!(f, "{} ", symbol)?;
            }
            writeln!(f)?;
        }
        writeln!(f, "Number of spins: {}", self.size.pow(2))?;
        writeln!(f, "Magnetization: {}", self.magnetization())?;
        writeln!(f, "Energy: {} J", self.energy())?;
        Ok(())
    }
}

pub fn main() {
    println!("Ising model...");
    let size = 10;
    let temperature = 1.0;
    let h = 0.0;
    let mut model = IsingModel::new(&size, &temperature, &h);

    println!("{}", model);

    // let (mean_mag, mean_energy, susceptibility, heat_capacity) = model.simulate(NUM_SWEEPS);
    // println!("Mean magnetization: {}", mean_mag);
    // println!("Mean energy: {}", mean_energy);
    // println!("Magnetic susceptibility: {}", susceptibility);
    // println!("Heat capacity: {}", heat_capacity);

    // Determine how many sweeps were required for convergence.
    let mut sweeps: Vec<usize> = (100..=1000)
        .step_by(100)
        .chain((2000..=10000).step_by(1000))
        .collect();
    sweeps.sort();
    let mut data: HashMap<usize, (f64, f64, f64, f64)> = HashMap::new();
    for &num_sweeps in &sweeps {
        println!("Simulating with {num_sweeps} sweeps...");
        let (mean_mag, mean_energy, susceptibility, heat_capacity) = model.simulate(num_sweeps);
        data.insert(
            num_sweeps,
            (mean_mag, mean_energy, susceptibility, heat_capacity),
        );
    }
    println!("Output data: {:?}", data);
}
