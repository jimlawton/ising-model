use rand::Rng; // Need the Rng trait.

const NUM_SWEEPS: usize = 1500;

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

pub struct IsingModel {
    size: usize,
    temperature: f64,
    lattice: Vec<Vec<i32>>,
    beta: f64,
    h: f64,
}

impl IsingModel {
    /// Constructor.
    // NOTE: assumes a square lattice matrix.
    pub fn new(size: usize, temperature: f64, h: f64) -> Self {
        let is_plus: bool = rand::random();
        let lattice_value = if is_plus { 1 } else { -1 };
        let lattice = vec![vec![lattice_value; size]; size];
        let beta = 1.0 / temperature;
        IsingModel {
            size,
            temperature,
            lattice,
            beta,
            h,
        }
    }

    /// Calculate the change in energy if the spin at (i, j) is flipped.
    pub fn compute_delta_energy(&self, i: usize, j: usize) -> f64 {
        let spin = self.lattice[i][j];
        let neighbors: i32 = self.lattice[(i + 1) % self.size][j]
            + self.lattice[i][(j + 1) % self.size]
            + self.lattice[(i - 1) % self.size][j]
            + self.lattice[i][(j - 1) % self.size];
        2 * spin * neighbors + 2.0 * self.h * (spin as f64)
    }

    /// Calculate the magnetization of the lattice.
    pub fn magnetization(&self) -> i32 {
        self.lattice.iter().flatten().sum()
    }

    /// Calculate the energy of the lattice.
    pub fn energy(&self) -> f64 {
        let mut energy = 0.0;
        for i in (0..self.size) {
            for j in (0..self.size) {
                let spin = self.lattice[i][j];
                let neighbors =
                    self.lattice[(i + 1) % self.size][j] + self.lattice[i][(j + 1) % self.size];
                energy -= spin * neighbors + self.h * spin;
            }
        }
        energy
    }

    // The Metropolis algorithm here is used to minimise the energy, as it accepts any change lowering energy,
    // and accepts those increasing energy randomly with probability of acceptance depending on a function that
    // is defined as `e^(-delta_E / (k_B * T))`, where k_B is Boltzmann's constant.

    /// Perform a single Metropolis step.
    pub fn step(&mut self) {
        for _ in (0..self.size.pow(2)) {
            let i: usize = rand::thread_rng().gen_range(0..self.size);
            let j: usize = rand::thread_rng().gen_range(0..self.size);
            let delta_e = self.compute_delta_energy(i, j);
            if delta_e < 0.0 || (-delta_e * self.beta).exp() > rand::random() {
                self.lattice[i][j] *= -1;
            }
        }
    }

    /// Sweep over the lattice such that all points are flipped.
    pub fn sweep(&mut self) {
        for i in (0..self.size) {
            for j in (0..self.size) {
                let delta_e = self.compute_delta_energy(i, j);
                if delta_e < 0.0 || (-delta_e * self.beta).exp() > rand::random() {
                    self.lattice[i][j] *= -1;
                }
            }
        }
    }

    /// Run the simulated model for a number of sweeps, and store average magnetization and energy.
    pub fn simulate(&self, num_sweeps: usize) -> (f64, f64, f64, f64) {
        let magnetizations = Vec::<i32>::new();
        let mag_squares = Vec::<f64>::new();
        let energies = Vec::<f64>::new();
        let energy_squares = Vec::<f64>::new();
        for _ in (0..num_sweeps) {
            self.sweep();
            let mag = self.magnetization();
            let energy = self.energy();
            magnetizations.append(&mut vec![mag]);
            mag_squares.append(&mut vec![mag.pow(2) as f64]);
            energies.append(&mut vec![energy]);
            energy_squares.append(&mut vec![energy.powf(2.0)]);
        }
        let exp_mag = int_mean(&magnetizations);
        let exp_magsquare = float_mean(&mag_squares);
        let exp_energy = float_mean(&energies);
        let exp_ensquare = float_mean(&energy_squares);

        let magnetic_susceptibility = (exp_magsquare - exp_mag.pow(2)) * self.beta;
        let heat_capacity =
            (exp_ensquare - exp_energy.pow(2)) * self.beta.powf(2.0) / self.size.pow(2);

        (exp_mag, exp_energy, magnetic_susceptibility, heat_capacity)
    }
}

pub fn main() {
    println!("Hello, world!");
    let model = IsingModel::new(10, 2.0, 1.0);
    let (exp_mag, exp_energy, mag_susc, heat_cap) = model.simulate(NUM_SWEEPS);
    println!("Expected magnetization: {}", exp_mag);
    println!("Expected energy: {}", exp_energy);
    println!("Magnetic susceptibility: {}", mag_susc);
    println!("Heat capacity: {}", heat_cap);
}
