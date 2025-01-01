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
        let is_plus: bool = rand::random();
        let lattice_value = if is_plus { 1 } else { -1 };
        let lattice = vec![vec![lattice_value; *size]; *size];
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
            if delta_e < 0.0 || (-delta_e * self.beta).exp() > rand::random() {
                self.lattice[i][j] *= -1;
            }
        }
    }

    /// Sweep over the lattice such that all points are flipped.
    pub fn sweep(&mut self) {
        for i in 0..self.size {
            for j in 0..self.size {
                let delta_e = self.compute_delta_energy(i, j);
                if delta_e < 0.0 || (-delta_e * self.beta).exp() > rand::random() {
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
        let exp_mag = int_mean(&magnetizations);
        let exp_magsquare = float_mean(&mag_squares);
        let exp_energy = float_mean(&energies);
        let exp_ensquare = float_mean(&energy_squares);

        let magnetic_susceptibility = (exp_magsquare - exp_mag.powf(2.0)) * self.beta;
        let heat_capacity =
            (exp_ensquare - exp_energy.powf(2.0)) * self.beta.powf(2.0) / (self.size.pow(2) as f64);

        (exp_mag, exp_energy, magnetic_susceptibility, heat_capacity)
    }
}

pub fn main() {
    println!("Ising model...");
    let size = 10;
    let temperature = 2.0;
    let h = 1.0;
    println!("Size: {}, Temperature: {}, h: {}", size, temperature, h);
    let mut model = IsingModel::new(&size, &temperature, &h);
    let (exp_mag, exp_energy, mag_susc, heat_cap) = model.simulate(NUM_SWEEPS);
    println!("Expected magnetization: {}", exp_mag);
    println!("Expected energy: {}", exp_energy);
    println!("Magnetic susceptibility: {}", mag_susc);
    println!("Heat capacity: {}", heat_cap);
}
