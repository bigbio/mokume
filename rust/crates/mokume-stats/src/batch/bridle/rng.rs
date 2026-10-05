//! NumPy `RandomState`-compatible MT19937 stream.
//!
//! BRIDLE draws random numbers in exactly three places: the 1% monitor hold-out
//! mask, the initial low-rank factors, and the cross-fit fold permutation. They
//! are drawn from a Mersenne Twister seeded like `np.random.RandomState(seed)`
//! so a fixed seed gives the same stream as the Python prototype. This keeps
//! the Rust fit deterministic and lets the golden test compare it cell by cell
//! against the prototype's frozen expected values (generated with its torch
//! initialisation replaced by the same NumPy stream).
//!
//! Implemented draws (bit-identical to NumPy's legacy `RandomState`):
//! `random_sample` (53-bit double), `standard_normal` (polar Box-Muller with a
//! cached second deviate) and `permutation` of a 1-d array (Fisher-Yates driven
//! by the masked-rejection `random_interval`).

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// Mersenne Twister seeded like `np.random.RandomState(seed)`.
#[derive(Clone)]
pub struct NumpyRandomState {
    key: [u32; N],
    pos: usize,
    gauss: Option<f64>,
}

impl NumpyRandomState {
    /// `RandomState(seed)` for an integer seed (NumPy's `mt19937_seed`).
    pub fn new(seed: u32) -> Self {
        let mut key = [0_u32; N];
        let mut s = seed;
        for (pos, slot) in key.iter_mut().enumerate() {
            *slot = s;
            s = 1_812_433_253_u32
                .wrapping_mul(s ^ (s >> 30))
                .wrapping_add(pos as u32 + 1);
        }
        Self {
            key,
            pos: N,
            gauss: None,
        }
    }

    fn generate(&mut self) {
        for i in 0..N {
            let y = (self.key[i] & UPPER_MASK) | (self.key[(i + 1) % N] & LOWER_MASK);
            let mut v = self.key[(i + M) % N] ^ (y >> 1);
            if y & 1 == 1 {
                v ^= MATRIX_A;
            }
            self.key[i] = v;
        }
        self.pos = 0;
    }

    /// Next tempered 32-bit output.
    pub fn next_u32(&mut self) -> u32 {
        if self.pos == N {
            self.generate();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `random_sample()`: uniform double in `[0, 1)` from two 32-bit draws.
    pub fn next_f64(&mut self) -> f64 {
        let a = f64::from(self.next_u32() >> 5);
        let b = f64::from(self.next_u32() >> 6);
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }

    /// `standard_normal()` (legacy polar method, second deviate cached).
    pub fn next_gauss(&mut self) -> f64 {
        if let Some(g) = self.gauss.take() {
            return g;
        }
        loop {
            let x1 = 2.0 * self.next_f64() - 1.0;
            let x2 = 2.0 * self.next_f64() - 1.0;
            let r2 = x1 * x1 + x2 * x2;
            if r2 < 1.0 && r2 != 0.0 {
                let f = (-2.0 * r2.ln() / r2).sqrt();
                self.gauss = Some(f * x1);
                return f * x2;
            }
        }
    }

    /// `random_interval(max)`: uniform integer in `[0, max]` by masked rejection.
    fn interval(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let v = self.next_u32() & mask;
            if v <= max {
                return v;
            }
        }
    }

    /// `permutation(values)` for a 1-d array: an in-place Fisher-Yates shuffle
    /// of a copy, identical to NumPy's legacy `shuffle` fast path.
    pub fn permutation<T: Clone>(&mut self, values: &[T]) -> Vec<T> {
        let mut out = values.to_vec();
        for i in (1..out.len()).rev() {
            let j = self.interval(u32::try_from(i).unwrap_or(u32::MAX)) as usize;
            out.swap(i, j);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::NumpyRandomState;

    #[test]
    fn matches_numpy_random_sample() {
        // np.random.RandomState(0).rand(3)
        let mut rs = NumpyRandomState::new(0);
        let want = [
            0.548_813_503_927_324_8,
            0.715_189_366_372_419_5,
            0.602_763_376_071_643_9,
        ];
        for w in want {
            assert!((rs.next_f64() - w).abs() < 1e-15);
        }
    }

    #[test]
    fn matches_numpy_standard_normal() {
        // np.random.RandomState(1).standard_normal(3)
        let mut rs = NumpyRandomState::new(1);
        let want = [
            1.624_345_363_663_241_7,
            -0.611_756_413_650_075_4,
            -0.528_171_752_263_455_7,
        ];
        for w in want {
            assert!((rs.next_gauss() - w).abs() < 1e-12);
        }
    }

    #[test]
    fn matches_numpy_permutation() {
        // np.random.RandomState(3).permutation(list(range(10)))
        let mut rs = NumpyRandomState::new(3);
        let got = rs.permutation(&(0..10).collect::<Vec<usize>>());
        assert_eq!(got, vec![5, 4, 1, 2, 9, 6, 7, 0, 3, 8]);
        // np.random.RandomState(2).permutation(list(range(300)))[:8]
        let mut rs = NumpyRandomState::new(2);
        let got = rs.permutation(&(0..300).collect::<Vec<usize>>());
        assert_eq!(&got[..8], &[98, 259, 184, 256, 29, 254, 7, 13]);
    }
}
