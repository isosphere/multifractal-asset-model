use conv::*;

use std::{
    error::Error,
    fmt,
    ops::{
        Add, AddAssign, Div, RangeInclusive, Rem
    },
    iter::Sum,
};

use num::{One, Zero};
use rand::rngs::ThreadRng;
use rand_distr::{LogNormal, Distribution};
use roots::{find_root_secant, SimpleConvergency};
use ndarray_glm::{Linear, ModelBuilder};
use ndarray::{Array, Array1, Array2, s};

#[derive(Debug)]
struct AnalysisFailure(String);

impl fmt::Display for AnalysisFailure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Unable to proceed with analysis: {}", self.0)
    }
}

impl Error for AnalysisFailure {}

/// Returns the largest highly composite number less than or equal to the given number.
/// There is nothing clever about this algorithm, it is slow.
/// # Arguments
/// * `maximum` - A reference to a number that the highly composite number must not exceed
/// # Examples
/// ```
/// assert_eq!(highest_highly_composite_number(&842), 840)
/// ```
pub fn highest_highly_composite_number<'a, T: 'a>(maximum: &T) -> T 
where
    T: Copy + Sum + PartialOrd + PartialEq + Add<Output=T> + AddAssign + Rem<Output=T> + Div<Output=T> + Zero + One,
    RangeInclusive<T>: Iterator<Item = T>
{
    let two = T::one() + T::one();
    let mut max_divisors = T::one();
    let mut max_result = T::one();

    let mut i = T::one();

    while i <= *maximum {
        let divisors = T::one() + (two ..=*maximum).map(|d| match i % d == T::zero() { true => T::one(), false => T::zero()} ).sum();

        if divisors > max_divisors {
            max_divisors = divisors;
            max_result = i;
        }

        i += T::one();
    }

    max_result
}

#[test]
fn test_hcn() {
    let known_hcn = vec![1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680, 2520, 5040];
    for hcn in &known_hcn {
        assert_eq!(*hcn, highest_highly_composite_number(hcn));
    }

    let known_not_hcn = vec![3, 5, 7, 13, 25, 37, 49, 55, 112, 145, 221, 350];
    for hcn in &known_not_hcn {
        assert_ne!(*hcn, highest_highly_composite_number(hcn));
    }
}

/// Calculates the **natural log** of the partition function for the series across all moments and factors.
///
/// The partition function is defined as:
/// S(q, T, delta_T) = sum_{i=0, N-1} abs( Xt(deltaT*(i+1)) - Xt(i*deltaT) )**q
///
/// Where q == moment, T == highly_composite_number, the total series length, delta_t = factor
/// N = total number of time increments for that particular deltaT = (T/delta_T)
pub fn calc_partition_function(xt: &Array1<f64>, moments: &Array1<f64>, factors: &[usize]) -> Array2<f64> {
    let mut partition_function: Array2<f64> = Array2::zeros((moments.shape()[0], factors.len()));
    let highly_composite_number = factors.last().copied().unwrap();

    for (m, q) in moments.iter().enumerate() {
        for (n, delta_t) in factors.iter().enumerate() {    
            let total_increments = highly_composite_number / delta_t;
            partition_function[[m, n]] = (0 .. total_increments).map(|i| (xt[[delta_t*(i+1)]] - xt[[delta_t*i]]).abs().powf(*q) ).sum::<f64>();
        }
    }

    partition_function
}

#[test]
fn test_calc_partition_function() {
    use ndarray::{arr1, arr2};

    let significance = 0.001; // relative error that is acceptable

    let moments = Array::linspace(0.01, 30.0, 10);

    // this part of the test is overkill, but I want to prevent any doubt about my approach
    let known_q = arr1(&[
        1.00000000e-02, 3.34222222e+00, 6.67444444e+00, 1.00066667e+01, 1.33388889e+01, 1.66711111e+01, 2.00033333e+01, 2.33355556e+01, 2.66677778e+01, 3.00000000e+01
    ]);

    for (i, q) in moments.iter().enumerate() {
        let relative_error: f64 = *q / known_q[[i]] - 1.0;
        assert!(relative_error.abs() < significance);
    }    

    let xt = arr1(&[0.00000000,0.01134002,0.016076559,0.019025544,0.024897552,0.021965917,0.02548286,0.005984458,0.00060006,0.003594911,0.011933375,0.01134002,0.012526319,0.014302985,0.015485717,0.011933375,0.004790428,0.004192878,0.009558018,0.021378486,0.023139508,0.023139508,0.033641414,0.037117721,0.038851266,0.033641414,0.032479915,0.036539185,0.034221628,0.023725847,0.023725847,0.019614299,0.028987537,0.031898805,0.030153038,0.032479915,0.036539185,0.036539185,0.033060804,0.034221628,0.033641414,0.037117721,0.038851266,0.031898805,0.031317241,0.0243119,0.02548286,0.027236792,0.034801507,0.046329069,0.048618652,0.046329069,0.045755839,0.046329069,0.052043256,0.052612895,0.052612895,0.046901856,0.050903119,0.045755839,0.037695806961566714]);

    let data_size = xt.shape()[0];

    let highly_composite_number = highest_highly_composite_number(&data_size);
    assert_eq!(60, highly_composite_number);

    let factors = (1 ..=highly_composite_number).filter(|i| highly_composite_number % i == 0).collect::<Vec<usize>>();
    assert_eq!(factors, vec![1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]);

    let partition_function = calc_partition_function(&xt, &moments, &factors);

    let known_answer: Array2<f64> = arr2(&[
        [5.285277e+01,2.843212e+01,1.902631e+01,1.430086e+01,1.049532e+01,9.552691e+00,5.710654e+00,4.744187e+00,3.799311e+00,2.864909e+00,1.921470e+00,9.677494e-01],
        [4.384661e-06,8.783990e-06,8.435539e-06,1.260745e-05,8.943654e-06,1.599561e-05,1.952530e-06,5.698792e-06,1.279986e-06,3.668047e-06,4.344352e-06,1.744436e-05],
        [4.420734e-12,2.210314e-11,1.390772e-11,4.088510e-11,2.630361e-11,8.485362e-11,1.028278e-12,2.883202e-11,5.936131e-13,1.216742e-11,1.472272e-11,3.144467e-10],
        [7.904361e-18,9.060246e-17,3.234505e-17,1.649821e-16,9.632451e-17,5.705727e-16,6.718074e-19,1.560994e-16,2.865497e-19,4.293358e-17,5.540482e-17,5.668120e-15],
        [1.555966e-23,4.025139e-22,8.502993e-23,7.042173e-22,3.698015e-22,4.089818e-21,4.999756e-25,8.501183e-22,1.436226e-25,1.521644e-22,2.126346e-22,1.021718e-19],
        [3.109124e-29,1.812223e-27,2.353167e-28,3.058861e-27,1.452279e-27,3.019727e-26,3.963850e-31,4.632048e-27,7.438521e-32,5.394693e-28,8.188499e-28,1.841718e-24],
        [6.227648e-35,8.179278e-33,6.680542e-34,1.337476e-32,5.794214e-33,2.270922e-31,3.230913e-37,2.523974e-32,3.957279e-38,1.912628e-33,3.155218e-33,3.319825e-29],
        [1.247914e-40,3.693475e-38,1.923866e-39,5.867126e-38,2.339468e-38,1.728471e-36,2.664664e-43,1.375303e-37,2.149506e-44,6.781017e-39,1.215901e-38,5.984216e-34],
        [2.500779e-46,1.668022e-43,5.587897e-45,2.579203e-43,9.531508e-44,1.325978e-41,2.208642e-49,7.493967e-43,1.185770e-50,2.404138e-44,4.685697e-44,1.078697e-38],
        [5.011536e-52,7.533184e-49,1.631585e-50,1.135757e-48,3.909700e-49,1.022390e-46,1.834576e-55,4.083432e-48,6.614686e-57,8.523615e-50,1.805725e-49,1.944426e-43],
    ]);
    
    for m in 0 .. known_answer.nrows() {
        for n in 0 .. known_answer.ncols() {
            let relative_error = {
                (known_answer[[m, n]] / partition_function[[m, n]] - 1.0).abs()
            };
            
            assert!(relative_error < significance, "[m,n] = [{}, {}]. error = {:.2}% > {:.2}%", m, n, relative_error*100.0, significance*100.0);
        }
    }
}

/// Calculates the Hurst-Holder exponent for a fractal series, using ndarray-glm and interpolation over q
pub fn calc_holder(partition_function: &Array2<f64>, moments: &Array1<f64>, factors: &[usize]) -> Result<(f64, Array2<f64>), Box<dyn Error>> {
    let highly_composite_number = factors.last().copied().unwrap();

    let mut scaling_function: Array2<f64> = Array2::zeros((moments.shape()[0], 2));

    let (mut last_q, mut last_slope): (Option<f64>, Option<f64>) = (None, None);
    let mut holder: Option<f64> = None;

    for (m, q) in moments.iter().enumerate() {
        let y = (partition_function.slice(s![m, ..]).to_owned() / partition_function[[m, 0]]).map(|p| p.ln());

        let mut x = Array2::from_elem((y.shape()[0], 2), 0.0);
        for (m, factor) in factors.iter().enumerate() {
            x[[m, 0]] = (*factor).value_as::<f64>().unwrap().ln();
            x[[m, 1]] = highly_composite_number.value_as::<f64>().unwrap().ln()
        }
        
        let model = ModelBuilder::<Linear>::data(y.view(), x.view()).no_constant().build().unwrap();

        let (tau_q, _c_q) = {
            let result = model.fit().unwrap().result.to_vec();
            (result[0], result[1])
        };

        scaling_function[[m, 0]] = *q;
        scaling_function[[m, 1]] = tau_q;
        
        // interpolate zero crossover
        match (holder, last_q, last_slope) {
            (None, Some(last_q), Some(last_slope)) => {
                if last_slope < 0.0 && tau_q >= 0.0 {
                    holder = Some(1.0/((-last_slope*(*q - last_q)/(tau_q-last_slope))+last_q));
                }
            }
            (_, _, _) => {}
        }

        last_q = Some(*q);
        last_slope = Some(tau_q);
    }

    match holder {
        Some(h) => {
            Ok((h, scaling_function))
        },
        None => {
            Err(Box::new(AnalysisFailure("Unable to determine Hurst expontent. Consider adding more Q's near the beginning of the series?".into())))
        }
    }
}

/// A function to determine the hurst-holder exponent for a fractal series for different depths of data.
fn _holder_stability(factors: &[usize], partition_function: &Array2<f64>, moments: &Array1<f64>) {
    for i in 0 .. factors.len() - 1 {
        let max_index = factors.len() - i;

        let (holder, _) = calc_holder(&partition_function.slice(s![.., 0..max_index]).to_owned(), &moments, &factors[..max_index]).unwrap();
        println!("{},{:.4}", factors[max_index-1], holder)
    }
}

type FractalSpectrum = (Array2<f64>, f64, f64, f64);

/// Estimate the fractal spectrum given the taq(q) matrix. Returns the required parameters for the MMAR simulation.
pub fn calc_spectrum(tau_q: &Array2<f64>) -> Result<FractalSpectrum, Box<dyn Error>> {
    let mut max_q = None;

    for (m, q) in tau_q.slice(s![.., 0]).iter().enumerate() {
        if *q >= 8.0 {
            max_q = Some(m);
            break;
        }
    }

    assert!(max_q.is_some());

    let y = tau_q.slice(s![..max_q.unwrap(), 1]).to_owned();
    let total_qs = y.shape()[0];

    let mut x: Array2<f64> = Array2::zeros((total_qs, 2));
    for (m, q) in tau_q.slice(s![..max_q.unwrap(), 0]).iter().enumerate() {
        x[[m, 0]] = q.powi(2);
        x[[m, 1]] = *q;
    }

    let model = ModelBuilder::<Linear>::data(y.view(), x.view()).build().unwrap();

    let (intercept, q_squared, q) = {
        let result = model.fit().unwrap().result.to_vec();
        (result[0], result[1], result[2])
    };

    let p = |i| {2.0*q_squared*tau_q[[i, 0]]+q};
    let f_a: Vec<f64> = (0 .. total_qs).map(|i| ((p(i)-q)/(2.0*q_squared))*p(i) - (q_squared*((p(i)-q)/(2.0*q_squared)).powi(2) + q*((p(i)-q)/(2.0*q_squared)) + intercept)).collect();

    let mut result: Array2<f64> = Array2::zeros((total_qs, 2));
    for m in 0 .. total_qs {
        result[[m, 0]] = p(m); // hurst-holder
        result[[m, 1]] = f_a[m];
    }

    let f = |x: f64| q_squared*x.powi(2) + q*x + intercept;
    let mut convergency = SimpleConvergency{ eps: 1e-15f64, max_iter: 30 };
    let root_1 = find_root_secant(0.0f64, 4.0f64, &f, &mut convergency).unwrap();

    let h_estimate = 1.0/root_1;
    let alpha_zero = result[[0, 0]];
    let lambda = alpha_zero/h_estimate;
    let sigma = (2.0*(lambda-1.0))/(2.0f64.ln());

    println!("H estimate from multifractal spectrum fit root = {:.2}", h_estimate);
    println!("alpha zero = {:.2}", alpha_zero);
    println!("lambda = {:.2} sigma^2 = {:.2} (assuming we partition our cascade in two at each step)", lambda, sigma);

    Ok((result, lambda, sigma, h_estimate))
}

/// Recursive function that generates a lognormal multiplicative cascade to be used as "trading time"
pub fn lognormal_cascade(k: &i32, mut cascade: Vec<f64>, ln_lambda: &f64, ln_theta: &f64, rng: &mut ThreadRng) -> Vec<f64> {
    let mut k = *k;

    k -= 1;

    let log_normal = LogNormal::new(*ln_lambda, *ln_theta).unwrap();
    let mass_left = log_normal.sample(rng);
    let mass_right = log_normal.sample(rng);

    if k > 0 {
        let left_side  = lognormal_cascade(&k, cascade.to_owned().into_iter().map(|v| v*mass_left).collect(), &ln_lambda, &ln_theta, rng);
        let mut right_side = lognormal_cascade(&k, cascade.to_owned().into_iter().map(|v| v*mass_right).collect(), &ln_lambda, &ln_theta, rng);

        cascade = left_side;
        cascade.append(&mut right_side);
    }

    cascade
}