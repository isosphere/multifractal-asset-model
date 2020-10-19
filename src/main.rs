extern crate conv;
extern crate csv;

extern crate ndarray;
extern crate ndarray_csv;

use std::{
    cmp::Ordering,
    error::Error,
    fmt,
    fs::OpenOptions,
    iter::Sum,
    ops::{
        Add, AddAssign, Div, RangeInclusive, Rem
    }
};

use clap::{Arg, App};
use conv::*;
use csv::ReaderBuilder;
use itertools_num::ItertoolsNum;
use indicatif::ParallelProgressIterator;
use fbm::{FBM};
use fbm::Methods::{Hosking, DaviesHarte};
use ndarray::{Array, Array1, Array2, Axis, stack, s};
use ndarray_csv::Array2Reader;
use ndarray_glm::{Linear, ModelBuilder};
use num::{One, Zero};
use plotters::prelude::*;
use rand::rngs::ThreadRng;
use rand_distr::{LogNormal, Distribution};
use rayon::prelude::*;
use roots::{find_root_secant, SimpleConvergency};


#[derive(Debug)]
struct AnalysisFailure(String);

impl fmt::Display for AnalysisFailure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Unable to proceed with analysis: {}", self.0)
    }
}

impl Error for AnalysisFailure {}

fn command_usage<'a, 'b>() -> App<'a, 'b> {
    const DEFAULT_K: &str = "13";
    const DEFAULT_ITERATIONS: &str = "1000";
    const DEFAULT_FBM_MAGNITUDE: &str = "0.10";
    const DEFAULT_OUTPUT: &str = "mmar.png";

    App::new("multifractal-asset-model")
    .author("Matthew Scheffel <matt@dataheck.com>")
    .about("Simulates price data using an estimated multifractal spectrum of a given price series")
    .arg(
        Arg::with_name("input")
            .long("input")
            .takes_value(true)
            .required(true)
            .help("Location of a CSV file that contains two columns of _FLOATING POINT_ data. Index, and price, in that order. The index will be ignored. Will break if given mixed data types.")
    )
    .arg(
        Arg::with_name("k")
            .short("k")
            .long("k")
            .takes_value(true)
            .default_value(DEFAULT_K)
            .help("Determines how many days to simulate: 2^k")
    )
    .arg(
        Arg::with_name("iterations")
            .long("iterations")
            .short("i")
            .takes_value(true)
            .default_value(DEFAULT_ITERATIONS)
            .help("Determines how many simulated price series are generated.")
    )
    .arg(
        Arg::with_name("fbm-magnitude")
            .long("fbm-magnitude")
            .short("m")
            .takes_value(true)
            .default_value(DEFAULT_FBM_MAGNITUDE)
            .help("The magnitude parameter for fractional brownian motion.")
    )
    .arg(
        Arg::with_name("output")
            .long("output")
            .short("o")
            .takes_value(true)
            .default_value(DEFAULT_OUTPUT)
            .help("Filename of output PNG file.")
    )
}

/// Returns the largest highly composite number less than or equal to the given number.
/// There is nothing clever about this algorithm, it is slow.
/// # Arguments
/// * `maximum` - A reference to a number that the highly composite number must not exceed
/// # Examples
/// ```
/// assert_eq!(highest_highly_composite_number(&842), 840)
/// ```
fn highest_highly_composite_number<'a, T: 'a>(maximum: &T) -> T 
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

/// Calculates the compounding natural log price for prices stored in a CSV file.
/// 
/// The compounding natural log price is X(t) = ln(P(t)) - ln(P(0))
/// This results in a series that begins at zero and represents the ln of the ratio of P(t) to the initial price.
fn compound_price(array_read: &Array2<f64>) -> Array1<f64> {
    let first_price = array_read.slice(s![0, 1]).first().unwrap().ln();
    array_read.index_axis(Axis(1), 1).map(|p| p.ln() - first_price)
}

#[test]
fn test_compound_price() {
    use ndarray::{arr1, arr2};

    let test_array = arr2(&[
        [0.00,16.66],
        [1.00,16.85],
        [2.00,16.93],
        [3.00,16.98],
        [4.00,17.08],
        [5.00,17.030001],
        [6.00,17.09],
        [7.00,16.76],
        [8.00,16.67],
        [9.00,16.72],
        [10.00,16.86],
        [11.00,16.85],
        [12.00,16.87],
        [13.00,16.90],
        [14.00,16.92],
        [15.00,16.86],
        [16.00,16.74],
        [17.00,16.73],
        [18.00,16.82],
        [19.00,17.02],
        [20.00,2643.850098],
        [21.00,2640.000000],
        [22.00,2681.050049],
        [26.00,2704.100098],
        [27.00,2706.530029]
    ]);

    let xt = compound_price(&test_array);
    let known_answer: Array1<f64> = arr1(&[
        0.000000, 0.011340, 0.016077, 0.019026, 0.024898, 0.021966, 0.025483, 0.005984, 0.000600, 0.003595, 0.011933, 0.011340, 
        0.012526, 0.014303, 0.015486, 0.011933, 0.004790, 0.004193, 0.009558, 0.021378, 5.066981, 5.065524, 5.080953, 5.089514, 5.090412
    ]);

    let significance = 1e4;
    for (m, ans) in xt.iter().enumerate() {
        let error = (ans / (known_answer[m]+1e-60) - 1.0).abs();
        assert!(error < significance, "error {} > {} ", error, significance);
    }
}

/// Calculates the **natural log** of the partition function for the series across all moments and factors.
///
/// The partition function is defined as:
/// S(q, T, delta_T) = sum_{i=0, N-1} abs( Xt(deltaT*(i+1)) - Xt(i*deltaT) )**q
///
/// Where q == moment, T == highly_composite_number, the total series length, delta_t = factor
/// N = total number of time increments for that particular deltaT = (T/delta_T)
fn calc_partition_function(xt: &Array1<f64>, moments: &Array1<f64>, factors: &[usize]) -> Array2<f64> {
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
fn calc_holder(partition_function: &Array2<f64>, moments: &Array1<f64>, factors: &[usize]) -> Result<(f64, Array2<f64>), Box<dyn Error>> {
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

/// Estimate the fractal spectrum given the taq(q) matrix. Returns the required parameters for the MMAR simulation.
fn calc_spectrum(tau_q: &Array2<f64>) -> Result<(Array2<f64>, f64, f64), Box<dyn Error>> {
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

    Ok((result, lambda, sigma))
}

/// Recursive function that generates a lognormal multiplicative cascade to be used as "trading time"
fn lognormal_cascade(k: &i32, mut cascade: Vec<f64>, ln_lambda: &f64, ln_theta: &f64, rng: &mut ThreadRng) -> Vec<f64> {
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

/// Simulates a multifractal model of asset returns using a combination of fractional brownian motion
/// and a lognormal cascade of trading time to generate a multifractal series that matches the characteristics
/// specified.
fn mmar_simulation(k: i32, holder: &f64, ln_lambda: &f64, ln_theta: &f64, fbm_magnitude: &f64) -> Vec<f64> {
    let mut cascade = vec![1.0, 1.0];
    let mut rng = rand::thread_rng();

    cascade = lognormal_cascade(&k, cascade, &ln_lambda, &ln_theta, &mut rng);
    let sum = cascade.iter().sum::<f64>();
    cascade = cascade.iter().cumsum::<f64>().map(|i| i*2.0f64.powi(k)/sum).collect(); // normalized trading time

    let samples: usize = 10*2usize.pow(k.value_as::<u32>().unwrap()) + 1usize;
    
    let mut fbm = FBM::new(Hosking, samples, *holder, *fbm_magnitude);
    let sampled_fbm = fbm.fbm();

    //let fbm = Motion::new(*holder);
    //let mut source = source::default().seed([rng.gen::<u64>(), rng.gen::<u64>()]);
    //let sampled_fbm = fbm.sample(samples, *fbm_magnitude, &mut source);

    let simulated_xt: Vec<f64> = (0 .. cascade.len()).map(|i| sampled_fbm[ (cascade[i] * 10.0) as usize] ).collect();

    simulated_xt
}

fn plot_simulation(output: &str, all_simulations: &[Vec<f64>], xt: &Array1<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut sorted_final_positions_min: Vec<f64> = all_simulations.iter().map(|v| {
        let mut v_sorted: Vec<f64> = v.to_owned();
        v_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        *v_sorted.first().unwrap()
    }).collect::<Vec<f64>>();
    sorted_final_positions_min.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut sorted_final_positions_max: Vec<f64> = all_simulations.iter().map(|v| {
        let mut v_sorted: Vec<f64> = v.to_owned();
        v_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        *v_sorted.last().unwrap()
    }).collect::<Vec<f64>>();
    sorted_final_positions_max.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let (mut xt_min, mut xt_max) = (0.0, 0.0);
    for e in xt {
        if *e > xt_max { xt_max = *e }
        else if *e < xt_min { xt_min = *e }
    }

    let (min_y, max_y) = {
        (
            match f64::partial_cmp(sorted_final_positions_min.first().unwrap(), &xt_min) {
                Some(Ordering::Less) => { *sorted_final_positions_min.first().unwrap() },
                _ => { xt_min }
            }*1.10, 
            match f64::partial_cmp(sorted_final_positions_max.last().unwrap(), &xt_max) {
                Some(Ordering::Greater) => { *sorted_final_positions_max.last().unwrap() },
                _ => { xt_max }
            }*1.10
        )
    };

    assert_ne!(min_y, max_y, "Chart min and max y values are the same - this can't be plotted!");

    let max_x = xt.shape()[0] as f64;
    println!("Chart min_y, max_y = ({:.2}, {:.2})", min_y, max_y);
    println!("Chart min_x, max_x = (0.0, {:.2})", max_x);

    let mut chart = ChartBuilder::on(&root)
        .caption("MMAR Simulations", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0 .. max_x, min_y .. max_y)?;
    
    chart.configure_mesh().draw()?;

    let mut upper_quartile: Vec<f64> = Vec::new();
    let mut lower_quartile: Vec<f64> = Vec::new();
    let mut median: Vec<f64> = Vec::new();
    
    for n in 0 .. all_simulations[0].len() {
        let column: Vec<f64> = (0 .. all_simulations.len()).map(|m| all_simulations[m][n]).collect();
        let quartiles = Quartiles::new(&column);
        median.push(quartiles.median());
        // FIXME: hardcoding locations. library also is restricted to 25% and 75% percentiles.
        upper_quartile.push(quartiles.values()[3].value_as::<f64>().unwrap());
        lower_quartile.push(quartiles.values()[1].value_as::<f64>().unwrap());
    }
    
    let mut flag = true;
    for simulation in all_simulations {
        let series: Vec<(f64, f64)> = (0 .. simulation.len()).map(|i| (i.value_as::<f64>().unwrap(), simulation[i])).collect();
        
        match flag {
            false => {
                chart
                .draw_series(
                    LineSeries::new(series, &RED)
                )?;
            },
            true => {
                chart
                .draw_series(
                    LineSeries::new(series, &RED)
                )?.label("simulated").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
                flag = false;
            }
        }
    }

    for stats in &[("25% quartile", lower_quartile), ("median", median), ("75% quartile", upper_quartile)] {
        let series: Vec<(f64, f64)> = (0 .. all_simulations[0].len()).map(|i| (i.value_as::<f64>().unwrap(), stats.1[i])).collect();
        chart
        .draw_series(
            LineSeries::new(series, &GREEN)
        )?.label(stats.0).legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
    }

    {
        let series: Vec<(f64, f64)> = (0 .. xt.shape()[0]).map(|i| (i as f64, xt[[i]])).collect();
        chart
        .draw_series(
            LineSeries::new(series, &BLACK)
        )?.label("real").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn _plot_simulation_histogram(output: &str, all_simulations: &[Vec<f64>]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output, (1024, 768)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .caption("1D Gaussian Distribution Demo", ("sans-serif", 30))
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .set_label_area_size(LabelAreaPosition::Right, 60)
        .build_cartesian_2d(-50f64..50f64, 0f64..1.0f64)?
        .set_secondary_coord(
            (-10f64..10f64).step(0.1).use_round().into_segmented(),
            0u32..15000u32,
        );

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .y_label_formatter(&|y| format!("{:.0}%", *y * 100.0))
        .y_desc("Percentage")
        .draw()?;

    chart.configure_secondary_axes().y_desc("Count").draw()?;

    let mut return_series: Vec<f64> = Vec::new();
    for simulation in all_simulations {
        for n in 1 .. all_simulations[0].len() {
            return_series.push( simulation[n]/simulation[n-1] - 1.0)
        }
    }
    
    let actual = Histogram::vertical(chart.borrow_secondary())
        .style(GREEN.filled())
        .margin(3)
        .data(return_series.iter().map(|x| (*x, 1)));

    println!("meow");

    chart
        .draw_secondary_series(actual)?
        .label("Observed")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], GREEN.filled()));

    // let pdf = LineSeries::new(
    //     (-400..400).map(|x| x as f64 / 100.0).map(|x| {
    //         (
    //             x,
    //             (-x * x / 2.0 / sd / sd).exp() / (2.0 * std::f64::consts::PI * sd * sd).sqrt()
    //                 * 0.1,
    //         )
    //     }),
    //     &RED,
    // );

    // chart
    //     .draw_series(pdf)?
    //     .label("PDF")
    //     .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.filled()));

    chart.configure_series_labels().draw()?;

    Ok(())
}

fn main() {
    let matches = command_usage().get_matches();

    let array_read: Array2<f64> = {
        let file = OpenOptions::new().read(true).write(false).create(false).open(matches.value_of("input").unwrap()).unwrap();
        let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
        
        match reader.deserialize_array2_dynamic() {
            Ok(v) => v,
            Err(_) => {panic!("Unable to process the given input file. Please ensure it contains two columns, with headers, and each column contains only floating point numbers.")}
        }
    };
    
    let xt = compound_price(&array_read);
    let price = array_read.slice(s![.., 1]).to_owned();
    let first_price = price[[0]];

    // We expect the zero crossover to occur somewhere around 2.0, so we use more points around there.
    let moments = stack![
        Axis(0),
        Array::linspace(0.01, 0.1, 100), Array::linspace(0.11, 0.5, 100), Array::linspace(0.51, 1.24, 100),
        Array::linspace(1.25, 2.5, 250), Array::linspace(2.51, 6.0, 100), Array::linspace(6.1, 30.0, 100)
    ];

    let data_size = xt.shape()[0];
    let highly_composite_number = highest_highly_composite_number(&data_size);
    let factors = (1 ..=highly_composite_number).filter(|i| highly_composite_number % i == 0).collect::<Vec<usize>>();

    println!("Processed {} rows of data.", data_size);

    if highly_composite_number < 7560 {
        println!("Warning: at least 30 years of data is required for a good fractal analysis, see Peters, E.E., 1991. 'Chaos and order in the capital markets: a new view of cycles, prices, and market volatility'.");
        println!("That timeframe corresponds with a highly composite number of 7560, but here we only have {}.", highly_composite_number);
    }

    println!("There are {} factors of our highly composite number ({}), and we have {} moments (q) to compute with.", factors.len(), highly_composite_number, moments.shape()[0]);
    println!("We are ignoring {} data points ({:.2}% of the series) due to our insistence on a highly composite number.", data_size - highly_composite_number, 1 - highly_composite_number/data_size*100);

    let partition_function = calc_partition_function(&xt, &moments, &factors);

    let (holder, tau_q) = calc_holder(&partition_function, &moments, &factors).unwrap();
    println!("Full-series holder exponent estimated via linear interpolation: {:.2}", holder);
    let (_f_a, ln_lambda, ln_theta) = calc_spectrum(&tau_q).unwrap();

    let k: i32 = matches.value_of("k").unwrap().parse::<i32>().unwrap_or_else(|_| panic!("Invalid k specified: '{}.'", matches.value_of("k").unwrap()));
    let iterations: usize = matches.value_of("iterations").unwrap().parse::<usize>().unwrap_or_else(|_| panic!("Invalid iterations specified: '{}.'", matches.value_of("iterations").unwrap()));
    let fbm_magnitude: f64 = matches.value_of("fbm-magnitude").unwrap().parse::<f64>().unwrap_or_else(|_| panic!("Invalid fbm magnitude specified: '{}.'", matches.value_of("fbm").unwrap()));
    let output = matches.value_of("output").unwrap();

    println!("Generating MMAR simulations.");
    let all_simulations: Vec<Vec<f64>> = (0..iterations).into_par_iter()
                                                        .progress_count(iterations.value_as::<u64>().unwrap())
                                                        .map(|_i| mmar_simulation(k, &holder, &ln_lambda, &ln_theta, &fbm_magnitude)).collect();

    println!("Transforming to price series.");
    let all_simulations_price: Vec<Vec<f64>> = (0 .. iterations).map(|i| all_simulations[i].iter().map(|v| v.exp()*first_price).collect()).collect();

    println!("Plotting.");
    plot_simulation(output, &all_simulations_price, &price).unwrap();
    //plot_simulation_histogram("distribution.png", &all_simulations).unwrap();
}