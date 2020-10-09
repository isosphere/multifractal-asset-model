extern crate conv;
extern crate csv;

extern crate ndarray;
extern crate ndarray_csv;

use std::{
    error::Error,
    fmt,
    fs::File,
    iter::Sum,
    ops::{
        Add, AddAssign, Div, RangeInclusive, Rem
    }
};

use conv::*;
use csv::ReaderBuilder;

use ndarray::{Array, Array1, Array2, Axis, stack, s};
use ndarray_csv::Array2Reader;
use ndarray_glm::{Linear, ModelBuilder};

use num::{One, Zero};

/// Path to asset price data
const DATA_PATH: &str = "D:\\SPX_since_1950-01-03_inclusive.csv";

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
        assert_eq!((significance*ans).round(), (significance*known_answer[m]).round())
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
            partition_function[[m, n]] = (0 .. total_increments).map(|i| (xt[[delta_t*(i+1)]] - xt[[delta_t*i]]).abs().powf(*q) ).sum::<f64>().ln();
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
        let relative_error = q / known_q[[i]] - 1.0;
        assert!(relative_error < significance);
    }    

    let xt = arr1(&[0.00000000,0.01134002,0.016076559,0.019025544,0.024897552,0.021965917,0.02548286,0.005984458,0.00060006,0.003594911,0.011933375,0.01134002,0.012526319,0.014302985,0.015485717,0.011933375,0.004790428,0.004192878,0.009558018,0.021378486,0.023139508,0.023139508,0.033641414,0.037117721,0.038851266,0.033641414,0.032479915,0.036539185,0.034221628,0.023725847,0.023725847,0.019614299,0.028987537,0.031898805,0.030153038,0.032479915,0.036539185,0.036539185,0.033060804,0.034221628,0.033641414,0.037117721,0.038851266,0.031898805,0.031317241,0.0243119,0.02548286,0.027236792,0.034801507,0.046329069,0.048618652,0.046329069,0.045755839,0.046329069,0.052043256,0.052612895,0.052612895,0.046901856,0.050903119,0.045755839,0.037695806961566714]);

    let data_size = xt.shape()[0];

    let highly_composite_number = highest_highly_composite_number(&data_size);
    assert_eq!(60, highly_composite_number);

    let factors = (1 ..=highly_composite_number).filter(|i| highly_composite_number % i == 0).collect::<Vec<usize>>();
    assert_eq!(factors, vec![1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]);

    let partition_function = calc_partition_function(&xt, &moments, &factors);

    let known_answer: Array2<f64> = arr2(&[
        [3.967510,3.347519,2.945823,2.660320,2.350929,2.256823,1.742334,1.556920,1.334820,1.052536,0.653091,-0.032782],
        [-12.337398,-11.642580,-11.683057,-11.281222,-11.624566,-11.043196,-13.146384,-12.075256,-13.568661,-12.515851,-12.346634,-10.956494],
        [-26.144715,-24.535301,-24.998577,-23.920255,-24.361315,-23.190093,-27.603135,-24.269535,-28.152549,-25.132259,-24.941629,-21.880207],
        [-39.379117,-36.940050,-37.970071,-36.340695,-36.878809,-35.099891,-41.844315,-36.396039,-42.696375,-37.686877,-37.431865,-32.803919],
        [-52.517361,-49.264313,-50.819039,-48.704955,-49.349076,-46.945786,-55.955238,-48.516667,-57.202608,-50.237081,-49.902467,-43.727631],
        [-65.640627,-61.575243,-63.616621,-61.051755,-61.796663,-58.762046,-70.002922,-60.636798,-71.676051,-62.786967,-62.369652,-54.651343],
        [-78.761479,-73.883704,-76.388694,-73.391939,-74.228448,-70.559952,-84.022884,-72.756888,-86.122677,-75.336830,-74.836251,-65.575056],
        [-91.881930,-86.191666,-89.146482,-85.728869,-86.648310,-82.345826,-98.031081,-84.876975,-100.548506,-87.886691,-87.302749,-76.498768],
        [-105.002312,-98.499521,-101.895726,-98.063678,-99.059141,-94.123839,-112.034292,-96.997061,-114.958862,-100.436553,-99.769229,-87.422480],
        [-118.122682,-110.807352,-114.639703,-110.396785,-111.463209,-105.896772,-126.035367,-109.117147,-129.358058,-112.986414,-112.235707,-98.346192],
    ]);
    
    for m in 0 .. known_answer.nrows() {
        for n in 0 .. known_answer.ncols() {
            let relative_error = {
                known_answer[[m, n]] / partition_function[[m, n]] - 1.0
            };
            
            assert!(relative_error < significance, "[m,n] = [{}, {}]. error = {:.2}% > {:.2}%", m, n, relative_error*100.0, significance*100.0);
        }
    }
}

/// Calculates the Hurst-Holder exponent for a fractal series, using ndarray-glm
fn calc_holder(partition_function: &Array2<f64>, moments: &Array1<f64>, factors: &[usize]) -> Result<f64, Box<dyn Error>> {
    let ln_factors: Vec<f64> = factors.iter().map(|f| f64::value_from(*f).unwrap().ln() ).collect();
    let highly_composite_number = factors.last().copied().unwrap();

    let mut scaling_function: Array2<f64> = Array2::zeros((moments.shape()[0], 2));

    let (mut last_q, mut last_slope): (Option<f64>, Option<f64>) = (None, None);
    let mut holder: Option<f64> = None;

    for (m, q) in moments.iter().enumerate() {
        let y = partition_function.slice(s![m, ..]).to_owned() / partition_function[[m, 0]];
        let mut x = Array2::from_elem((y.shape()[0], 2), 0.0);
        
        for (m, factor) in ln_factors.iter().enumerate() {
            x[[m, 0]] = *factor;
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
            Ok(h)
        },
        None => {
            Err(Box::new(AnalysisFailure("Unable to determine Hurst expontent. Consider adding more Q's near the beginning of the series?".into())))
        }
    }
}

/// A function to determine the hurst-holder exponent for a fractal series for different depths of data.
/// If the result is volatile and shows no stability, it may imply this analysis approach is invalid for the series.
fn holder_stability(factors: &[usize], partition_function: &Array2<f64>, moments: &Array1<f64>) {
    for i in 0 .. factors.len() - 1 {
        let max_index = factors.len() - i;

        let holder = calc_holder(&partition_function.slice(s![.., 0..max_index]).to_owned(), &moments, &factors[..max_index]).unwrap();
        println!("{},{:.4}", factors[max_index-1], holder)
    }
}

fn main() {
    let file = File::open(DATA_PATH).unwrap();
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let array_read: Array2<f64> = reader.deserialize_array2_dynamic().unwrap();
        
    let xt = compound_price(&array_read);

    // We expect the zero crossover to occur somewhere around 2.0, so we use more points around there.
    let moments = stack![
        Axis(0),
        Array::linspace(0.01, 0.1, 30), Array::linspace(0.11, 0.5, 30), Array::linspace(0.51, 1.5, 30),
        Array::linspace(1.51, 2.5, 120), Array::linspace(2.51, 6.0, 30), Array::linspace(6.1, 30.0, 20)
    ];

    let data_size = xt.shape()[0];
    let highly_composite_number = highest_highly_composite_number(&data_size);
    let factors = (1 ..=highly_composite_number).filter(|i| highly_composite_number % i == 0).collect::<Vec<usize>>();

    if highly_composite_number < 7560 {
        println!("Warning: at least 30 years of data is required for a good fractal analysis, see Peters, E.E., 1991. 'Chaos and order in the capital markets: a new view of cycles, prices, and market volatility'.");
        println!("That timeframe corresponds with a highly composite number of 7560, but here we only have {}.", highly_composite_number);
    }

    println!("There are {} factors of our highly composite number ({}), and we have {} moments (q) to compute with.", factors.len(), highly_composite_number, moments.shape()[0]);
    println!("We are ignoring {} data points ({:.2}% of the series) due to our insistence on a highly composite number.", data_size - highly_composite_number, 1 - highly_composite_number/data_size*100);

    let partition_function = calc_partition_function(&xt, &moments, &factors);

    let holder = calc_holder(&partition_function, &moments, &factors).unwrap();
    println!("Full-series holder exponent: {:.2}", holder);

    //holder_stability(&factors, &partition_function, &moments);
}