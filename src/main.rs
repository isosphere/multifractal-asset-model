extern crate conv;
extern crate csv;

extern crate ndarray;
extern crate ndarray_csv;

use conv::*;
use csv::{ReaderBuilder, WriterBuilder};

use linreg::linear_regression;

use ndarray::{Array, Array1, Array2, Axis, stack, s};
use ndarray_csv::{Array2Reader, Array2Writer};

use std::{
    error::Error,
    fmt,
    fs::File,
    iter::Sum,
    ops::{
        Add, AddAssign, Div, RangeInclusive, Rem
    }
};

use num::{One, Zero};

/// Path to asset price data
//const DATA_PATH: &str = "D:\\SPX_since_1950-01-03_inclusive.csv";
const DATA_PATH: &str = "D:\\SPX_since_2001-06-15_inclusive.csv";
const OUTPUT_PATH: &str = "D:\\scaling_function.csv";

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

fn compound_price(file_path: &str) -> Result<Array1<f32>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let array_read: Array2<f32> = reader.deserialize_array2_dynamic()?;

    let first_price = array_read.slice(s![0, 1]).first().unwrap().ln();
    Ok(array_read.index_axis(Axis(1), 1).map(|p| p.ln() - first_price))
}

/// S(q, T, delta_T) -  q == moment, T == highly_composite_number, delta_t = factors
/// S(q, T, delta_T) = sum_{i=0, N-1} abs( Xt(deltaT*(i+1)) - Xt(i*deltaT) )**q
/// N = total number of time increments for that particular deltaT
fn calc_partition_function(xt: &Array1<f32>, moments: &Array1<f32>, factors: &[usize]) -> Array2<f32> {
    let mut partition_function: Array2<f32> = Array2::zeros((moments.shape()[0], factors.len()));
    let highly_composite_number = factors.last().copied().unwrap();


    for (m, q) in moments.iter().enumerate() {
        for (n, delta_t) in factors.iter().enumerate() {    
            let total_increments = highly_composite_number / delta_t;
            partition_function[[m, n]] = (0 .. total_increments).map(|i| (xt[[delta_t*(i+1)]] - xt[[delta_t*i]]).abs().powf(*q) ).sum::<f32>().ln();
        }
    }

    partition_function
} 

fn calc_holder(partition_function: &Array2<f32>, moments: &Array1<f32>, factors: &[usize]) -> Result<f32, Box<dyn Error>> {
    let ln_factors: Vec<f32> = factors.iter().map(|f| f32::value_from(*f).unwrap().ln() ).collect();
    let mut scaling_function: Array2<f32> = Array2::zeros((moments.shape()[0], 2));

    let (mut last_q, mut last_slope): (Option<f32>, Option<f32>) = (None, None);
    let mut holder: Option<f32> = None;

    for (m, q) in moments.iter().enumerate() {
        let y = partition_function.slice(s![m, ..]).to_vec();
        //println!("y = {:?}", y);
        let (slope, _intercept): (f32, f32) = match linear_regression(&ln_factors[..ln_factors.len()], &y) {
            Ok((slope, _intercept)) => {
                (slope, _intercept)
            },
            Err(e) => {
                println!("q = {}, slope error: '{}'. breaking from loop", q, e);
                break;
            }
        };
        scaling_function[[m, 0]] = *q;
        scaling_function[[m, 1]] = slope;
        
        // interpolate zero crossover
        match (holder, last_q, last_slope) {
            (None, Some(last_q), Some(last_slope)) => {
                if last_slope < 0.0 && slope >= 0.0 {
                    holder = Some(1.0/((-last_slope*(*q - last_q)/(slope-last_slope))+last_q));
                }
            }
            (_, _, _) => {}
        }

        last_q = Some(*q);
        last_slope = Some(slope);
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

fn main() {
    let xt = compound_price(DATA_PATH).unwrap();

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
    println!("We are ignoring {} data points ({:.2}% of series) due to our insistence on a highly composite number.", data_size - highly_composite_number, 1 - highly_composite_number/data_size*100);

    let partition_function = calc_partition_function(&xt, &moments, &factors);

    let holder = calc_holder(&partition_function, &moments, &factors).unwrap();
    println!("Holder exponent: {:.2}", holder)

}