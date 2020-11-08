extern crate conv;
extern crate csv;

extern crate ndarray;
extern crate ndarray_csv;

use std::{
    fs::OpenOptions,
};

use clap::{Arg, App};
use conv::*;
use csv::ReaderBuilder;
use indicatif::ParallelProgressIterator;
use itertools_num::ItertoolsNum;
use fbm::{FBM};
use fbm::Methods::{Hosking, DaviesHarte, Cholesky};
use ndarray::{Array, Array2, s, Axis, stack};
use ndarray_csv::Array2Reader;
use rayon::prelude::*;



mod utilities;
mod fractal;
mod plot;


fn command_usage<'a, 'b>() -> App<'a, 'b> {
    const DEFAULT_K: &str = "13";
    const DEFAULT_ITERATIONS: &str = "1000";
    const DEFAULT_FBM_MAGNITUDE: &str = "1.00";
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
        Arg::with_name("hurst")
            .short("h")
            .long("hurst")
            .takes_value(true)
            .help("Override hurst exponent")
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

/// Simulates a multifractal model of asset returns using a combination of fractional brownian motion
/// and a lognormal cascade of trading time to generate a multifractal series that matches the characteristics
/// specified.
fn mmar_simulation(k: i32, holder: &f64, ln_lambda: &f64, ln_theta: &f64, fbm_magnitude: &f64, low_limit: Option<f64>) -> Vec<f64> {
    {
        let mut cascade = vec![1.0, 1.0];
        let mut rng = rand::thread_rng();

        cascade = fractal::lognormal_cascade(&k, cascade, &ln_lambda, &ln_theta, &mut rng);
        let sum = cascade.iter().sum::<f64>();
        cascade = cascade.iter().cumsum::<f64>().map(|i| i*2.0f64.powi(k)/sum).collect(); // normalized trading time

        let samples: usize = 10*2usize.pow(k.value_as::<u32>().unwrap()) + 1usize;
        
        let mut fbm = FBM::new(DaviesHarte, samples, *holder, *fbm_magnitude);
        let sampled_fbm = fbm.fbm();
        
        let simulated_xt: Vec<f64> = (0 .. cascade.len()).map(|i| sampled_fbm[ (cascade[i] * 10.0) as usize] ).collect();
        if low_limit.is_none() || !simulated_xt.iter().any(|v| *v <= low_limit.unwrap()) {
            return simulated_xt;
        }
    }

    mmar_simulation(k, holder, ln_lambda, ln_theta, fbm_magnitude, low_limit)
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

    let xt = utilities::compound_price(&array_read);
    let price = array_read.slice(s![.., 1]).to_owned();
    let first_price = price[[0]];

    // We expect the zero crossover to occur somewhere around 2.0, so we use more points around there.
    let moments = stack![
        Axis(0),
        Array::linspace(0.01, 0.1, 100), Array::linspace(0.11, 0.5, 100), Array::linspace(0.51, 1.24, 100),
        Array::linspace(1.25, 2.5, 250), Array::linspace(2.51, 6.0, 100), Array::linspace(6.1, 30.0, 100)
    ];

    let data_size = xt.shape()[0];
    let highly_composite_number = fractal::highest_highly_composite_number(&data_size);
    let factors = (1 ..=highly_composite_number).filter(|i| highly_composite_number % i == 0).collect::<Vec<usize>>();

    println!("Processed {} rows of data.", data_size);

    if highly_composite_number < 7560 {
        println!("Warning: at least 30 years of data is required for a good fractal analysis, see Peters, E.E., 1991. 'Chaos and order in the capital markets: a new view of cycles, prices, and market volatility'.");
        println!("That timeframe corresponds with a highly composite number of 7560, but here we only have {}.", highly_composite_number);
    }

    println!("There are {} factors of our highly composite number ({}), and we have {} moments (q) to compute with.", factors.len(), highly_composite_number, moments.shape()[0]);
    println!("We are ignoring {} data points ({:.2}% of the series) due to our insistence on a highly composite number.", data_size - highly_composite_number, 1 - highly_composite_number/data_size*100);

    let partition_function = fractal::calc_partition_function(&xt, &moments, &factors);

    let (holder, tau_q) = match matches.is_present("hurst") {
        true => (
            matches.value_of("hurst").unwrap().parse::<f64>().unwrap_or_else(|_| panic!("Invalid hurst exponent override specified: '{}'.", matches.value_of("hurst").unwrap())),
            fractal::calc_holder(&partition_function, &moments, &factors).unwrap().1
        ),
        false => fractal::calc_holder(&partition_function, &moments, &factors).unwrap()
    };

    println!("Full-series holder exponent estimated via linear interpolation: {:.2}", holder);
    let (_f_a, ln_lambda, ln_theta, _holder_spectrum) = fractal::calc_spectrum(&tau_q).unwrap();

    let k: i32 = matches.value_of("k").unwrap().parse::<i32>().unwrap_or_else(|_| panic!("Invalid k specified: '{}.'", matches.value_of("k").unwrap()));
    let iterations: usize = matches.value_of("iterations").unwrap().parse::<usize>().unwrap_or_else(|_| panic!("Invalid iterations specified: '{}.'", matches.value_of("iterations").unwrap()));
    let fbm_magnitude: f64 = matches.value_of("fbm-magnitude").unwrap().parse::<f64>().unwrap_or_else(|_| panic!("Invalid fbm magnitude specified: '{}.'", matches.value_of("fbm").unwrap()));
    let output = matches.value_of("output").unwrap();

    println!("Generating MMAR simulations.");
    let all_simulations: Vec<Vec<f64>> = (0..iterations).into_par_iter()
                                                        .progress_count(iterations.value_as::<u64>().unwrap())
                                                        .map(|_i| mmar_simulation(k, &holder, &ln_lambda, &ln_theta, &fbm_magnitude, None)).collect();

    println!("Transforming to price series.");
    let all_simulations_price: Vec<Vec<f64>> = (0 .. iterations).map(|i| all_simulations[i].iter().map(|v| v.exp()*first_price).collect()).collect();

    println!("Plotting.");
    plot::plot_simulation(output, &all_simulations_price, &price).unwrap();
    //plot_simulation_histogram("distribution.png", &all_simulations).unwrap();
}