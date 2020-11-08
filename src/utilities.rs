use ndarray::{Array1, Array2, s, Axis};

// Translated from https://github.com/ritchieng/fractional_differencing_gpu/blob/master/notebooks/gpu_fractional_differencing.ipynb
fn floored_weights(d: f64, k: usize, floor: f64) -> Vec<f64> {
    let mut w_k = vec![1.0];

    for i in 1 .. k {
        let w_current: f64 = -w_k.last().unwrap() * (d - (i as f64) + 1.0) / i as f64;

        if w_current.abs() <= floor {
            break
        }
        w_k.push(w_current);
    }

    w_k
}

fn fractionally_difference(series: &Array2<f64>, d:f64, floor: f64, log: bool) -> Vec<f64> {
    let n = series.shape()[0];

    let target = match log {
        true => series.map(|x| x.ln()),
        false => series.to_owned()
    };

    let weights: Array1<f64> = { 
        let mut forward_weights = floored_weights(d, n, floor);
        forward_weights.reverse();
        Array1::from(forward_weights)
    };
    let w_n = weights.shape()[0];

    let mut series_diff: Vec<f64> = Vec::new();

    assert!(w_n < n, "Window is not smaller than data shape: consider increasing the floor parameter.");

    for i in w_n .. n {
        series_diff.push(
            weights.dot(&target.slice(s![i - w_n .. i, 1]))
        )
    }

    series_diff
}

struct ADFResult {
    adf: f64,
    pvalue: f64,
    lags: usize,
    n: usize,
    critical_1: f64,
    critical_5: f64,
    critical_10: f64,
}

enum ADFRegressionType {
    ConstantOnly,
    ConstantAndTrend,
    ConstantAndLinearAndQuadraticTrend,
    NoConstantNoTrend
}

/// Augmented Dickey-Fuller unit root test.
/// The Augmented Dickey-Fuller test can be used to test for a unit root in a univariate process in the presence of serial correlation.
///
/// The null hypothesis of the Augmented Dickey-Fuller is that there is a unit root, with the alternative that there is no unit root. 
/// If the pvalue is above a critical size, then we cannot reject that there is a unit root.
/// 
/// The p-values are obtained through regression surface approximation from MacKinnon 1994, but using the updated 2010 tables. 
/// If the p-value is close to significant, then the critical values should be used to judge whether to reject the null.
///
/// Minimizes AIC to determine lag within max_lag restriction.
/// 
/// This code is adapted from statsmodels.tsa.statstools.adfuller
fn adfuller(series: &[f64], max_lag: &usize, regression: &ADFRegressionType) -> () {
    let series_n = series.len();
    let regression_n = match regression {
        ADFRegressionType::ConstantOnly => 1,
        ADFRegressionType::ConstantAndTrend => 2,
        ADFRegressionType::ConstantAndLinearAndQuadraticTrend => 3,
        ADFRegressionType::NoConstantNoTrend => 0
    };
    assert!(*max_lag > series_n / 2 - 1 - regression_n, "max_lag must be less than (n/2 - 1 - regressors)");

    let series_diff = {
        let mut diff = Vec::new();
        for (i, x) in series[1 ..].iter().enumerate() {
            diff.push(x - series[i - 1]);
        }
        diff
    };
}

pub fn lag_matrix_univariate(series: &[f64], max_lag: usize) -> Vec<Vec<f64>> {
    let series_n = series.len();
    let mut lag_matrix: Vec<Vec<f64>> = Vec::new();
    
    lag_matrix.push(vec![0.0; max_lag]);
    
    for row_idx in 1 ..= series_n - 1 {
        let mut buffer: Vec<f64> = Vec::new();
        for column_idx in 1 ..= max_lag {
            if (row_idx as i32 - column_idx as i32) < 0i32 {
                buffer.push(0.0);
            } else {
                buffer.push(series[row_idx - column_idx]);
            }
        }
        lag_matrix.push(buffer);
    }

    lag_matrix
}

#[test]
fn test_lag_matrix_univariate() {
    let test_series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let lagged_result = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![2.0, 1.0, 0.0],
        vec![3.0, 2.0, 1.0],
        vec![4.0, 3.0, 2.0],
        vec![5.0, 4.0, 3.0],
    ];

    let result = lag_matrix_univariate(&test_series, 3);

    assert!(result == lagged_result);
}

/// Calculates the compounding natural log price for prices stored in a CSV file.
/// 
/// The compounding natural log price is X(t) = ln(P(t)) - ln(P(0))
/// This results in a series that begins at zero and represents the ln of the ratio of P(t) to the initial price.
pub fn compound_price(array_read: &Array2<f64>) -> Array1<f64> {
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