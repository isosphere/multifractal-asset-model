use plotters::prelude::*;

use conv::*;
use ndarray::{Array1};

pub fn plot_simulation(output: &str, all_simulations: &[Vec<f64>], xt: &Array1<f64>) -> Result<(), Box<dyn std::error::Error>> {
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

    // let (min_y, max_y) = {
    //     (
    //         match f64::partial_cmp(sorted_final_positions_min.first().unwrap(), &xt_min) {
    //             Some(Ordering::Less) => { *sorted_final_positions_min.first().unwrap() },
    //             _ => { xt_min }
    //         }*1.10, 
    //         match f64::partial_cmp(sorted_final_positions_max.last().unwrap(), &xt_max) {
    //             Some(Ordering::Greater) => { *sorted_final_positions_max.last().unwrap() },
    //             _ => { xt_max }
    //         }*1.10
    //     )
    // };

    let (min_y, max_y) = (*sorted_final_positions_min.first().unwrap(), *sorted_final_positions_max.last().unwrap());

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
    
    println!("simulation length = {}", all_simulations[0].len());
    for n in 0 .. all_simulations[0].len() {
        let column: Vec<f64> = (0 .. all_simulations.len()).map(|m| all_simulations[m][n]).collect();
        let quartiles = Quartiles::new(&column);
        median.push(quartiles.median());
        // FIXME: hardcoding locations. library also is restricted to 25% and 75% percentiles.
        upper_quartile.push(quartiles.values()[3].value_as::<f64>().unwrap());
        lower_quartile.push(quartiles.values()[1].value_as::<f64>().unwrap());
    }
    
    let mut flag = true;
    let simulation_colour = RGBColor(255, 69, 0).mix(0.2);
    
    for simulation in all_simulations {
        let series: Vec<(f64, f64)> = (0 .. simulation.len()).map(|i| (i.value_as::<f64>().unwrap(), simulation[i])).collect();
        
        match flag {
            false => {
                chart
                .draw_series(
                    LineSeries::new(series, &simulation_colour)
                )?;
            },
            true => {
                chart
                .draw_series(
                    LineSeries::new(series, &simulation_colour)
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