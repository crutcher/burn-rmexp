use clap::Parser;
use indicatif::{ProgressBar, ProgressDrawTarget};

#[derive(Parser, Debug)]
#[command(long_about = None)]
pub struct Args {
    /// Example arg.
    #[arg(long, default_value = "false")]
    pub example: bool,

    /// Show progress bar.
    #[arg(short, long, default_value_t = false)]
    pub progress: bool,
}

fn main() {
    let args = Args::parse();
    println!("{args:#?}");

    let bar = ProgressBar::new_spinner();
    if !args.progress {
        bar.set_draw_target(ProgressDrawTarget::hidden());
    }

    let x = tbs_common::add(1, 2);
    println!("Hello, world!: {x}");
}
