use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use facefeature::default_detector;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let first = arguments.next().ok_or_else(usage)?;
    if matches!(first.as_str(), "-h" | "--help" | "help") {
        println!("{}", usage());
        return Ok(());
    }

    let image_path = if first == "detect" {
        arguments.next().map(PathBuf::from).ok_or_else(usage)?
    } else {
        PathBuf::from(first)
    };
    let mut pretty = false;
    for argument in arguments {
        match argument.as_str() {
            "--pretty" => pretty = true,
            _ => return Err(format!("unknown argument: {argument}\n\n{}", usage())),
        }
    }

    let detector = default_detector().map_err(|error| error.to_string())?;
    let result = detector
        .detect_path(&image_path)
        .map_err(|error| error.to_string())?;
    let json = if pretty {
        serde_json::to_string_pretty(&result)
    } else {
        serde_json::to_string(&result)
    }
    .map_err(|error| format!("could not serialize result: {error}"))?;
    println!("{json}");
    Ok(())
}

fn usage() -> String {
    "Usage: facefeature [detect] <IMAGE> [--pretty]\n\nDetect local face landmarks and geometry in an image.".to_owned()
}
