// Bootstrap helper to find candle-notebooks path from any location
// This file should be at the repo root to help notebooks find dependencies

use std::env;
use std::path::{Path, PathBuf};

fn find_repo_root() -> Option<PathBuf> {
    let mut current = env::current_dir().ok()?;
    loop {
        if current.join("find_candle_notebooks.rs").exists() 
            || current.join(".git").exists() 
            || current.join("Cargo.toml").exists() {
            return Some(current);
        }
        if !current.pop() {
            break;
        }
    }
    None
}

fn main() {
    if let Some(root) = find_repo_root() {
        let notebooks_path = root.join("0aEXPLORATION/research/notebooks/candle_notebooks");
        println!("{}", notebooks_path.display());
    } else {
        eprintln!("Repository root not found");
        std::process::exit(1);
    }
}