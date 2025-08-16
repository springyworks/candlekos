use anyhow::Result;
use std::path::PathBuf;

fn find_repo_root(mut p: PathBuf) -> Option<PathBuf> {
    loop {
        if p.join(".git").exists() || p.join("Cargo.toml").exists() {
            return Some(p);
        }
        if !p.pop() {
            break;
        }
    }
    None
}

/// Attempt to set the current working directory to the repository root so
/// notebook `:dep` path directives resolve relative to the repo.
///
/// Behavior:
/// - If `NOTEBOOK_PATH` env var is set, use that as starting point.
/// - Otherwise start from the kernel's current directory and walk up.
pub fn set_notebook_cwd() -> Result<()> {
    let here = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let nb_path = std::env::var("NOTEBOOK_PATH").ok().map(PathBuf::from);
    let start = nb_path.unwrap_or_else(|| here.clone());
    if let Some(root) = find_repo_root(start) {
        std::env::set_current_dir(&root)?;
        println!("Notebook CWD set to repository root: {}", root.display());
    } else {
        println!("Repository root not found; leaving CWD: {}", here.display());
    }
    Ok(())
}

/// Converts a string into Morse code.
/// Each letter is separated by a space, and words are separated by a slash ('/').
pub fn to_morse_code(input: &str) -> String {
    let morse_map = [
        ("A", ".-"),
        ("B", "-..."),
        ("C", "-.-."),
        ("D", "-.."),
        ("E", "."),
        ("F", "..-."),
        ("G", "--."),
        ("H", "...."),
        ("I", ".."),
        ("J", ".---"),
        ("K", "-.-"),
        ("L", ".-.."),
        ("M", "--"),
        ("N", "-."),
        ("O", "---"),
        ("P", ".--."),
        ("Q", "--.-"),
        ("R", ".-."),
        ("S", "..."),
        ("T", "-"),
        ("U", "..-"),
        ("V", "...-"),
        ("W", ".--"),
        ("X", "-..-"),
        ("Y", "-.--"),
        ("Z", "--.."),
        ("1", ".----"),
        ("2", "..---"),
        ("3", "...--"),
        ("4", "....-"),
        ("5", "....."),
        ("6", "-...."),
        ("7", "--..."),
        ("8", "---.."),
        ("9", "----."),
        ("0", "-----"),
        (" ", "/"),
    ];

    let morse_map: std::collections::HashMap<_, _> = morse_map.iter().cloned().collect();

    input
        .to_uppercase()
        .chars()
        .filter_map(|c| morse_map.get(&c.to_string()[..]))
        .cloned()
        .collect::<Vec<_>>()
        .join(" ")
}
