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

/// Generate smart dependency paths for evcxr notebooks that work from any location
pub fn notebook_deps() -> Result<String> {
    let here = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let nb_path = std::env::var("NOTEBOOK_PATH").ok().map(PathBuf::from);
    let start = nb_path.unwrap_or_else(|| here.clone());

    if let Some(root) = find_repo_root(start.clone()) {
        let rel_to_root = start.strip_prefix(&root).unwrap_or(&start);
        let depth = rel_to_root.components().count();
        let back_to_root = "../".repeat(depth);

        let candle_core_path = format!("{back_to_root}candle-core");
        let notebooks_path =
            format!("{back_to_root}0aEXPLORATION/research/notebooks/candle_notebooks");

        Ok(format!(
            r#":dep candle-core = {{ path = "{candle_core_path}", default-features = false }}
:dep anyhow = "1"
:dep candle-notebooks = {{ path = "{notebooks_path}" }}"#
        ))
    } else {
        // Fallback to absolute paths
        let default_root = PathBuf::from("/home/rustuser/projects/rust/from_github/candle");
        let candle_core = default_root.join("candle-core");
        let notebooks = default_root.join("0aEXPLORATION/research/notebooks/candle_notebooks");

        Ok(format!(
            r#":dep candle-core = {{ path = "{}", default-features = false }}
:dep anyhow = "1"
:dep candle-notebooks = {{ path = "{}" }}"#,
            candle_core.display(),
            notebooks.display()
        ))
    }
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
