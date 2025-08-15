//! Code Overview Utility - Lists all exploration files with their documentation descriptions
//! Extracts and displays the first two lines of Rust doc comments from each source file for easy browsing

use std::fs;
use std::path::Path;

fn main() -> std::io::Result<()> {
    println!("🗂️  Candle Exploration Code Overview\n");
    
    let exploration_dir = Path::new(".");
    let mut files = Vec::new();
    
    // Collect all .rs files
    for entry in fs::read_dir(exploration_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
                files.push(filename.to_string());
            }
        }
    }
    
    // Sort files for consistent output
    files.sort();
    
    // Display each file with its description
    for filename in files {
        if let Ok(content) = fs::read_to_string(&filename) {
            let lines: Vec<&str> = content.lines().collect();
            
            // Look for the first two //! doc comment lines
            let mut doc_lines = Vec::new();
            for line in lines.iter().take(10) { // Only check first 10 lines
                let trimmed = line.trim();
                if trimmed.starts_with("//!") {
                    doc_lines.push(trimmed.trim_start_matches("//!").trim());
                } else if !trimmed.is_empty() && !trimmed.starts_with("//") {
                    break; // Stop at first non-comment line
                }
            }
            
            println!("📄 **{filename}**");
            if doc_lines.len() >= 2 {
                println!("   {}", doc_lines[0]);
                println!("   {}", doc_lines[1]);
            } else if doc_lines.len() == 1 {
                println!("   {}", doc_lines[0]);
            } else {
                println!("   No documentation found");
            }
            println!();
        }
    }
    
    Ok(())
}
