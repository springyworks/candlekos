/// Notebook testing utilities for Rust/evcxr notebooks
use anyhow::Result;
use std::path::Path;
use std::process::Command;

/// Execute a Rust notebook and capture output/errors
pub fn test_notebook(notebook_path: &Path) -> Result<()> {
    println!("Testing notebook: {}", notebook_path.display());
    
    // Use jupyter nbconvert to execute the notebook
    let output = Command::new("jupyter")
        .args(&[
            "nbconvert",
            "--to", "notebook",
            "--execute",
            "--output", "/tmp/test_output.ipynb",
            notebook_path.to_str().unwrap()
        ])
        .output()?;
    
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Notebook execution failed: {}", stderr);
    }
    
    println!("✓ Notebook executed successfully");
    Ok(())
}

/// Test all notebooks in a directory
pub fn test_notebooks_in_dir(dir: &Path) -> Result<usize> {
    let mut failed = 0;
    let mut total = 0;
    
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("ipynb") {
            total += 1;
            if let Err(e) = test_notebook(&path) {
                eprintln!("❌ Failed: {} - {}", path.display(), e);
                failed += 1;
            }
        }
    }
    
    println!("Notebook Test Results: {}/{} passed", total - failed, total);
    
    if failed > 0 {
        anyhow::bail!("{} notebooks failed", failed);
    }
    
    Ok(total)
}

/// Extract and validate specific cell outputs for regression testing
pub fn validate_notebook_outputs(notebook_path: &Path, expected_outputs: &[&str]) -> Result<()> {
    // Parse executed notebook JSON and validate specific cell outputs
    let content = std::fs::read_to_string(notebook_path)?;
    let notebook: serde_json::Value = serde_json::from_str(&content)?;
    
    if let Some(cells) = notebook["cells"].as_array() {
        for (i, expected) in expected_outputs.iter().enumerate() {
            if let Some(cell) = cells.get(i) {
                if let Some(outputs) = cell["outputs"].as_array() {
                    // Simple text matching - could be more sophisticated
                    let output_text = format!("{:?}", outputs);
                    if !output_text.contains(expected) {
                        anyhow::bail!("Cell {} output doesn't contain expected text: {}", i, expected);
                    }
                }
            }
        }
    }
    
    Ok(())
}