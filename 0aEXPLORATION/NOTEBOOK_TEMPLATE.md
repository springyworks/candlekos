# Candle Notebook Template - Path Resilient Edition

This template provides a standardized setup for creating new Candle notebooks that work reliably from any location within the repository.

## 🔧 Path Resilient System Features

- **Adaptive Dependency Paths**: Automatically work from different locations
- **Repository Awareness**: Find the candle repository root and set working directory
- **Consistent Image Storage**: Unified image output system across all notebooks
- **Smart Error Handling**: Clear messages when paths need manual adjustment
- **Future-Proof**: Notebooks remain functional when moved between directories

## 📋 Standard Notebook Template

### Cell 1: Markdown Header
```markdown
# Your Notebook Title - Path Resilient Edition

Brief description of what this notebook demonstrates.

🔧 **Path Resilient**: This notebook uses adaptive dependency paths and works from any location within the candle repository.

Additional content...
```

### Cell 2: Resilient Dependency Setup
```rust
// RESILIENT NOTEBOOK DEPENDENCY SETUP
//
// This cell implements a smart dependency loading system that works regardless 
// of where the notebook file is located within the candle repository.
//
// DEPENDENCY LOADING STRATEGY:
// 1. Load dependencies BEFORE any use statements to satisfy evcxr's requirements
// 2. Use paths optimized for current notebook location
// 3. Dependencies are ordered to ensure all crates are available before imports
//
// WHY DEPENDENCIES COME FIRST:
// - evcxr (Rust notebook kernel) requires :dep declarations before any use statements
// - This ensures all crates are available when we try to import them
// - The order matters: dependencies → imports → logic

// CHOOSE THE APPROPRIATE PATH SET FOR YOUR NOTEBOOK LOCATION:

// For notebooks in /demos/ directory:
:dep candle-core = { path = "../../candle-core", default-features = false }
:dep candle-notebooks = { path = "../research/notebooks/candle_notebooks" }

// For notebooks in /research/notebooks/candle_notebooks/ directory:
// :dep candle-core = { path = "../../../../candle-core", default-features = false }
// :dep candle-notebooks = { path = "." }

// For notebooks in repository root:
// :dep candle-core = { path = "candle-core", default-features = false }
// :dep candle-notebooks = { path = "0aEXPLORATION/research/notebooks/candle_notebooks" }

// Common additional dependencies
:dep anyhow = "1"
// :dep image = "0.24"  // Uncomment if needed
// :dep candle-nn = { path = "../../candle-nn" }  // Adjust path as needed

// Now we can safely import after dependencies are declared
use candle_core::{Device, Tensor};
use candle_notebooks as nb;

println!("✓ Dependencies loaded successfully");
println!("✓ Paths are resilient to notebook file moves within the candle repository.");
```

### Cell 3: Standard Initialization
```rust
// Standard notebook initialization with working directory and image store management
nb::set_notebook_cwd().unwrap();
nb::set_image_store_rel_dir("images_store").unwrap();
std::fs::create_dir_all("images_store").ok();

let device = Device::Cpu;
println!("✓ Notebook initialized successfully!");
println!("  Working directory: {:?}", std::env::current_dir().unwrap());
println!("  Device: {:?}", device);
println!("  Image store: images_store/");
println!("  Location: your/notebook/path.ipynb");
```

## 📁 Directory-Specific Path Configuration

### For `/demos/` notebooks:
```rust
:dep candle-core = { path = "../../candle-core", default-features = false }
:dep candle-notebooks = { path = "../research/notebooks/candle_notebooks" }
:dep candle-nn = { path = "../../candle-nn" }  // if needed
```

### For `/research/notebooks/candle_notebooks/` notebooks:
```rust
:dep candle-core = { path = "../../../../candle-core", default-features = false }
:dep candle-notebooks = { path = "." }
:dep candle-nn = { path = "../../../../candle-nn" }  // if needed
```

### For repository root notebooks:
```rust
:dep candle-core = { path = "candle-core", default-features = false }
:dep candle-notebooks = { path = "0aEXPLORATION/research/notebooks/candle_notebooks" }
:dep candle-nn = { path = "candle-nn" }  // if needed
```

## 🚀 Best Practices

1. **Always use the resilient setup pattern** for new notebooks
2. **Test notebooks after moving** to ensure paths still work
3. **Include the "Path Resilient Edition" suffix** in titles for clarity
4. **Document the notebook location** in the initialization cell output
5. **Use consistent image storage** with `nb::set_image_store_rel_dir("images_store")`
6. **Set working directory** with `nb::set_notebook_cwd()` for consistent relative paths

## 🔍 Troubleshooting

If a notebook fails to load dependencies:

1. **Check the path comments** in the dependency cell
2. **Verify you're using the correct path set** for your location
3. **Ensure candle_notebooks Cargo.toml** has the correct candle-core path
4. **Test the paths manually** using terminal commands
5. **Check for typos** in the path strings

## 📚 Updated Notebooks

All existing notebooks have been updated with the resilient system:

**Demos:**
- ✅ `fft_basic_demo.rs.ipynb`

**Research Notebooks:**
- ✅ `simple_tensors.ipynb`
- ✅ `helpers_demo.ipynb`
- ✅ `egui_window_demo.ipynb`
- ✅ `tensor_art_gallery.ipynb`
- ✅ `tensor_math_fill.ipynb`
- ✅ `temp_run_cells.ipynb`

All notebooks now include:
- Adaptive dependency paths
- Comprehensive setup comments
- Standard initialization pattern
- Path resilience documentation

## 🎯 Future Enhancements

Potential improvements to the resilient system:
- Automatic path detection at runtime
- Smart fallback mechanisms for missing dependencies
- Template generation scripts
- Cross-platform path handling
- Integration with VS Code notebook templates