//! Developer utilities ("xtask" pattern) for the Candle workspace.
//!
//! This binary groups together convenience commands used during local
//! development and CI: feature‑space checking, quick test builds, and the
//! `run-file` helper which lets you execute a standalone Rust source file
//! (with a `main`) just by its filesystem path, or open Jupyter notebooks
//! in VS Code. The helper auto‑detects the owning crate, selects/creates the
//! right binary target, and enables any declared `required-features`
//! automatically unless the user already provides an explicit `--features` flag.
//!
//! Typical usage examples:
//!
//! ```bash
//! # List canonical feature combinations for the exploration crate
//! cargo run -p xtask -- list
//!
//! # Check compilation for a small curated set of feature combinations
//! cargo run -p xtask -- check
//!
//! # Run an exploration binary by path (auto features)
//! cargo run -p xtask -- run-file 0aEXPLORATION/gpu_stream_display.rs
//!
//! # Open a Jupyter notebook in VS Code
//! cargo run -p xtask -- run-file 0aEXPLORATION/candle_notebooks/simple_tensors.ipynb
//!
//! # Force additional cargo flags / features and pass program args
//! cargo run -p xtask -- run-file 0aEXPLORATION/gpu_stream_display.rs --release -- --help
//! ```
use anyhow::{Context, Result};
use cargo_metadata::MetadataCommand;
use std::collections::BTreeSet;
use std::process::Command;

// Canonical feature combos we care about in fast CI (exploration crate scope).
// NOTE: We intentionally do not list an invalid combination like `cudnn` without
// `cuda`. If new feature dependencies are added, update `is_valid_combo` below.
const CANONICAL: &[&[&str]] = &[
    &[], // baseline CPU
    &["cuda"],
    &["cuda", "fft"],
    // `cudnn` implies `cuda`; explicit combo keeps visibility.
    &["cuda", "cudnn"],
    &["fft"], // CPU FFT only
];

// Optional core (candle-core) advanced FFT GPU feature combos. Activated when the
// environment variable XTASK_CORE_FFT=1 is set. We do not enumerate powersets here
// to keep runtime bounded; just representative tiers.
const CORE_GPU_FFT_COMBOS: &[&[&str]] = &[
    &["cuda", "fft", "gpu-fft"],
    &["cuda", "fft", "gpu-fft", "gpu-fft-vkfft"],
    &[
        "cuda",
        "fft",
        "gpu-fft",
        "gpu-fft-vkfft",
        "gpu-fft-vkfft-ffi",
    ],
];

/// Entry point dispatching to the various subcommands.
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("list") => list()?,
        Some("check") => check(false)?,
        Some("check-all") => check(true)?,
        Some("test") => test(false)?,
        Some("test-all") => test(true)?,
        Some("lint") => lint()?,
        Some("lint-workspace") => lint_workspace()?,
        Some("comprehensive") => comprehensive_test()?,
        Some("run-file") => {
            if let Some(path) = args.next() {
                run_file(&path, args.collect())?;
            } else {
                anyhow::bail!(
                    "usage: xtask run-file <path/to/file.rs|file.ipynb> [-- <extra cargo args>]"
                );
            }
        }
        Some(cmd) => anyhow::bail!("unknown subcommand: {cmd}"),
        None => {
            eprintln!("xtask commands:");
            eprintln!("  list                    - Show canonical feature combinations");
            eprintln!("  check                   - Check canonical feature combinations");
            eprintln!("  check-all               - Check extended feature combinations");
            eprintln!("  test                    - Build tests for canonical features");
            eprintln!("  test-all                - Build tests for extended features");
            eprintln!("  lint                    - Run clippy on xtask + exploration crates");
            eprintln!("  lint-workspace          - Run clippy on entire workspace");
            eprintln!("  comprehensive           - Run comprehensive workspace testing");
            eprintln!("  run-file <file>         - Run a Rust file or open notebook");
        }
    }
    Ok(())
}

/// Print the fast canonical feature sets we validate in CI.
fn list() -> Result<()> {
    println!("Fast canonical feature sets:");
    for combo in CANONICAL {
        println!("  {combo:?}");
    }
    Ok(())
}

/// Run `cargo check` for either the canonical set or a broader (power) set of
/// feature combinations of the `candle-exploration` crate. When `power` is
/// true we enumerate (bounded) small combinations of features (size ≤ 3) to
/// catch unexpected interactions.
fn check(power: bool) -> Result<()> {
    let meta = MetadataCommand::new().no_deps().exec()?;
    let exploration = meta
        .packages
        .iter()
        .find(|p| p.name == "candle-exploration")
        .context("candle-exploration crate not found in workspace")?;

    let feature_space: BTreeSet<String> = exploration.features.keys().cloned().collect();
    let combos: Vec<Vec<String>> = if power {
        // Limited powerset (exclude default empty set is still included, up to size 3 to limit explosion)
        let feats: Vec<String> = feature_space
            .into_iter()
            .filter(|f| f != "default")
            .collect();
        let mut acc = Vec::new();
        for mask in 0..(1u32 << feats.len()).min(1 << 6) {
            // guard: limit to first 6 features if it grows
            let mut v = Vec::new();
            for (i, f) in feats.iter().enumerate() {
                if (mask & (1 << i)) != 0 {
                    v.push(f.clone());
                }
            }
            // prune large sets ( >3 ) to keep runtime sane
            if v.len() <= 3 && is_valid_combo(&v) {
                acc.push(v);
            }
        }
        acc
    } else {
        CANONICAL
            .iter()
            .map(|c| c.iter().map(|s| (*s).to_string()).collect())
            .collect()
    };

    for combo in combos {
        run_check(&combo)?;
    }
    Ok(())
}

/// Internal helper to perform a single `cargo check` invocation for the given
/// feature list, also optionally validating advanced core FFT feature combos.
fn run_check(features: &[String]) -> Result<()> {
    let feat_arg = if features.is_empty() {
        "(none)".to_string()
    } else {
        features.join(",")
    };
    println!("==> cargo check --features {feat_arg}");
    let status = if features.is_empty() {
        Command::new("cargo").arg("check").status()?
    } else {
        Command::new("cargo")
            .arg("check")
            .arg("--features")
            .arg(&feat_arg)
            .status()?
    };
    if !status.success() {
        anyhow::bail!("check failed for features: {feat_arg}");
    }
    // Optionally also check candle-core with advanced GPU FFT combos (env gated)
    if std::env::var("XTASK_CORE_FFT").ok().as_deref() == Some("1") && !features.is_empty() {
        for combo in CORE_GPU_FFT_COMBOS {
            // Skip if combo not superset-compatible with current exploration features (requires cuda+fft at minimum)
            if combo
                .iter()
                .all(|f| features.contains(&f.to_string()) || !["cuda", "fft"].contains(f))
            {
                let core_feat_arg = combo.join(",");
                println!("   -> candle-core check (core FFT) --features {core_feat_arg}");
                let status = Command::new("cargo")
                    .arg("check")
                    .arg("-p")
                    .arg("candle-core")
                    .arg("--features")
                    .arg(&core_feat_arg)
                    .status()?;
                if !status.success() {
                    anyhow::bail!("candle-core check failed for features: {core_feat_arg}");
                }
            }
        }
    }
    Ok(())
}

/// Build (but do not run) tests for the exploration crate across a selection
/// of feature combinations. Mirrors `check` logic but uses `cargo test --no-run`.
fn test(power: bool) -> Result<()> {
    let meta = MetadataCommand::new().no_deps().exec()?;
    let exploration = meta
        .packages
        .iter()
        .find(|p| p.name == "candle-exploration")
        .context("candle-exploration crate not found in workspace")?;
    let feature_space: BTreeSet<String> = exploration.features.keys().cloned().collect();
    let combos: Vec<Vec<String>> = if power {
        let feats: Vec<String> = feature_space
            .into_iter()
            .filter(|f| f != "default")
            .collect();
        let mut acc = Vec::new();
        for mask in 0..(1u32 << feats.len()).min(1 << 6) {
            let mut v = Vec::new();
            for (i, f) in feats.iter().enumerate() {
                if (mask & (1 << i)) != 0 {
                    v.push(f.clone());
                }
            }
            if v.len() <= 3 && is_valid_combo(&v) {
                acc.push(v);
            }
        }
        acc
    } else {
        CANONICAL
            .iter()
            .map(|c| c.iter().map(|s| (*s).to_string()).collect())
            .collect()
    };
    for combo in combos {
        run_tests(&combo)?;
    }
    Ok(())
}

/// Validate a feature combination for the exploration crate. This is a
/// lightweight manual dependency check so that `check-all`/`test-all` do not
/// waste cycles (or fail noisily) on invalid sets like `cudnn` without `cuda`.
fn is_valid_combo(features: &[String]) -> bool {
    // cudnn requires cuda
    if features.iter().any(|f| f == "cudnn") && !features.iter().any(|f| f == "cuda") {
        return false;
    }
    true
}

/// Internal helper to perform a single `cargo test --no-run` invocation for a
/// given feature list (plus optional core FFT combos).
fn run_tests(features: &[String]) -> Result<()> {
    let feat_arg = if features.is_empty() {
        "(none)".to_string()
    } else {
        features.join(",")
    };
    println!("==> cargo test --no-run --features {feat_arg}");
    let status = if features.is_empty() {
        Command::new("cargo")
            .arg("test")
            .arg("--no-run")
            .arg("-p")
            .arg("candle-exploration")
            .status()?
    } else {
        Command::new("cargo")
            .arg("test")
            .arg("--no-run")
            .arg("-p")
            .arg("candle-exploration")
            .arg("--features")
            .arg(&feat_arg)
            .status()?
    };
    if !status.success() {
        anyhow::bail!("test build failed for features: {feat_arg}");
    }
    if std::env::var("XTASK_CORE_FFT").ok().as_deref() == Some("1") && !features.is_empty() {
        for combo in CORE_GPU_FFT_COMBOS {
            if combo
                .iter()
                .all(|f| features.contains(&f.to_string()) || !["cuda", "fft"].contains(f))
            {
                let core_feat_arg = combo.join(",");
                println!("   -> candle-core test build (core FFT) --features {core_feat_arg}");
                let status = Command::new("cargo")
                    .arg("test")
                    .arg("--no-run")
                    .arg("-p")
                    .arg("candle-core")
                    .arg("--features")
                    .arg(&core_feat_arg)
                    .status()?;
                if !status.success() {
                    anyhow::bail!("candle-core test build failed for features: {core_feat_arg}");
                }
            }
        }
    }
    Ok(())
}

/// Run clippy (lint) across a curated set of crates with `-D warnings` so the
/// CI surface stays clean. By default we lint `xtask` and the exploration
/// crate. Set `XTASK_LINT_CORE=1` to also lint `candle-core` (can be slower).
fn lint() -> Result<()> {
    let mut cmd = Command::new("cargo");
    cmd.arg("clippy")
        .arg("-p")
        .arg("xtask")
        .arg("-p")
        .arg("candle-exploration")
        .arg("--all-targets")
        .arg("--")
        .arg("-D")
        .arg("warnings");
    let status = cmd.status()?;
    if !status.success() {
        anyhow::bail!("clippy failed (xtask + exploration)");
    }
    if std::env::var("XTASK_LINT_CORE").ok().as_deref() == Some("1") {
        let mut core = Command::new("cargo");
        core.arg("clippy")
            .arg("-p")
            .arg("candle-core")
            .arg("--all-targets")
            .arg("--")
            .arg("-D")
            .arg("warnings");
        let status = core.status()?;
        if !status.success() {
            anyhow::bail!("clippy failed (candle-core)");
        }
    }
    Ok(())
}

/// Execute an arbitrary Rust source file containing a `main` function by
/// determining its owning workspace crate and resolving/creating an
/// appropriate binary target. Also handles Jupyter notebook (.ipynb) files
/// by opening them in VS Code's notebook viewer.
///
/// Behavior summary:
/// * If the path is a `.ipynb` file -> open in VS Code notebook viewer.
/// * If the path exactly matches a declared `[[bin]]` target -> run it.
/// * If it lives under `src/bin/<name>.rs` -> cargo auto‑discovers it.
/// * If it is the crate root `src/main.rs` -> run the crate.
/// * Otherwise we copy it temporarily to `src/bin/__xtask_temp_<stem>.rs` and
///   run that, cleaning up afterwards.
/// * Any `required-features` for the target are auto‑enabled unless user
///   passed an explicit `--features ...` in the cargo flag section.
/// * Extra cargo flags (`--release`, `--features`, etc.) are parsed before a
///   `--` separator; tokens after `--` are forwarded to the program.
fn run_file(path_arg: &str, extra: Vec<String>) -> Result<()> {
    use std::path::{Path, PathBuf};
    // NOTE: we previously cloned `extra` for debugging; removed to avoid
    // unused variable warnings.
    let path = Path::new(path_arg)
        .canonicalize()
        .with_context(|| format!("cannot canonicalize {path_arg}"))?;

    // Handle Jupyter notebook files by opening in VS Code
    if let Some(ext) = path.extension() {
        if ext == "ipynb" {
            eprintln!(
                "[xtask] opening notebook {} in current VS Code workspace",
                path.display()
            );

            // Try multiple approaches to open in current VS Code instance
            let mut cmd = Command::new("code");

            // If we're running inside VS Code (detected by environment variables),
            // use --add to add to current workspace. Otherwise use --reuse-window.
            if std::env::var("VSCODE_INJECTION").is_ok()
                || std::env::var("VSCODE_PID").is_ok()
                || std::env::var("TERM_PROGRAM").as_deref() == Ok("vscode")
            {
                cmd.arg("--add");
            } else {
                cmd.arg("--reuse-window");
            }

            let status = cmd.arg(&path).status()?;
            if !status.success() {
                anyhow::bail!("failed to open notebook in VS Code");
            }
            return Ok(());
        }
    }
    // Separate cargo-level flags (like --release, --features <...>) from program args.
    let mut cargo_flags: Vec<String> = Vec::new();
    let mut program_args: Vec<String> = Vec::new();
    let mut iter = extra.into_iter();
    while let Some(tok) = iter.next() {
        if tok == "--" {
            // explicit separator forces rest as program args
            program_args.extend(iter);
            break;
        } else if tok == "--release" {
            cargo_flags.push(tok);
        } else if tok == "--features" {
            if let Some(val) = iter.next() {
                cargo_flags.push(tok);
                cargo_flags.push(val);
            } else {
                anyhow::bail!("--features requires a value");
            }
        } else if tok.starts_with("--features=") {
            cargo_flags.push(tok);
        } else {
            program_args.push(tok);
        }
    }

    // Load workspace metadata to locate owning package.
    let meta = MetadataCommand::new().exec()?;
    // Find the package whose directory is an ancestor of the file path (choose deepest ancestor).
    let mut best: Option<(&cargo_metadata::Package, usize)> = None;
    for pkg in &meta.packages {
        let manifest_dir = PathBuf::from(&pkg.manifest_path)
            .parent()
            .unwrap()
            .canonicalize()?;
        if path.starts_with(&manifest_dir) {
            let depth = manifest_dir.components().count();
            match &best {
                Some((_, best_depth)) if *best_depth >= depth => {}
                _ => best = Some((pkg, depth)),
            }
        }
    }
    let (pkg, _depth) = best.context("could not find a workspace crate containing the file")?;

    // Determine if the file corresponds to an existing binary target declared in Cargo.toml.
    // Strategy: compare canonical path to each target's path.
    let mut matched_bin: Option<String> = None;
    for tgt in &pkg.targets {
        if tgt.kind.iter().any(|k| k == "bin") {
            let tgt_path = std::path::PathBuf::from(&tgt.src_path).canonicalize().ok();
            if Some(&path) == tgt_path.as_ref() {
                matched_bin = Some(tgt.name.clone());
                break;
            }
        }
    }

    // If not matched but file lies in src/bin or crate root with main, cargo can run via --bin <stem> or via rustc invocation? Prefer ephemeral cargo run --bin if in [[bin]]; else use cargo run --package pkg --bin <auto-temp> by creating a temp wrapper? Simpler: if not matched and file is inside crate dir, build via --package and --bin path file access using cargo's support for `cargo run --package pkg --bin name` only for declared binaries. Fallback: use `rustc` directly with correct -L paths by leveraging `cargo rustc -- -Z unstable-options --pretty=...` is complex.
    // Instead, if not declared and file is at <crate>/src/bin/<stem>.rs, cargo will auto-discover it WITHOUT needing declaration.
    if matched_bin.is_none() {
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            // Check for auto bin discovery condition.
            // Condition: path == <crate>/src/bin/<stem>.rs
            let crate_dir = PathBuf::from(&pkg.manifest_path)
                .parent()
                .unwrap()
                .canonicalize()?;
            let auto_bin = crate_dir.join("src").join("bin").join(format!("{stem}.rs"));
            if auto_bin == path {
                matched_bin = Some(stem.to_string());
            }
        }
    }

    let mut auto_features: Vec<String> = Vec::new();
    if let Some(bin_name) = matched_bin.as_ref() {
        if let Some(tgt) = pkg.targets.iter().find(|t| t.name == *bin_name) {
            if !tgt.required_features.is_empty() {
                // Only auto-apply if user has not explicitly provided --features
                let user_specified = cargo_flags
                    .iter()
                    .any(|e| e == "--features" || e.starts_with("--features="));
                if !user_specified {
                    auto_features = tgt.required_features.clone();
                }
            }
        }
    }

    let manifest_path_buf = std::path::PathBuf::from(&pkg.manifest_path);
    let crate_dir_buf = manifest_path_buf.parent().unwrap().to_path_buf();
    let crate_dir = &crate_dir_buf;

    let mut cmd = Command::new("cargo");
    if let Some(bin) = matched_bin {
        eprintln!(
            "[xtask] running existing bin '{}' in crate '{}'",
            bin, pkg.name
        );
        cmd.arg("run")
            .arg("-p")
            .arg(&pkg.name)
            .arg("--bin")
            .arg(&bin);
        if !auto_features.is_empty() && !cargo_flags.iter().any(|f| f.starts_with("--features")) {
            cmd.arg("--features").arg(auto_features.join(","));
        }
        for f in &cargo_flags {
            cmd.arg(f);
        }
    } else {
        let main_rs = crate_dir.join("src").join("main.rs");
        if path == main_rs.canonicalize().unwrap_or(main_rs.clone()) {
            eprintln!("[xtask] running crate '{}' (src/main.rs)", pkg.name);
            cmd.arg("run").arg("-p").arg(&pkg.name);
        } else {
            // As a last resort, create a temporary copy under src/bin and run it.
            use std::fs;
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("runfile");
            let bin_dir = crate_dir.join("src").join("bin");
            fs::create_dir_all(&bin_dir)?;
            let temp_stem = format!("__xtask_temp_{stem}");
            let temp_path = bin_dir.join(format!("{temp_stem}.rs"));
            if temp_path.exists() {
                std::fs::remove_file(&temp_path)?;
            }
            let contents = fs::read_to_string(&path)?;
            fs::write(&temp_path, contents)?;
            eprintln!(
                "[xtask] created temporary bin {} to run file {}",
                temp_path.display(),
                path.display()
            );
            cmd.arg("run")
                .arg("-p")
                .arg(&pkg.name)
                .arg("--bin")
                .arg(&temp_stem);
            if !auto_features.is_empty() && !cargo_flags.iter().any(|f| f.starts_with("--features"))
            {
                cmd.arg("--features").arg(auto_features.join(","));
            }
            for f in &cargo_flags {
                cmd.arg(f);
            }
            if !program_args.is_empty() {
                cmd.arg("--");
                for a in &program_args {
                    cmd.arg(a);
                }
            }
            let status = cmd.status()?;
            std::fs::remove_file(&temp_path)?;
            if !status.success() {
                anyhow::bail!("cargo run failed");
            }
            return Ok(());
        }
    }

    if !program_args.is_empty() {
        cmd.arg("--");
        for a in &program_args {
            cmd.arg(a);
        }
    }

    let status = cmd.status()?;
    if !status.success() {
        anyhow::bail!("cargo run failed");
    }
    Ok(())
}

/// Run clippy across the entire workspace with warnings treated as warnings only.
/// This provides a comprehensive view of code quality across all crates.
fn lint_workspace() -> Result<()> {
    println!("Running clippy across entire workspace...");
    let mut cmd = Command::new("cargo");
    cmd.arg("clippy").arg("--workspace").arg("--all-targets");

    let status = cmd.status()?;
    if !status.success() {
        println!("Note: clippy found issues but this is informational only");
    } else {
        println!("✓ Workspace clippy completed successfully");
    }
    Ok(())
}

/// Run comprehensive testing across the workspace including:
/// - Basic compilation check
/// - Feature combination testing  
/// - Test builds
/// - Workspace clippy (informational)
/// - Documentation build
/// - Format check
fn comprehensive_test() -> Result<()> {
    println!("🚀 Starting comprehensive Candle workspace testing");
    println!("==================================================");

    let mut test_count = 0;
    let mut passed_count = 0;

    // Helper function for running tests with status tracking
    let mut run_test = |name: &str, test_fn: fn() -> Result<()>| {
        test_count += 1;
        print!("📋 Testing: {name} ... ");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        match test_fn() {
            Ok(_) => {
                println!("✅ PASS");
                passed_count += 1;
            }
            Err(e) => {
                println!("❌ FAIL: {e}");
            }
        }
    };

    // 1. Basic workspace check
    run_test("Workspace compilation", || -> Result<()> {
        let status = Command::new("cargo")
            .arg("check")
            .arg("--workspace")
            .status()?;
        if status.success() {
            Ok(())
        } else {
            anyhow::bail!("compilation failed")
        }
    });

    // 2. Canonical feature combinations
    run_test("Canonical feature combinations", || check(false));

    // 3. Extended feature combinations (if environment variable set)
    if std::env::var("XTASK_COMPREHENSIVE").ok().as_deref() == Some("1") {
        run_test("Extended feature combinations", || check(true));
    }

    // 4. Test builds
    run_test("Test builds (canonical)", || test(false));

    // 5. Extended test builds (if environment variable set)
    if std::env::var("XTASK_COMPREHENSIVE").ok().as_deref() == Some("1") {
        run_test("Extended test builds", || test(true));
    }

    // 6. Workspace clippy (informational)
    run_test("Workspace clippy (informational)", || -> Result<()> {
        let _status = Command::new("cargo")
            .arg("clippy")
            .arg("--workspace")
            .arg("--all-targets")
            .status()?;
        // Always succeed for clippy in comprehensive mode
        println!("  (clippy warnings are informational only)");
        Ok(())
    });

    // 7. Documentation build
    run_test("Documentation build", || -> Result<()> {
        let status = Command::new("cargo")
            .arg("doc")
            .arg("--workspace")
            .arg("--no-deps")
            .status()?;
        if status.success() {
            Ok(())
        } else {
            anyhow::bail!("doc build failed")
        }
    });

    // 8. Format check
    run_test("Format check", || -> Result<()> {
        let status = Command::new("cargo")
            .arg("fmt")
            .arg("--all")
            .arg("--")
            .arg("--check")
            .status()?;
        if status.success() {
            Ok(())
        } else {
            anyhow::bail!("formatting issues found")
        }
    });

    // 9. Core FFT features (if environment variable set)
    if std::env::var("XTASK_CORE_FFT").ok().as_deref() == Some("1") {
        run_test("Core FFT feature testing", || {
            std::env::set_var("XTASK_CORE_FFT", "1");
            check(false)
        });
    }

    // Summary
    println!();
    println!("📊 COMPREHENSIVE TEST SUMMARY");
    println!("=============================");
    println!("Total tests: {test_count}");
    println!("Passed: {passed_count} ✅");
    println!("Failed: {} ❌", test_count - passed_count);

    let success_rate = (passed_count * 100) / test_count;
    println!("Success rate: {success_rate}%");

    if passed_count == test_count {
        println!();
        println!("🎉 All tests passed! Workspace is healthy.");
        Ok(())
    } else {
        println!();
        println!("⚠️  Some tests failed. See output above for details.");
        println!();
        println!("Environment variables for extended testing:");
        println!("  XTASK_COMPREHENSIVE=1    Enable extended feature/test combinations");
        println!("  XTASK_CORE_FFT=1        Enable core FFT feature testing");
        anyhow::bail!(
            "{} out of {} tests failed",
            test_count - passed_count,
            test_count
        )
    }
}
