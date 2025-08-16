use std::fs;
use std::path::PathBuf;

// Basic integration smoke tests for `xtask run-file`.
// These tests intentionally avoid heavy compilation permutations; they
// validate that: (1) a standalone temp file can be executed, (2) program
// arguments after `--` are forwarded, and (3) temporary bin cleanup occurs.
//
// NOTE: Running cargo within tests can be slow; keep assertions minimal and
// guard with environment variable to allow opting out (XTASK_RUN_FILE_TESTS=1).

fn cargo_bin() -> PathBuf {
    assert_cmd::cargo::cargo_bin("xtask")
}

#[test]
fn run_temp_file_executes_and_cleans() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("XTASK_RUN_FILE_TESTS").ok().as_deref() != Some("1") {
        eprintln!("skipping run-file integration test (set XTASK_RUN_FILE_TESTS=1 to enable)");
        return Ok(());
    }
    let tmpdir = tempfile::tempdir()?;
    let file = tmpdir.path().join("hello_temp.rs");
    fs::write(&file, r#"fn main(){ println!("HELLO_INTEGRATION"); }"#)?;

    // Invoke xtask run-file on this standalone file; it should create a temp bin inside the
    // owning crate (xtask itself or nearest) – since the file is outside any crate, this will
    // error. So instead we copy into the exploration crate directory to ensure ownership.
    // Choose candle-exploration crate root (assumed present).
    let exploration_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap() // move out of xtask/
        .join("candle-exploration");
    assert!(exploration_root.exists());
    let target_file = exploration_root.join("standalone_temp_run.rs");
    fs::write(
        &target_file,
        r#"fn main(){ println!("XTASK_ARG_FORWARD:{}", std::env::args().skip(1).next().unwrap_or_default()); }"#,
    )?;

    let mut cmd = assert_cmd::Command::new(cargo_bin());
    cmd.arg("run-file")
        .arg(target_file.to_string_lossy().to_string())
        .arg("--")
        .arg("ARG42");
    cmd.assert()
        .success()
        .stdout(predicates::str::contains("XTASK_ARG_FORWARD:ARG42"));

    // Ensure temp file cleaned (pattern __xtask_temp_*)
    let bin_dir = exploration_root.join("src").join("bin");
    if bin_dir.exists() {
        for entry in fs::read_dir(&bin_dir)? {
            let p = entry?.path();
            if p.file_name()
                .and_then(|s| s.to_str())
                .map(|n| n.starts_with("__xtask_temp_"))
                .unwrap_or(false)
            {
                panic!("temporary bin not cleaned: {}", p.display());
            }
        }
    }

    fs::remove_file(target_file)?;
    Ok(())
}
