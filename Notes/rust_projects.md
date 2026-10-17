Cargo commands to download are crates and create macros files.  
Need to run on new projects before running on projects. Just keeping these here. Forgetting it these days.
### Commands Executed

  1. cargo check
    cargo check

  2. cargo check --all-targets
    cargo check --all-targets

  3. cargo check --all-targets --features kvm
    cargo check --all-targets --features kvm

  ──────
  ### What These Commands Do & Why LSP Needs Them

  Rust language servers (such as rust-analyzer) rely on Cargo to prepare files before features like code completion, macro expansion, and jump-to-definition can work:

  1. Download External Crates:
      • Downloads source code and metadata for all dependencies and dev-dependencies from crates.io into ~/.cargo/registry/.
      • Allows the LSP to navigate into definitions and traits of third-party crates.
  2. Execute Build Scripts (build.rs):
      • Runs crate build scripts that generate Rust code into target/debug/build/<crate>-<hash>/out/ ($OUT_DIR).
      • Enables the LSP to resolve files included via include!(concat!(env!("OUT_DIR"), "/...")) (such as generated C bindings, syscall wrappers, or protocol definitions).
  3. Compile Procedural Macro Dynamic Libraries:
      • Compiles proc-macro crates (e.g., serde_derive, clap_derive, thiserror-impl) into shared objects (.so) under target/debug/deps/.
      • Allows rust-analyzer to expand derive macros and custom attributes in real time.
  4. Generate Compiler Metadata & Type Indices:
      • Generates .rmeta crate metadata used by rust-analyzer for type inference, cross-crate symbol resolution, and hover documentation.
      • Activating --features kvm ensures that conditional compilation flags (#[cfg(feature = "kvm")]) across vmm, hypervisor, and devices are active and analyzed.
