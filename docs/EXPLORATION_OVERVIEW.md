# Exploration overview (sandbox)

This fork contains a lightweight sandbox under `0aEXPLORATION/` used to try ideas, notebooks, and small binaries around tensors, visualization, and build plumbing. It is not part of Candle’s public API and should be treated as experimental notes and prototypes.

Highlights
- A few Jupyter notebooks for quick tensor demos and visualization helpers
- Small Rust bins to experiment with display and data flow
- Build notes tying native components to Rust when relevant (see `build/README.md`)

Scope and expectations
- Keep claims modest. Nothing here implies upstream Candle features unless merged there.
- Prefer short demos over large frameworks. Remove stale material regularly.
- When something proves useful, upstream it in smaller focused PRs with tests and docs.

Related docs
- `build/README.md` – why a native build tree can exist locally and how it relates to Candle
