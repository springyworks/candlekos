<!--
Auto-maintained task list.
Numbering rules:
  Level 1: Capital letters (A., B., ...)
  Level 2: Decimal numbers (1., 2., ...)
  Level 3: Dotted numbers (1.1, 1.2, ...)
Anchors:
  Each Level 2 item has an HTML id like A1, B2 etc for stable links.
  Keep ids stable; when inserting a new item prefer appending or renumbering *and* updating links.
Status Legend: ✅ Done · 🟠 In Progress · ⏳ Not Started · ➕ Optional/Future
-->

# Project Task & Completion Map

## Quick Index

| Letter | Category | Status |
|--------|----------|--------|
| [A](#A) | Core Cleanup | ✅ |
| [B](#B) | Notebooks | ⏳ (partially done) |
| [C](#C) | Expression Engine | ✅ (phase 1) |
| [D](#D) | Interactive UI | ⏳ |
| [E](#E) | Validation & QA | ⏳ |

---

## A. Core Cleanup <a id="A"></a>

<details>
<summary>1. <a id="A1"></a> ✅ Ungate IO feature – io gating removed & always compiled</summary>

1.1 Goal: Remove obsolete `io` feature gating from `candle-core` so I/O modules always compile.  
1.2 Result: Feature removed; code builds cleanly without `--features io`.  
1.3 Related: See also [C.3 Evaluator Implementation](#C3) (relies on always-available tensor I/O helpers if extended later).  

</details>

<details>
<summary>2. <a id="A2"></a> ✅ Keyword & edition fixes – reserved names & patterns updated</summary>

2.1 Changes: Renamed `gen` -> `generator`; adjusted closures / match patterns for Rust 2024 edition ergonomics.  
2.2 Outcome: Eliminated edition warnings.  

</details>

<details>
<summary>3. <a id="A3"></a> ✅ Unsafe cleanup cpu/mod.rs – minimized unsafe scope</summary>

3.1 Action: Removed superfluous nested `unsafe {}` blocks (consolidated minimal unsafe scope).  
3.2 Benefit: Clearer safety boundaries; no new warnings introduced.  

</details>

---

## B. Notebooks <a id="B"></a>

<details>
<summary>1. <a id="B1"></a> ✅ Visualization helpers notebook – helper notebook exists</summary>

1.1 Created `consolidated_helpers_egui.ipynb` with tensor/image display utilities.  
1.2 Future Link: Planned integration with [D.1 Egui interactive workbench](#D1).  

</details>

<details>
<summary>2. <a id="B2"></a> ✅ Analytical tensor fill notebook – formula demos added</summary>

2.1 Added `tensor_math_fill.ipynb` for sin/cos, gaussian, radial, checkerboard examples.  
2.2 Will reference parser via [B.3 Parsed expression integration](#B3).  

</details>

<details open>
<summary>3. <a id="B3"></a> ⏳ Notebook integration (parsed exprs) – add parsed expression demo cells</summary>

3.1 Task: Add cells to evaluate user expression strings using [C.2 Parser](#C2) & [C.3 Evaluator](#C3).  
3.2 Demos: Compare manual tensor vs parsed expression; show max diff; gaussian parametrization.  
3.3 Dependency: Requires completion of [C.4 Lifetime issue fix](#C4) (done).  
3.4 Next Step: Implement evaluation cell set & diff metrics.  

</details>

---

## C. Expression Engine <a id="C"></a>

<details>
<summary>1. <a id="C1"></a> ✅ Add parser dependency – initial external crate added</summary>

1.1 Initially added `arithmetic-parser` (later superseded by custom implementation).  
1.2 Kept as historical step; can remove if unused in future pruning.  

</details>

<details>
<summary>2. <a id="C2"></a> ✅ Custom expression parser – in-house lexer & parser</summary>

2.1 Implemented lexer + recursive descent (supports + - * / ^, unary, parentheses, functions).  
2.2 Chose custom approach due to inaccessible internal AST of external crate.  

</details>

<details>
<summary>3. <a id="C3"></a> ✅ Evaluator implementation – tensor ops + constants</summary>

3.1 Maps AST -> Candle tensor ops; constants: `pi`, `e`; params via hashmap.  
3.2 Pow via transform: `a^b = exp(b * log(a))`.  
3.3 Provides foundation for [B.3 Notebook integration](#B3) and future [D.1 Interactive workbench](#D1).  

</details>

<details>
<summary>4. <a id="C4"></a> ✅ Lifetime issue fix – closures refactored</summary>

4.1 Refactored closure helpers returning references into structured matching to satisfy borrow checker.  
4.2 Result: `candle-notebooks` crate builds cleanly.  

</details>

<details open>
<summary>5. <a id="C5"></a> ⏳ ➕ Extended functions set (optional) – more math ops</summary>

5.1 Candidates: `sinh`, `cosh`, `atan`, `atan2`, `sign`, `relu`, `sigmoid`.  
5.2 After adding, update docs & cross-links (e.g., notebook demos).  
5.3 Depends on stable base evaluator (see [C.3](#C3)).  

</details>

<details open>
<summary>6. <a id="C6"></a> ⏳ ➕ Expression cache / perf – parse & fold optimization</summary>

6.1 Plan: Hash expression string -> cached AST; maybe constant folding & simple algebraic simplifications.  
6.2 Benefit: Reduce re-parse overhead in interactive scenarios ([D.1](#D1)).  
6.3 Consider LRU to bound memory usage.  

</details>

---

## D. Interactive UI <a id="D"></a>

<details open>
<summary>1. <a id="D1"></a> ⏳ Egui interactive workbench – real-time editor & heatmap</summary>

1.1 Goal: Real-time expression edit & heatmap / surface visualization.  
1.2 Inputs: Expression string, param sliders (tied to `params` map), resolution controls.  
1.3 Reuse: Visualization helpers ([B.1](#B1)), expression engine ([C.2](#C2), [C.3](#C3)), potential cache ([C.6](#C6)).  
1.4 Stretch: Live gradient preview & performance profiling overlay.  

</details>

---

## E. Validation & QA <a id="E"></a>

<details open>
<summary>1. <a id="E1"></a> ⏳ Build & quick test validation – broaden tests</summary>

1.1 Current: Parser unit test passes (simple case).  
1.2 Todo: Add edge-case tests (precedence, unary chain, pow associativity, error handling).  
1.3 Add notebook smoke cell executing multiple expressions & reporting timing (feeds into [C.6](#C6)).  

</details>

---

## Cross-Reference Matrix (Selected)

- Parser foundation [C.2](#C2) → Evaluator [C.3](#C3) → Notebook integration [B.3](#B3) → Interactive workbench [D.1](#D1).
- Performance enhancements [C.6](#C6) amplify responsiveness of [D.1](#D1).
- Extended functions [C.5](#C5) enrich demos in [B.3](#B3) & UI [D.1](#D1).

## Update Procedure

1. Add new item under appropriate letter (append if possible).  
2. Assign next number; create `<a id="LETTERn"></a>` anchor.  
3. If inserting mid-list, adjust numbering and update any explicit textual references (search for `LETTERn`).  
4. Keep anchor ids stable where possible; if renumbering, you may keep old id as hidden alias: `<a id="OldId"></a>`.  
5. Mark status with emoji and, if completed, move descriptive tense to past.  

## Status Summary

✅ Completed: A1 A2 A3 · B1 B2 · C1 C2 C3 C4  
🟠 In Progress: (none)  
⏳ Not Started: B3 C5 C6 D1 E1  
➕ Optional/Future: C5 C6 D1  

---

_Last updated: <!--DATE-->2025-08-15<!--/DATE-->_

