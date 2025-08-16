Hey folks,

Opening a draft to poke at adding some new tensor ops to Candle, specifically FFT and Scan for both CPU and GPU.

What's cooking in this branch:
*   FFT: Scaffolding for proper Fast Fourier Transform implementations.
*   Scan: Laying down tracks for parallel prefix-sum primitives.
*   0aEXPLORATION Playground: A new dir (`/0aEXPLORATION`) for hacking on prototypes and notebooks before they're ready for primetime in the core crates.

This is an early-stage feeler to get eyes on the direction.

---
On the workflow: Hacking with an AI assistant

Full disclosure: I built this branch with an AI coding assistant. It was a new workflow for me.

The good: it's incredibly fast for bootstrapping boilerplate and exploring different structures. The bad: it can generate a lot of noise, subtle bugs, and artifacts that need a human to spot and clean up. It's a powerful tool, but it definitely doesn't replace the programmer.

---

Let me know what you think.
