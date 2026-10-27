# Validation record

Date: 2026-10-27. All tests ran on CPU with two OpenMP threads.

- The 10 extracted original test files passed: **55 passed**.
- All 32 provenance records verified against the source working tree: 31 files
  byte-identical and one package-initializer export-table edit explicitly
  recorded. No model/loss/configuration algorithm was rewritten.
- An AST dependency walk selected all 16 local method modules required by the
  B2/B3/B4 implementations, their processor hooks, and temporal-ensemble helper.
- Selected method modules were scanned for machine-specific absolute paths;
  no `/home/` or `/media/` paths were found.

Test environment:

- Python 3.12.13
- PyTorch 2.10.0+cu128 (tests ran on CPU)
- LeRobot 0.5.2, imported from an existing local source checkout
- The release preserves the source project's `lerobot==0.5.1` and
  `transformers==5.3.0` dependency declarations. These differ from the locally
  imported LeRobot version. Fresh-install compatibility is **unverified**;
  passing local tests must not be described as a clean-environment reproduction.

Tests use small synthetic tensors and local temporary files. No dataset,
scene, satellite asset, or learned checkpoint is required by these tests.
They validate the original components' existing contracts, including B3's
output-only head and B4's separate rollout; they do not validate equivalence
to the manuscript's architecture or reproduce any capture-performance table.
