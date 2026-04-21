# Additional reproducibility files for the MHD dynamic-alignment paper

This folder contains information beyond the paper itself:

## manifests/
- manifest_15_random_nonoverlap_320_1based_seed20260421.csv
  Exact list of the 15 reported 320^3 cubes used in the final ensemble.
- audit_15_random_nonoverlap_1based_seed20260421.csv
  Audit confirming the new 15 cubes have the expected files/shapes and distinct selections.

## processed_json/
- C01...C15 *_xyplane_interp_r32_192.json
  Final processed per-cube outputs for the reported 15-cube ensemble.
- largecube_t0057_x289-736_y289-736_z289-736_xyplane_interp_r32_192.json
  Processed output for the separate 448^3 robustness run.

These files are intended to complement the paper and code repository.
Raw JHTDB .npy cubes are not included here.
