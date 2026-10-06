# Adding a Method

1. Keep the native method under `EXIST/3D/` or `EXIST/4D/`, preserving its license
   and internal imports. Record its source revision and modification provenance.
2. Create an isolated environment profile. Do not add Torch/MMCV/model packages
   to the lightweight `occstress` package dependencies.
3. Add the formal dataset configs and entrypoints to
   `configs/evaluation_contract.json`. State input offsets, modality, available
   tracks, control policy and required temporal-alignment switches. Direct-camera
   models must not be advertised as accepting manual occupancy inputs.
4. Extend the thin command adapter only when the native interface differs. Keep
   model internals and model registries in their own subprocess. Dataset input
   must obey the canonical H4/current/F6 contract and class/coordinate convention.
5. Export six-horizon raw confusion matrices or sufficient counts, record the
   actual evaluated anchor IDs and all checkpoint components. Add a count reader
   only if a native format cannot already be read. Do not normalize rounded mIoU.
6. Add CPU tests for config selection, input window, raw-count conversion and
   resume identity. Refresh the derived method inventory and reviewed source lock.
7. In the native environment, pass checkpoint loading, clean/corruption smoke,
   original-loader versus clean-adapter equivalence and paper-reproduction gates.
   CPU tests are not substitutes for those gates. Publish the gate evidence with
   the reference result, including the precise score policy and anchor coverage.

See `docs/CORE_WORKFLOW.md` for output identity and `docs/FORMAL_EVALUATION.md`
for the authoritative contract. Adding a supported interface does not imply
that its entire dataset/source/protocol product has been evaluated.
