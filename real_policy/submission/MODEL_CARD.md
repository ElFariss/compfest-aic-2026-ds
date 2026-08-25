# RealBackhaulNet frozen model card

## Artifact

| Field | Value |
|---|---|
| Model | RealBackhaulNet |
| Artifact | `artifacts/real_backhaul_policy_frozen.pt` |
| SHA-256 | `11cb3412afc970a6c8c2a26a8d7f0a07ba2da6ff16c8c6c54d1b8375acd1a1f5` |
| Parameters | 3,407,060 |
| Training hardware | NVIDIA A100-SXM4-80GB |
| Completed training | 1,100 minibatches, batch 192, 40 min 33 s total release wall time |
| Initialization | random |
| Pretrained weights | false |
| Synthetic feature rows | false |
| Cross-dataset row joins | false |
| Runtime parameter updates | false |

The manifest binds the TorchScript bytes, preprocessing schemas, model tensor
dimensions, source-data manifest hash, dispatcher gate, and runtime immutability
flags. Startup fails if the artifact hash or contract differs.

## Intended use

This is a competition prototype for dispatcher-reviewed backhaul decision
support. It provides source-domain neural outputs plus a graph-pointer route
recommendation guarded by deterministic operational constraints. It is not an
autonomous dispatch controller.

Appropriate uses:

- demonstration of static neural core inference;
- shadow-mode ranking of already feasible route candidates;
- component-level evaluation against the documented public-source domains;
- engineering integration behind Haulio's authenticated backend.

Prohibited interpretations:

- field-validated Haulio savings, safety, SLA, or margin improvement;
- a live empty-return classifier calibrated to Indonesian trucks;
- an Indonesian freight price or margin quote;
- cargo-weight prediction when the actual load sensor/manifest is missing;
- device authentication, legal hazmat/reefer approval, or automatic dispatch.

## Architecture and learning

The network is trained from scratch. Amazon and LaDe activate a 32-candidate,
16-feature graph pointer with four candidate-attention blocks, a
stop-after-action auxiliary, and a LaDe-supervised ETA quantile head. Runtime
ignores the stop auxiliary while submitted candidates remain. A 64-step
Singapore commercial-vehicle prefix activates a three-layer temporal
Transformer. DT-CARGO, VIUS, Scania APS, and NYC TLC use separate adapters.
Cross-family transfer occurs through the shared 160-dimensional trunk; Amazon
and LaDe additionally share the complete route branch. Records from different
providers are never joined.

This release uses supervised behavior cloning and source-specific auxiliary
losses, not reinforcement learning. No public source provides the aligned
logged action propensities and realized Haulio reward required for defensible
offline RL, and the competition runtime cannot explore online.

## Data and final evidence

Exact raw and processed hashes are under `../evidence/`. Final split sizes:

| Source | Train / validation / final |
|---|---|
| Amazon | 60,000 / 10,000 / 10,000 |
| LaDe | 60,000 / 10,000 / 10,000 |
| Singapore | 1,831 / 126 / 465 |
| DT-CARGO | 71,120 / 14,370 / 16,336 |
| VIUS | 17,340 / 4,073 / 3,335 |
| Scania APS | 51,085 / 8,915 / 16,000 |
| NYC TLC | 100,000 / 20,000 / 20,000 |

Representative deadline-bounded final metrics (up to 4,608 rows per source):

| Source | Metric | Value |
|---|---|---:|
| Amazon | Next-stop top-1 | 0.6934 |
| LaDe | Next-task top-1 | 0.4364 |
| Singapore | Fuel MAE (canonical) | 0.4342 |
| Singapore | Future-idle AP | 0.4919 |
| DT-CARGO | Signal-loss MAE (canonical) | 0.3391 |
| VIUS | Deadhead MAE (canonical) | 0.1728 |
| Scania APS | Failure AP | 0.7474 |
| NYC TLC | Fare MAE (canonical) | 0.0606 |

All regression metrics marked `canonical` remain in their documented
transformed source spaces. `metrics.json` contains the complete set. Final
partitions were first opened only after validation checkpoint selection and
TorchScript freezing. The ledger records the cap, ordering, and exact
per-source count. Eager-to-frozen maximum absolute parity difference was
`0.000e+00`.

## Safety and fallback

The route endpoint validates units, ranges, timestamp order, freshness, fuel
reserve, GNSS accuracy, domain-shift threshold, balanced pickup/delivery
quantities, and a complete plausible directed road matrix. Every action is
masked for capacity, precedence, compatibility, windows, route duration, and
road availability. Hazmat and reefer candidates are conservatively rejected
because no legal/temperature verifier is present. A missing fact, stranded
pickup, impossible edge, corrupt artifact, or non-finite output returns
reason-coded `ABSTAIN`.

Amazon and LaDe route results are conditional on the actual next action being
inside the target-independent nearest-32 candidate set. Mask compliance checks
model indexing; it is not an empirical claim that the public route records
exercise every production truck constraint.

Fallback is the existing deterministic solver or manual dispatcher flow. Every
non-abstaining response still sets `dispatcher_approval_required=true`; the
container has no authenticated dispatch capability.

## Domain and fairness limitations

Amazon and LaDe are parcel/courier analogues, Singapore contains ten commercial
vehicles, DT-CARGO withholds cargo/coordinates, VIUS is an annual U.S. survey,
Scania features are anonymous, and TLC is passenger transport. These domains do
not establish Indonesian fleet calibration. Driver preference, labor rules,
hazmat law, reefer compliance, axle/bridge restrictions, weather disruption,
and port state require explicit upstream facts and policy controls. Missing
evidence is not imputed from a different dataset.
