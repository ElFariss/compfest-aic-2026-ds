# Preliminary-round compliance map

This map is based on the supplied **AI Innovation Challange-1.pdf** brief and
the supplied preliminary-round implementation rule. It separates what the
container implements from what the research tree uses to reproduce training.

## Static core-inference rule

| Rule | Shipped evidence | Result |
|---|---|---|
| AI implementation focuses on core inference | `real_policy/submission/` contains only the frozen runtime, HTTP wrapper, dependency lock, Compose file, documentation, examples, and checksum-bound artifacts | PASS |
| Parameters remain static during demonstration | TorchScript is loaded once at startup; the model manifest requires `runtime_auto_update=false`; the API exposes no load/reload/update method | PASS |
| Docker Compose instructions in README | Both root and submission READMEs give loopback Compose commands | PASS |
| No automatic tuning | No optimizer or tuning primitive exists inside `submission/` | PASS |
| No bulk-testing mechanism in submission | Evaluation and unit tests remain outside the demonstration boundary | PASS |
| No automatic feedback loop | The service does not store outcomes, accept labels, retrain, or promote weights | PASS |
| No automatic dispatch side effect | Every learned response requires dispatcher approval and only returns JSON | PASS |

The reproducibility trainer, raw-data acquisition script, one-time audit, unit
tests, and executed notebook remain in the repository, but none is copied into
the competition container. They document and check how the static artifact was
produced; they are not runtime features. The integrity audit verifies hashes,
tensor contracts, manifest invariants, and split isolation, but does not
independently rederive transformations or labels from raw records.

## Challenge-function map

| Brief capability | Implemented neural evidence | Runtime boundary |
|---|---|---|
| Empty-return risk | VIUS real-truck annual deadhead/reposition/loaded fractions | Source-domain prior only; no live empty-return claim |
| ETA | LaDe route ETA auxiliary learning and Singapore/DT-CARGO duration heads | LaDe intervals are capped at 72 hours; supplied road times remain hard constraints |
| Cargo-vehicle matching | Set-conditioned graph pointer trained on actual Amazon/LaDe action sequences | Ranking metrics are conditional on actual-next-node coverage in the independently selected candidate set; actual kg/cm3, compatibility, precedence, windows, and road data remain required |
| Multi-hop backhaul route | Frozen pointer is rolled out autoregressively over a maximum of 32 candidates | STOP is ignored while at least one valid candidate remains; stranded pickup or no feasible action returns `ABSTAIN` |
| Abnormal data/trip | Scania APS probability and DT-CARGO GNSS signal-loss head | APS- and DT-CARGO-specific outputs, not generic Haulio anomaly claims |
| Price and margin | NYC TLC metered-fare quantile head | Implementation proxy only; never returned as Indonesian freight margin |
| IoT position, state, load, fuel, identity | Singapore commercial-vehicle prefix encoder plus independent DT-CARGO track head; route endpoint requires current measured/manifest load and freshness | MQTT device authentication/topic authorization stays in the Haulio backend; missing actual load blocks routing |

Final held-out values are point estimates from deterministic capped subsets
(at most 24 batches or 4,608 rows per source), without confidence intervals or
probability-calibration evidence. Full final-split counts are authoritative in
the processed manifest and audit report; the metrics ledger records the exact
evaluated count. The reported
`hard_constraint_violations` value checks masked selection consistency; it is
not an independent proof that every runtime feasibility constraint was tested.

## Submission isolation

The Docker build context is `real_policy/submission/`. The resulting service:

- binds to `127.0.0.1:8088` through Compose;
- runs as an unprivileged user on a read-only filesystem;
- verifies the artifact SHA-256 before becoming ready;
- exposes only `/health` and six inference endpoints;
- returns reason-coded `ABSTAIN` for unsafe or incomplete input;
- cannot train, update, tune, record feedback, or dispatch a vehicle.

The exact public-data, domain-shift, and licensing limitations remain part of
the submitted documentation. Compliance with a static-inference rule does not
turn component-level public evidence into a field-validation claim.
