# VRP, IoT, and deployment edge-case matrix

This matrix is the operational contract for the prototype. `ENFORCED` means
the submitted runtime rejects or masks the condition today. `LEARNED` means a
real public source supplies a held-out task label. `UPSTREAM` means Haulio or a
road-data service must supply the fact before inference. `ABSTAIN` means the
policy returns a reason-coded non-recommendation. `NOT CLAIMED` means no honest
public evidence supports the capability.

| Edge case | Required actual data | Current evidence | Architecture handling | Safe fallback | Status |
|---|---|---|---|---|---|
| Weight capacity | truck rated kg, current measured kg, per-stop kg delta | proposed cargo-weight interface; no paired public route label | hard check before every selection | reject candidate; manual/solver route | ENFORCED, not learned |
| Volume capacity | truck cm3, current cm3, per-stop cm3 delta | Amazon executor capacity and package dimensions | hard check plus learned volume features | reject candidate | ENFORCED + LEARNED analogue |
| Simultaneous weight and cube limit | both capacity dimensions in the same request | no joint public truck-routing source | both constraints must pass | ABSTAIN if no candidate passes | ENFORCED |
| Pickup-before-delivery | shipment ID, stop type, onboard IDs | LaDe pickup tasks; runtime manifests | delivery infeasible until pickup/onboard | leave stop unserved and explain | ENFORCED |
| Pickup stranded by early stop | paired pickup/delivery IDs and rollout state | runtime manifest | STOP is ignored while at least one valid candidate remains; every pickup requires a same-request delivery | ABSTAIN and solve/manual | ENFORCED |
| Duplicate pickup/delivery | unique shipment and stop IDs | runtime schema | duplicate pair rejected before model | correct manifest and retry | ENFORCED |
| Delivery already onboard | actual onboard shipment IDs | IoT/backend manifest | delivery allowed without in-request pickup | dispatcher review | ENFORCED |
| Pickup already onboard | actual onboard shipment IDs | IoT/backend manifest | pickup rejected | correct manifest/manual | ENFORCED |
| Negative or inconsistent load delta | unit-labelled kg and cm3 | runtime schema | sign rule tied to stop type | ABSTAIN invalid request | ENFORCED |
| Pickup/delivery quantity mismatch | paired shipment kg/cm3 deltas and compatibility metadata | runtime manifest | paired deltas must cancel and cargo/vehicle metadata must agree | correct manifest and retry | ENFORCED |
| Time windows and service time | UTC windows, service seconds, travel seconds | Amazon/LaDe windows and events | feasibility uses arrival, waiting, service completion | reject late stop | ENFORCED + LEARNED analogue |
| Route-duration budget | route start, current UTC, allowed seconds | backend/driver policy | service end cannot exceed deadline | existing solver/manual | ENFORCED |
| Driver hours, rests, jurisdiction rules | duty log, local law, driver calendar | absent from public training stack | must become another deterministic verifier | manual dispatch | UPSTREAM, NOT CLAIMED |
| Heterogeneous vehicle type | actual vehicle type, required type | DT-CARGO fleet metadata; runtime manifest | exact/`any` compatibility check | choose another truck | ENFORCED |
| Cargo class compatibility | actual compatible cargo classes | shipment/vehicle master data | hard membership check | choose qualified truck | ENFORCED |
| Hazmat permits and segregation | hazmat class, permits, incompatibility graph | CFS candidate source only; not trained | any hazmat candidate is conservatively masked because no legal verifier is shipped | mandatory manual compliance flow | ENFORCED reject; capability NOT CLAIMED |
| Temperature-controlled cargo | setpoint, sensor stream, reefer capability | no paired source | any temperature-controlled candidate is conservatively masked | manual reefer workflow | ENFORCED reject; capability NOT CLAIMED |
| Axle/road/bridge restrictions | axle class, gross mass, restriction-aware graph | DT-CARGO axle and mass; road service required | road availability is hard; legal mass routing is upstream | restriction-aware solver/manual | UPSTREAM |
| Road closure or missing edge | complete matrix with availability bit | road provider/backend | unavailable edges rejected; incomplete matrix rejected | refresh road matrix/manual | ENFORCED |
| Asymmetric travel | directed origin-destination matrix | runtime schema | every directed edge validated separately | ABSTAIN incomplete matrix | ENFORCED |
| Impossible distance/time pair | directed distance, travel seconds, operator maximum plausible speed | road provider/backend | available nonzero edge needs positive time and cannot exceed the declared speed limit | refresh matrix/manual | ENFORCED |
| Traffic uncertainty | timestamped travel-time distribution | Amazon matrix analogue; no Indonesia live labels | only supplied point travel time is enforced | conservative upstream matrix/manual | NOT CLAIMED as stochastic VRP |
| Multiple depots | depot IDs and vehicle home/base | DT-CARGO home-base label | request can contain depot candidates; one active origin | decompose per vehicle/depot upstream | PARTIAL |
| Open route/no return | explicit route policy and depot requirement | no aligned label | completion follows submitted candidates; return-to-depot requirement is not encoded | dispatcher/solver | PARTIAL |
| Split deliveries | divisible demand and remaining quantity | absent | one manifest stop has atomic delta | split into explicitly authorized stops upstream | NOT NATIVE |
| Multiple trips per shift | reload/depot events, shift state | absent | one request is one rolling route state | issue new request after verified depot reload | PARTIAL |
| Trailer swap/drop-and-hook | tractor/trailer IDs and compatibility | DT-CARGO mass metadata only | not represented as a state transition | manual yard workflow | NOT CLAIMED |
| Dynamic order arrival | event ID/version and current candidate set | live backend only | stateless re-inference on a new immutable snapshot | keep accepted prefix, rerun/manual | UPSTREAM |
| Cancellation after recommendation | order version and execution state | live backend only | submitted runtime never executes dispatch | backend invalidates recommendation | UPSTREAM |
| Already-served or locked stop | execution ledger and lock owner | live backend only | not inferable from telemetry alone | exclude upstream; manual conflict resolution | UPSTREAM |
| More than 32 candidates | candidate count | runtime schema | request rejected rather than silently truncating | deterministic geographic/temporal decomposition | ENFORCED limit |
| No feasible candidate | all constraint inputs | runtime verifier | model is not called on an empty feasible set | ABSTAIN with per-stop reasons | ENFORCED |
| Model stop request before completion | stop-after-action logits | LaDe final-transition label | STOP is ignored while at least one valid candidate remains, so it cannot strand work | continue under hard verifier | ENFORCED |
| Non-finite model output | artifact output | runtime validation | recommendation suppressed | ABSTAIN | ENFORCED |
| Corrupt or replaced checkpoint | artifact SHA-256 and manifest | frozen artifact contract | startup fails before serving ready | deploy last verified image | ENFORCED |
| Missing required field | complete schema | runtime validation | no plausible-value imputation | ABSTAIN with field path | ENFORCED |
| Wrong unit | explicit schema units | preprocessing/runtime contract | fixed kg, cm3, km, seconds and UTC fields | reject and correct adapter | ENFORCED by schema |
| Clock drift/out-of-order or inconsistent intervals | device/server timestamps, monotonic sequence, sample interval | Singapore timestamp analogue | timestamps must strictly increase; canonical `delta_t_s` must match each interval | quarantine batch; device-health alert | ENFORCED |
| Stale telemetry | event age and safety threshold | proposed gateway health | stale route state rejected | last safe plan/manual location check | ENFORCED |
| Poor GPS accuracy | observed accuracy and operator safety limit | DT-CARGO GNSS fields | route request is rejected when observed accuracy exceeds the declared limit | upstream geofence/manual check | ENFORCED |
| Low fuel reserve | observed fuel fraction and operator reserve threshold | Singapore OBD/fuel analogue | route request is rejected below the declared reserve; no unobserved consumption model is invented | refuel/manual dispatch | ENFORCED |
| Cellular outage | gateway buffer offsets and message IDs | proposed store-and-forward | outside model; QoS 1 can duplicate | deduplicate backend, replay in order | UPSTREAM |
| MQTT duplicate delivery | message/event ID and device sequence | MQTT QoS 1 semantics | outside inference service | idempotent persistence/backend dedupe | UPSTREAM |
| Late replay after route changed | event time, ingest time, route version | gateway/backend | freshness gate blocks stale state | retain audit record, do not route | UPSTREAM + ENFORCED freshness |
| Sensor missingness | per-field presence bits | Scania, VIUS, DT-CARGO | frozen masks for supported heads; route-required facts cannot be imputed | ABSTAIN or source-specific mask | ENFORCED |
| Implausible sensor value | physical ranges and unit metadata | Singapore/Scania analogues | strict runtime ranges for exposed semantic fields | reject/quarantine | ENFORCED |
| Sensor spoofing/device impersonation | device identity, signature, topic binding | no model label | backend HMAC/TLS/auth authorization | reject before DS | UPSTREAM security |
| CAN/J1939 anomaly | raw CAN frames and attack labels | ROAD candidate source not used | no shipped CAN anomaly head | gateway rules/manual maintenance | NOT CLAIMED |
| Predictive maintenance | named sensors and failure semantics | Scania APS anonymous 170-field failure task | source-domain APS probability only | maintenance review; never stop truck automatically | LEARNED analogue |
| Fuel prediction | OBD sequence and trip boundary | Singapore real commercial vehicles | temporal encoder predicts source-domain trip fuel | display uncertainty boundary/manual | LEARNED analogue |
| Actual cargo load state | load cell/manifest | Singapore survey is too coarse | prediction intentionally disabled | require actual measured/manifest load | ENFORCED absence |
| Empty-return probability | trip-level empty/load label and open jobs | VIUS annual percentages only | annual prior head, not live classification | do not automate; gather Haulio labels | NOT CLAIMED dynamically |
| Freight price/margin | accepted quote, costs, lane, truck, time | TLC is passenger-fare proxy only | separate proxy endpoint marked non-operational | use existing pricing/human quote | NOT CLAIMED |
| Weather/flood/port closure | timestamped route impact | no aligned current training source | not fed into frozen policy | road-provider rule/manual operations | NOT CLAIMED |
| New geography/domain shift | source/domain score, operator maximum, and shadow outcomes | U.S./China/Singapore/U.S. public sources | route inference rejects a supplied score above the supplied operator limit; auxiliary claims remain source-domain | ABSTAIN/manual; shadow evaluation | ENFORCED threshold, calibration UPSTREAM |
| New truck/cold start | actual static truck specs | VIUS/DT-CARGO analogues | semantic inputs permit inference but no Haulio calibration | conservative rules/manual | PARTIAL |
| Concept/data drift | versioned real outcomes | no online Haulio stream in submission | no automatic update by competition design | offline governed retraining after review | NOT IN RUNTIME |
| Fairness/driver preference | consented driver constraints/preferences | no source | not inferred | human scheduling policy | NOT CLAIMED |
| Automatic dispatch side effect | authenticated dispatch API | deliberately absent | service returns recommendation only | dispatcher approval | ENFORCED |

## Fallback order

1. Reject malformed, stale, incomplete, or checksum-invalid inputs before model
   execution.
2. Apply deterministic feasibility and safety constraints to every active
   candidate.
3. Run the frozen pointer only when at least one candidate is feasible.
4. Re-check the selected candidate and suppress non-finite output.
5. Return `ABSTAIN` with machine-readable reasons when a safe learned
   recommendation cannot be formed.
6. Hand off to Haulio's existing deterministic solver or a dispatcher. The DS
   container never executes a dispatch and never learns from the response.

This ordering makes missing evidence visible. It does not convert an unobserved
fact into a neural prediction merely to keep the pipeline moving.

## Evaluation terminology

Route-ranking metrics are conditional on the actual next node being present in
the independently selected candidate set; they do not measure uncovered route
steps. The reported `hard_constraint_violations` field is a masked-selection
consistency check. It is not independent evidence that the public evaluation
rows exercised, or proved, every capacity, legal, time-window, road, and route
budget constraint listed above. All final metrics are point estimates on
deterministic capped subsets (at most 24 batches or 4,608 rows per source),
without confidence intervals or probability-calibration evidence.
