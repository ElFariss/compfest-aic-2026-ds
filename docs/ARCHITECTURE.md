# RealBackhaulNet architecture

The shipped network is trained from random initialization. Public records stay
inside their source domain: a Singapore trip is never attached to an Amazon
route, a VIUS truck is never assigned to a LaDe task, and a TLC fare is never
presented as a freight quote. Cross-domain transfer occurs only through shared
parameters.

The editable architecture board is available in
[Figma/FigJam](https://www.figma.com/board/3hAmTGtBxIFa0zhAcEsWpn?utm_source=other&utm_content=edit_in_figjam&architecture=true).

## Training graph

```mermaid
flowchart TB
  subgraph REAL["Observed public records - never row-joined"]
    AMZ["Amazon routes\nactual next stop"]
    LAD["LaDe courier-days\nactual next task + interval"]
    SGP["Singapore vehicles\n64-step OBD + trip fuel"]
    DTC["DT-CARGO N3 trucks\ntrack and GNSS summaries"]
    VIU["VIUS 2021 trucks\nannual mile fractions"]
    APS["Scania APS\n170 values + 170 missing bits"]
    TLC["NYC TLC Jan 2024\ntrip features + metered fare"]
  end

  AMZ --> N["route adapter\nnode 32x16; context 16"]
  LAD --> N
  N --> G1["4x graph-pointer block\n8-head candidate self-attention"]
  N --> C["context adapter\n160 dimensions"]
  C --> ST["shared fleet-state trunk\n2 residual MLP blocks"]
  ST --> G1
  G1 --> RP["context query x candidate keys\nmasked pointer logits"]
  G1 --> ETA["LaDe-only ETA q50/q90 head"]
  G1 --> STOP["stop-after-action auxiliary"]

  SGP --> T["temporal adapter\n64x8 to 64x160"]
  T --> TE["3x pre-norm Transformer encoder\n8 heads"]
  TE --> POOL["masked temporal mean"]
  DTC --> DA["DT adapter\n16 to 160"]
  VIU --> VA["VIUS adapter\n24 to 160"]
  APS --> HA["APS adapter\n340 to 320 to 160"]
  TLC --> PA["cost adapter\n12 to 160"]
  POOL --> ST
  DA --> ST
  VA --> ST
  HA --> ST
  PA --> ST

  ST --> TH["fuel, duration, idle head"]
  ST --> DH["track duration, signal, class heads"]
  ST --> VH["annual deadhead/reposition/loaded head"]
  ST --> HH["APS failure head"]
  ST --> PH["ordered fare-proxy quantile head"]

  RP --> L["masked supervised losses"]
  ETA --> L
  STOP --> L
  TH --> L
  DH --> L
  VH --> L
  HH --> L
  PH --> L
```

The network uses a 160-wide latent state, four graph blocks, three temporal
Transformer layers, eight attention heads, and at most 32 active route
candidates. The route decoder scores the candidate set jointly rather than
predicting a scalar edge cost for A*. It is rolled out autoregressively during
inference.

Cross-family coupling occurs through the shared trunk; Amazon and LaDe also
instantiate the same route adapter, graph blocks, pointer, ETA, and stop
modules. Only LaDe minibatches supervise the ETA head; Amazon ETA targets are
absent and masked out. A minibatch activates one source and only labels that
exist there. Missing labels are represented by `NaN` and excluded from the
loss; missing input values use frozen train-split robust statistics plus
explicit missing bits. STOP is an auxiliary output, but the deployment decoder
ignores it whenever at least one valid candidate remains.

## Deployment graph

```mermaid
flowchart LR
  MQTT["signed MQTT telemetry"] --> BE["Haulio backend\nauth, identity, schema"]
  JOBS["actual jobs and manifests"] --> BE
  ROAD["observed or road-derived\ncomplete travel matrix"] --> BE
  BE --> API["loopback inference API"]
  API --> V0["schema, unit, timestamp,\nrange and freshness checks"]
  V0 -->|invalid or missing| ABS["ABSTAIN + reason code"]
  V0 --> HCV["hard candidate feasibility\ncapacity, precedence, compatibility,\ntime windows, road and route budget"]
  HCV -->|none feasible| ABS
  HCV --> TS["checksum-bound frozen TorchScript"]
  TS --> PTR["masked next-stop pointer"]
  PTR --> V1["post-selection hard verifier\nSTOP ignored while valid candidates remain"]
  V1 -->|rejected or non-finite| ABS
  V1 --> REC["recommendation only"]
  REC --> HUMAN["dispatcher approval"]
  ABS --> SAFE["existing safe solver or manual dispatch"]
```

The competition container has static parameters. It exposes only health and
core inference endpoints; there is no optimizer, training endpoint, model
reload, auto-tuning, bulk tester, feedback store, or automatic dispatch action.

## Outputs and evidence boundary

| Head | Output | Held-out evidence | Deployment interpretation |
|---|---|---|---|
| Route pointer | next-stop logits, stop-after-action auxiliary, ETA q50/q90 | Amazon and LaDe sequence labels; ETA from LaDe only, capped at 72 h | ranking metrics include only steps whose actual next node is inside the independently selected candidate set; the hard verifier remains authoritative |
| Commercial telemetry | full-trip fuel/duration and future idle-heavy probability from a first-half prefix | vehicle-disjoint Singapore trips | source-domain IoT forecast; no cargo-load prediction |
| Heavy-truck track | duration, signal loss, home/long-haul probabilities | vehicle-disjoint DT-CARGO tracks through an independent endpoint | N3 truck/GNSS auxiliary output; never joined to Singapore payloads |
| Deadhead | annual deadhead, repositioning, loaded fractions | state-disjoint VIUS trucks | annual survey prior, not a live empty-return label |
| Health | APS failure probability | official Scania test | APS-specific source-domain health classification |
| Cost proxy | ordered p10/p50/p90 | chronologically held-out TLC trips | NYC passenger fare proxy, never an Indonesian freight quote |

No public source contains the aligned Haulio tuple of live telemetry, measured
cargo weight, open jobs, dispatcher choice, executed route, and realized
profit. Consequently this release reports component-level held-out metrics and
does not claim causal Haulio savings or field safety.

Final metric values are single point estimates on deterministic capped held-out
subsets (at most 24 batches or 4,608 rows per source); the release does not
provide confidence intervals or probability-calibration evidence. Authoritative
full-split counts live in the processed manifest and integrity-audit report;
the metrics ledger records exact evaluated counts. The integrity audit checks hashes, tensor contracts,
manifest invariants, and split-group isolation, but it does not independently
rederive features, targets, or candidate neighbourhoods from raw records. The
reported `hard_constraint_violations` value checks whether a selected index was
masked; it is not an independent feasibility proof for the runtime constraints.
