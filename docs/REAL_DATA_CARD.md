# Real Public Data Card

Version: 1.0

Ledger verified: 2026-08-25

Machine-readable sources: [`data_sources/registry.yaml`](../data_sources/registry.yaml)
Remote artifact locks: [`data_sources/checksums.lock`](../data_sources/checksums.lock)

## Scope and current status

This card governs the shipped real-public-data release. Its frozen artifact is
under `real_policy/submission/artifacts/`; processed-file hashes and exact final
split counts live in the processed manifest and integrity-audit report under
`real_policy/evidence/`. The metrics file contains held-out scores, not the
authoritative split counts. The older simulator-trained release is not part of
the current tree or claim set.

The achievable competition artifact is a real-public-data-trained,
cross-domain prototype. It is not a field-validated Haulio optimizer because
no public dataset contains the required joint operational history:

```text
verified truck telemetry
  + current load and availability
  + contemporaneous open cargo
  + truck-order tender and dispatcher decision
  + executed multi-hop route
  + realized fuel, SLA, revenue, and margin
```

## Non-negotiable provenance rules

1. Raw public data live under `data_sources/raw/` or another explicit external
   directory and are never committed.
2. Synthetic feature rows are prohibited. There are no random orders,
   coordinates, telemetry values, time windows, prices, weather values, or
   cargo records in a real-data run.
3. Records from unrelated sources are never joined into a fake operational
   episode. In particular:

   - Singapore telemetry is not assigned to an Amazon route.
   - A VIUS truck is not assigned to a CFS shipment.
   - An Olist price is not copied onto a LaDe or Amazon order.
   - ROAD CAN attacks are not presented as attacks observed on a Haulio truck.

4. Same-source grouping is allowed when the source supplies the cohort key.
   For example, Amazon routes at the same station and service date may form a
   real multi-executor cohort. Candidate feasibility derived inside that
   cohort must be labelled `same_source_solver_derived`, not historical truth.
5. Each minibatch belongs to one source. Task masks activate only labels that
   source actually observes or permits deriving from its own records.
6. Each processed row records a stable `sample_id`, a traceable `source_id`,
   and a group key where the source permits grouping. The processed manifest
   binds every split to its SHA-256 and binds each source to its raw files and
   SHA-256 values.
7. Missing values remain missing and receive an explicit mask. They are not
   filled with plausible-looking values from another source.

Allowed label origins are:

- `observed`
- `same_source_derived`
- `same_source_solver_derived`

## Source ledger and evidence boundary

The current training stack is Singapore, lightweight DT-CARGO, VIUS, Amazon,
LaDe, Scania APS, and NYC TLC January 2024. Athens pharmaceutical deliveries
and planned-versus-driven routes are external audits only. CFS, ROAD, Olist,
and OSM remain documented candidates but are **not used by the current model**;
their presence in the registry is not a training claim.

| Source | Real observations and permitted labels | Intended task | Exact evidence gap |
|---|---|---|---|
| Singapore commercial vehicle GPS/OBD/payload/fuel | Ten commercial vehicles, including six HGVs; speed, engine state, grade, engine load, mass airflow, publisher-calculated instantaneous/trip fuel, and coarse pickup/delivery surveys. Current inputs use the first half of each trip; labels are full-trip calculated fuel/duration and a future-half idle-heavy class. | Primary truck-like IoT fuel, future idle, and duration learning. | Coordinates are removed; payload surveys lack initial load and consistent quantitative units. The load target is deliberately missing and no load prediction is exposed. Fuel is calculated rather than measured by a calibrated flow meter. |
| DT-CARGO v2 | The locked lightweight tables contain 101,826 tracks, 53 observed vehicle IDs, and about 1.269 million km. The publication describes 54 N3 trucks; this release reports the file contents rather than silently reconciling the difference. | Heavy-truck duration, GNSS signal-loss, home-base, and long-haul tasks. | No coordinates, cargo identity, load state, fuel, order, price, or revenue. The current artifact does not include the 9.7 GB full speed corpus. |
| 2021 VIUS PUF | Truck body/trailer/configuration, weight, GVWR, MPG, fuel, commodities, annual mileage, `DEADHEADPCT`, and `REPOSITIONPCT`. | Static vehicle representation and annual deadhead prior. | Annual survey percentages are not dynamic empty-return events and are not joined to an order book. |
| Scania APS Failure | Official train/test records with 170 anonymized numeric sensor and histogram fields, explicit missing values, and observed APS-related failure class. | Missingness-aware truck health and APS component-failure classification. | Anonymous fields cannot be mapped to proposed IoT/J1939 sensors; the label is APS-specific; no timestamps or vehicle IDs support fleet/temporal validation. |
| Amazon Last Mile 2021 | The downloaded official files contain 6,112 build routes and 3,052 evaluation routes with date/station, executor volumetric capacity, packages, time windows, stops, and actual driver sequence. | Learned next-stop sequence analogue. | Last-mile U.S. vans/parcels, obfuscated locations, and no truck fuel/CAN. Not FTL backhaul. |
| LaDe | The current merged five-city files contain 472,419 delivery rows and 531,115 pickup rows; the broader source publishes substantially more records. Fields include courier, location, accept/receipt and pickup/delivery event times. | Pickup/delivery ETA, task sequencing, and spatial-demand analogue. | Courier domain; no truck capacity, fuel, CAN, line-haul price, or tender decision. |
| Athens pharmaceutical 3PL | Nine real daily order sets with shipment weight, volume, service time, time windows, distance matrix, and optimistic/most-likely/pessimistic time matrices. | External schema and travel-uncertainty audit. | The archive supplies no vehicle capacity or actual driven sequence. It is not joined to another source and does not enter training. |
| NYC TLC Yellow Taxi, January 2024 | Real taxi pickup/drop-off times, zones, distance, rate code, payment type, metered fare, taxes, tolls, tips, and surcharges. | Same-row metered-fare regression and duration auxiliary task. | NYC passenger-taxi fare is not Indonesian truckload price or margin; zones are coarse and fare rules are month/locality-specific. |

The Amazon dataset paper describes 3,072 evaluation routes and 9,184 routes in
total. The six currently locked public JSON objects contain 3,052 matching
evaluation IDs and 9,164 routes in total. This release records the discrepancy
and reports the byte-for-byte files it actually processed.

CFS, ROAD, Athens, planned-versus-driven routes, Olist, and OpenStreetMap
remain documented candidates or external audits in the machine-readable
registry. They do not enter the current checkpoint.

The Deliveree public website is explicitly excluded. No verified open bulk
training dataset or licensed training API was identified, so the site must not
be scraped or named as a training source without written permission.

## Primary IoT artifact

The primary immediately downloadable IoT source is the Singapore commercial
vehicle dataset:

- DOI: `10.6084/m9.figshare.9741035.v2`
- Metadata API: `https://api.figshare.com/v2/articles/9741035`
- Immutable file ID: `figshare:24337976`
- Direct file: `https://ndownloader.figshare.com/files/24337976`
- Size: `46,804,275` bytes
- SHA-256: `dd6587b8d7c568ec08320449dd14305bfc29e52a20baf8876ed9a8a60f121560`
- Provider MD5: `ac872cf3e3bf6a0aa163250ed166ac42`
- License: CC BY 4.0

The per-vehicle GPS/OBD tables contain:

```text
timestamp_GMTplus8
speed_kmh
engine_status
TripID
road_grade_proportion
engine_load_obd_percent
mass_air_flow_g_per_s
instantaneous_fuel_L_per_s
```

`instantaneous_fuel_L_per_s` is calculated by the publisher from engine load
and mass airflow; it is not a direct flow-meter observation. The exact payload
headers are `StartTime`, `EndTime`, `Pickup_type`, `Pickup_volume`,
`Deliver_type`, and `Deliver_volume`. The trip-level label is exactly
`OBD - Fuel used (L)` and is also calculated from the instantaneous stream.
The fuel head does not ingest `instantaneous_fuel_L_per_s`: preprocessing uses
only the first half of the trip with speed-derived longitudinal acceleration,
engine state, grade, OBD engine load, and mass airflow. The targets cover the
full trip (or the future half for idling), avoiding direct reuse of the target's
integrated stream and avoiding a same-window idle aggregation masquerading as
prediction.

Vehicles B, C, D, G, I, and J additionally have coarse stop-survey fields for
pickup/delivery type and volume. Because initial load is unknown and the volume
answers are inconsistent free text, the current preprocessing retains their
provenance but does not create a continuous or binary load label. They are not
a substitute for the proposed cargo-weight sensor.

## License and access controls

Downloading data is not permission to ignore its terms.

- Amazon Last Mile is CC BY-NC 4.0. It is non-commercial and is used here only
  as research/prototype data. The downloader requires
  `ACCEPT_AMAZON_CC_BY_NC_4_0=1`. Commercial or production reuse requires
  separate rights.
- LaDe's Hugging Face metadata declares Apache-2.0, while its README separately
  states that it may be used for research and instructs users to read terms.
  Both statements are preserved as a license caveat. The downloader pins the
  verified repository revision and requires `ACCEPT_LADE_RESEARCH_TERMS=1`.
- Olist's authenticated Kaggle page is the license authority. Public mirrors
  commonly identify CC BY-NC-SA 4.0, but the displayed license must be captured
  at download time. The downloader requires
  `ACCEPT_OLIST_DATASET_TERMS=1`, and raw redistribution is disabled.
- ROAD raw redistribution is disabled until the source record's applicable
  license has been reviewed and recorded.
- The current UCI landing page labels Scania APS Failure CC BY 4.0, while the
  exact official ZIP embeds a GPLv3 notice from Scania. Both notices are
  preserved rather than silently choosing one; redistribution must satisfy the
  applicable source terms. The lock points to the official ZIP, not a mirror.
- NYC TLC directs trip-record users to the NYC Open Data Terms of Use. This
  repository does not reinterpret those terms as a permissive software license.
- DT-CARGO and OpenStreetMap are ODbL-governed. Attribution and applicable
  share-alike/database obligations remain in force.
- Census VIUS and CFS files retain their official public-use documentation,
  citations, disclosure-protection flags, and survey weights.

Raw Amazon, LaDe, Olist, ROAD, and other third-party data are not packaged in
the model repository or competition archive.

## Downloading

The downloader takes explicit source names and never starts a multi-gigabyte
download when called with no arguments:

```bash
./data_sources/download_real_data.sh --list
./data_sources/download_real_data.sh small
./data_sources/download_real_data.sh scania tlc
./data_sources/download_real_data.sh singapore vius deviation
```

Restricted examples require explicit acknowledgement:

```bash
ACCEPT_AMAZON_CC_BY_NC_4_0=1 \
  ./data_sources/download_real_data.sh amazon

ACCEPT_LADE_RESEARCH_TERMS=1 \
  ./data_sources/download_real_data.sh lade
```

The script resumes partial direct-file downloads, verifies locked digests or
provider manifests, and skips already verified artifacts. It does not silently
replace a complete file that fails verification. `cfs_2022_pums.zip` is locked
to the verified official URL and byte count because the inspected source did
not publish a digest; a local SHA-256 must be recorded before processing.

## Integrity-audit boundary

The integrity audit verifies raw and processed file sizes and SHA-256 values,
the preprocessing-code hash, declared tensor shapes and dtypes, lineage-ID
presence, split-group disjointness, and documented tensor invariants. It trusts
the preprocessing program that created the manifest: it does not independently
recompute transformations, labels, or candidate neighbourhoods from the raw
records. Reproducing those derivations requires rerunning `prepare_real.py` and
comparing the resulting manifest and processed hashes.

## Split and leakage requirements

- Singapore: fixed vehicle-disjoint split, A--F train, G--H validation, I--J
  final test. This is imbalanced because vehicle A contributes most records;
  metrics remain source-domain and are reported with the split counts.
- DT-CARGO: deterministic vehicle-disjoint 70/15/15 split.
- VIUS: state-disjoint split; robust medians/IQRs are fit on train states only,
  and missing inputs retain explicit mask bits. Survey weights are not used by
  the current loss, so no population-weighted claim is made.
- Scania APS: fit preprocessing and select the checkpoint using only the
  official training file; preserve the official test file for one final test.
  Because IDs and timestamps are absent, do not call it a fleet/time split.
- Amazon: official evaluation routes are final-only; official build routes are
  deterministically route-disjoint between train and validation. Candidate
  selection uses nearest published coordinates and never reads the target.
  Route-ranking metrics are conditional on the actual next node appearing in
  that candidate set; uncovered steps are not counted as ranked examples.
- LaDe: deterministic courier-disjoint 80/10/10 split. No chronological or
  held-out-city result is claimed by this release. LaDe is the only source that
  supervises the route ETA head; Amazon supplies no ETA target.
- NYC TLC: pickup-day-grouped chronological split; fit every transform on train
  days only and exclude fare-derived totals from the inputs.

If later activated, candidate sources retain their registry safeguards: CFS
needs shipment/origin-disjoint evaluation, ROAD needs capture/attack-family
separation, planned-versus-driven needs driver/chronological separation, and
Olist needs chronological plus seller-disjoint sensitivity. None of those
checks is reported as a current checkpoint result.

No aggregate simulator reward or invented "Haulio score" is permitted. This
checkpoint reports per-source point metrics on deterministic capped final
subsets (at most 24 batches or 4,608 rows per source) and explicit domain gaps.
It does not report confidence intervals, calibrated probabilities, or empirical
post-verifier violation rates where public sources lack aligned runtime facts.
The metric named `hard_constraint_violations` only checks whether route decoding
selected a masked candidate; it is not an independent feasibility proof for the
submission runtime's operational verifier.

## What can and cannot be validated

Real held-out public data can validate implementation-level tasks:

- commercial-vehicle fuel, duration, and idle-state prediction;
- annual deadhead prior;
- LaDe-only task-interval ETA prediction; any reported quantile coverage is a
  held-out point estimate, not a calibration study;
- last-mile assignment ranking and route sequence;
- heavy-truck GNSS-quality and duration estimation;
- Scania APS component-failure classification;
- NYC taxi fare regression as a price-head implementation proxy.

It cannot validate:

- Indonesian fleet-calibrated empty-return probability;
- actual Haulio cargo-truck acceptance or dispatcher preference;
- realized multi-hop empty-kilometre reduction;
- Haulio SLA or contribution-margin uplift;
- Indonesian truckload price elasticity;
- causal benefit over current dispatch operations;
- an end-to-end IoT-to-margin claim on one joined held-out dataset.

Those claims require a frozen, chronological Haulio shadow evaluation followed
by a dispatcher-controlled pilot. Until then, the accurate product label is:

> Real-public-data-trained cross-domain backhaul policy prototype.
