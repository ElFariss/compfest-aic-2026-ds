# Literature and evidence-gap review

The central research gap is not the absence of another attention layer. It is
the absence of a public, aligned operational dataset in which route actions,
truck IoT, measured load, open freight, dispatcher decisions, and realized
economic outcomes refer to the same vehicles and timestamps.

| Work/source | What it contributes | Gap for Haulio-style trucks | Decision in this release |
|---|---|---|---|
| Nazari et al., *Reinforcement Learning for Solving the VRP* (2018) | end-to-end policy-gradient sequence construction for sampled CVRP instances | sampled problem distribution; no real truck telemetry, orders, or field outcome | architectural precedent only; no weights reused |
| Kool et al., *Attention, Learn to Solve Routing Problems!* (2019) | attention decoder and REINFORCE rollout baseline across several routing variants | benchmark objectives and generated instances do not identify tacit dispatcher or truck-IoT effects | graph-pointer precedent; training here is from random initialization on observed sequences |
| Kool et al., *Deep Policy Dynamic Programming* (2021) | learned edge policy restricts a classical DP search | hybrid search still depends on the benchmark state/objective and does not solve missing operational data | safe-solver fallback remains separate from the learned recommendation |
| Wang et al., *Cluster-Aware Attention-Based DRL for PDP* (2026) | global plus intra-cluster attention and a gated dual decoder | published experiments use generated clustered/uniform PDP instances; the repository itself provides a generator and coordinate-pair format rather than truck operations | cluster density is a useful inductive bias, not evidence of truck readiness; no checkpoint or generated rows are used |
| Falkner and Schmidt-Thieme, *Learning to Solve VRPTW through Joint Attention* (2020) | jointly selects a vehicle/tour and customer, addressing multi-vehicle time-window trade-offs | training instances are sampled from benchmark statistics; the action state does not contain actual truck IoT, tenders, or economics | establishes that fleet-level actions need explicit vehicle state; the current runtime intentionally handles one rolling truck snapshot and does not claim fleet-joint optimality |
| Li, Yan, and Wu, *Learning to Delegate for Large-Scale VRP* (2021) | learned spatial subproblem selection plus a black-box solver scales to 500--3000 customers | Transformer supervision is generated and assumes an available subsolver/objective | supports the deterministic decomposition fallback for more than 32 candidates; no generated regression rows enter the checkpoint |
| Bi et al., *Learning to Handle Complex Constraints for VRPs* (2024) | proactive infeasibility prevention and learned masks for constraints whose feasibility itself is difficult | evidence is TSP with time windows/draft-limit benchmarks, not jurisdictional truck compliance or live sensor state | hard rules remain authoritative; hazmat/reefer candidates are conservatively masked when their legal verifier is absent |
| Heakl et al., *SVRPBench* (2025) | time-dependent congestion, stochastic delays/accidents, multi-depot and multi-vehicle evaluation; reports strong degradation under shift | the scenarios are simulator-generated despite being empirically grounded and do not provide aligned Haulio operations | external benchmark candidate only; excluded from the real-row checkpoint and used to justify explicit domain-shift abstention |
| Amazon Last Mile Challenge | real driver-operated routes, package dimensions, windows, capacity, and stop order | U.S. parcel vans; obfuscated geography; no CAN, fuel, measured mass, tender, or truckload economics | real route-pointer supervision and final-only official evaluation analogue |
| LaDe | large real courier/order event data with time and location | couriers and parcels; no truck capacity, fuel, CAN, or price | real pickup/delivery ordering and interval supervision |
| Singapore commercial vehicles | real GPS-derived/OBD features and trip fuel for ten commercial vehicles | small fleet; coordinates withheld; no order book or reliable quantitative load label | vehicle-disjoint temporal fuel/duration/idle head; direct instantaneous-fuel input excluded; load prediction disabled |
| DT-CARGO | real N3 fleet/track/GNSS-quality records over about 1.269 million km | coordinates and cargo removed; no fuel or routing decisions | vehicle-disjoint heavy-truck auxiliary head |
| VIUS 2021 | real truck characteristics and annual deadhead/reposition/loaded-mile percentages | annual survey values are not dynamic empty-return events | annual prior only, explicitly not a live decision label |
| Scania APS | real heavy-truck sensor/counter records with official failure labels | 170 predictors are anonymous and have no time or vehicle IDs | source-domain APS health head with missing masks |
| Athens pharmaceutical 3PL | nine real order sets with kg, m3, service, windows, and scenario travel matrices | vehicle capacities and actual driven sequences are absent | external schema/uncertainty audit only; no foreign capacity is attached |
| NYC TLC January 2024 | real chronological distance/duration/metered-fare rows | passenger fare is neither Indonesian truckload price nor dispatcher margin | non-operational cost-proxy head, labelled as such |

## Why not train CAADRL/CluPDTSP directly?

Its cluster-aware inductive bias is reasonable for dense pickup-and-delivery
instances, but its reported evidence is on generated coordinate problems. A
checkpoint trained on those rows would violate the real-data requirement and
would not establish heavy-truck, IoT, heterogeneous-fleet, legal, or economic
validity. RealBackhaulNet instead learns candidate interactions from actual
Amazon/LaDe sequences and treats all safety constraints as deterministic.

## Why not use A* with one learned cost?

A* is appropriate after a state graph and edge cost are defined. The difficult
observed signal here is the driver's next choice among a changing set of stops,
including latent operational preferences not expressed by shortest distance.
The pointer decoder scores the entire feasible candidate set jointly and is
rolled out as a sequence. The nearest-distance baseline remains in the held-out
report, so a learned model cannot be presented as useful merely because it is
neural.

## Why not claim reinforcement learning?

An RL label would not make the evidence stronger. Online RL would require the
system to explore dispatch actions against real trucks, which is outside the
competition's frozen-inference boundary and unsafe without a controlled pilot.
Offline RL would require logged state, candidate action set, chosen action,
behavior-policy support, and a reward such as realized empty kilometres or
margin for the same decision. None of the public sources contains that tuple.

The route branch is therefore reported accurately as supervised behavior
cloning of real next actions, trained from random initialization. It is still a
learned neural policy: candidate interactions and action logits are estimated
from data, not written as rules. Once Haulio has a leakage-safe chronological
decision log, the same decoder can be evaluated as an offline-RL policy only
after propensity/support and counterfactual evaluation gates are defined. No
such result is claimed by this release.

## Why no fabricated multimodal fusion?

Attaching a Singapore telemetry window to an Amazon route would create a row
that never occurred. The model therefore uses source-specific adapters and
losses with a shared parameter trunk. This supports transfer without claiming
that fuel or APS condition caused a particular public route choice. Actual
Haulio telemetry, load, manifests, and travel matrices enter the deployment
contract as measured facts and hard constraints. Proving their learned joint
effect requires a future chronological Haulio shadow dataset; this release
does not simulate that proof.

## References

- M. Nazari et al., arXiv:1802.04240, 2018.
- W. Kool et al., arXiv:1803.08475v3, ICLR 2019.
- W. Kool et al., arXiv:2102.11756v2, 2021.
- W. Wang et al., arXiv:2603.10053, 2026, with code at
  <https://github.com/Botwwt/CluPDTSP>.
- J. K. Falkner and L. Schmidt-Thieme, arXiv:2006.09100, 2020.
- S. Li, Z. Yan, and C. Wu, *NeurIPS* 2021,
  <https://proceedings.neurips.cc/paper/2021/hash/dc9fa5f217a1e57b8a6adeb065560b38-Abstract.html>.
- J. Bi et al., *NeurIPS* 2024, doi:10.52202/079017-2964.
- A. Heakl et al., *NeurIPS Datasets and Benchmarks* 2025,
  doi:10.52202/085713-3983.
- D. Merchán et al., *Transportation Science*, doi:10.1287/trsc.2022.1173.
- L. Wu et al., arXiv:2306.10675, 2023.
- G. Balke and L. Adenaw, *Data in Brief* 48 (2023), 109246.
- L. W. Yeow and L. Cheah, *Transportation Research Record* (2021),
  doi:10.1177/03611981211007478.
- A. Vrani et al., *Data in Brief* 61 (2025), 111762.
