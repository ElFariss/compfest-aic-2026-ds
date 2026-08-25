#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
raw_dir="${REAL_DATA_RAW_DIR:-${repo_dir}/data_sources/raw}"

singapore_sha256="dd6587b8d7c568ec08320449dd14305bfc29e52a20baf8876ed9a8a60f121560"
vius_sha256="25261e9188c0d69d9e0774d6d06c9314b2b83de14d95865f3c9d2c71472b479b"
dt_cargo_fleet_sha256="ef21179d2b383044046bd6e226f43daa6c09a330e81bb5192bc6b0b68d3e1bd0"
dt_cargo_tracks_sha256="93843c433deb3f3347cb261a7458cdba827b96225f828413f3aa1b42489af548"
dt_cargo_speed_sha256="91942c997045fbeded961e272b32cf78cfba9571c4ea7ac34e86d77bf315f9a6"
scania_sha256="5504d0402f54faaf97ac0ca085a621645763f5cfea2eb29c592b057d43d4db89"
deviation_sha256="8f17d15e189c99a34f2446daeaaf1a65d1e79045a94c4d3c452391c96d9cee00"
tlc_sha256="c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510"
athens_sha256="f28906181e925fa1697b069febff350a331ee8cd73d99cd678de661a350a5005"
road_md5="cab184cfc2fe12c0834bc46188c0f330"
lade_revision="be2cec02775cafc8d52230303f32134382bcc50b"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

log() {
  printf '[real-data] %s\n' "$*"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command is unavailable: $1"
}

require_acknowledgement() {
  local variable_name="$1"
  local explanation="$2"
  if [[ "${!variable_name:-}" != "1" ]]; then
    die "${explanation} Set ${variable_name}=1 only after reviewing the source terms."
  fi
}

file_bytes() {
  wc -c < "$1" | tr -d '[:space:]'
}

verify_sha256() {
  local target_file="$1"
  local expected_digest="$2"
  printf '%s  %s\n' "${expected_digest}" "${target_file}" | sha256sum --check --status -
}

verify_md5() {
  local target_file="$1"
  local expected_digest="$2"
  printf '%s  %s\n' "${expected_digest}" "${target_file}" | md5sum --check --status -
}

verify_locked_file() {
  local target_file="$1"
  local digest_algorithm="$2"
  local expected_digest="$3"
  local expected_bytes="$4"

  [[ -f "${target_file}" ]] || return 1

  if [[ -n "${expected_bytes}" ]] && [[ "$(file_bytes "${target_file}")" != "${expected_bytes}" ]]; then
    return 1
  fi

  case "${digest_algorithm}" in
    sha256)
      verify_sha256 "${target_file}" "${expected_digest}"
      ;;
    md5)
      verify_md5 "${target_file}" "${expected_digest}"
      ;;
    size-only)
      [[ -n "${expected_bytes}" ]]
      ;;
    *)
      die "unsupported digest algorithm: ${digest_algorithm}"
      ;;
  esac
}

fetch_locked_file() {
  local source_name="$1"
  local source_url="$2"
  local relative_target="$3"
  local digest_algorithm="$4"
  local expected_digest="$5"
  local expected_bytes="$6"
  local target_file="${raw_dir}/${relative_target}"
  local current_bytes="0"

  require_command curl
  mkdir -p "$(dirname "${target_file}")"

  if verify_locked_file "${target_file}" "${digest_algorithm}" "${expected_digest}" "${expected_bytes}"; then
    log "${source_name}: verified artifact already exists; skipping"
    return 0
  fi

  if [[ -f "${target_file}" ]]; then
    current_bytes="$(file_bytes "${target_file}")"
    if [[ -n "${expected_bytes}" ]] && (( current_bytes >= expected_bytes )); then
      die "${source_name}: existing artifact is complete-sized but failed verification. Move it aside explicitly before retrying: ${target_file}"
    fi
    log "${source_name}: resuming partial artifact at ${current_bytes} bytes"
    curl --fail --location --retry 4 --retry-delay 3 --connect-timeout 30 \
      --continue-at - --output "${target_file}" "${source_url}"
  else
    log "${source_name}: downloading ${source_url}"
    curl --fail --location --retry 4 --retry-delay 3 --connect-timeout 30 \
      --output "${target_file}" "${source_url}"
  fi

  verify_locked_file "${target_file}" "${digest_algorithm}" "${expected_digest}" "${expected_bytes}" || \
    die "${source_name}: downloaded artifact failed ${digest_algorithm} verification: ${target_file}"
  log "${source_name}: verified ${target_file}"
}

download_singapore() {
  fetch_locked_file \
    "singapore-commercial-vehicle" \
    "https://ndownloader.figshare.com/files/24337976" \
    "singapore-commercial-vehicle/figshare_TRR.zip" \
    "sha256" "${singapore_sha256}" "46804275"
}

download_dt_cargo() {
  fetch_locked_file \
    "dt-cargo-fleet" \
    "https://raw.githubusercontent.com/TUMFTM/dt-cargo/805c534c73ed4d247babd053f60468b486f92519/input/public/fleet.csv" \
    "dt-cargo/fleet.csv" \
    "sha256" "${dt_cargo_fleet_sha256}" "1130"
  fetch_locked_file \
    "dt-cargo-tracks" \
    "https://raw.githubusercontent.com/TUMFTM/dt-cargo/805c534c73ed4d247babd053f60468b486f92519/input/public/tracks.csv" \
    "dt-cargo/tracks.csv" \
    "sha256" "${dt_cargo_tracks_sha256}" "16381647"
  fetch_locked_file \
    "dt-cargo-example-speed" \
    "https://raw.githubusercontent.com/TUMFTM/dt-cargo/805c534c73ed4d247babd053f60468b486f92519/input/public/speed.zip" \
    "dt-cargo/speed.zip" \
    "sha256" "${dt_cargo_speed_sha256}" "40734"
}

download_vius() {
  fetch_locked_file \
    "vius-2021-puf" \
    "https://www2.census.gov/programs-surveys/vius/datasets/2021/vius_2021_puf_csv.zip" \
    "vius-2021/vius_2021_puf_csv.zip" \
    "sha256" "${vius_sha256}" "3528049"
}

download_scania() {
  fetch_locked_file \
    "scania-aps-failure" \
    "https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip" \
    "scania-aps/aps+failure+at+scania+trucks.zip" \
    "sha256" "${scania_sha256}" "56618034"
}

download_cfs() {
  fetch_locked_file \
    "cfs-2022-pums" \
    "https://www2.census.gov/programs-surveys/cfs/datasets/2022/cfs_2022_pums.zip" \
    "cfs-2022/cfs_2022_pums.zip" \
    "size-only" "" "677530696"
  log "cfs-2022-pums: source publishes no digest in the verified ledger; record a local SHA-256 before processing"
}

download_amazon() {
  require_acknowledgement \
    "ACCEPT_AMAZON_CC_BY_NC_4_0" \
    "Amazon Last Mile data are CC BY-NC 4.0 and restricted to non-commercial use."
  local base="https://amazon-last-mile-challenges.s3.us-west-2.amazonaws.com/almrrc2021"
  fetch_locked_file "amazon-train-route" \
    "${base}/almrrc2021-data-training/model_build_inputs/route_data.json" \
    "amazon/train/route_data.json" sha256 \
    "da3a5d4e73b683d756111f0d9e6d3f20eb8e82adf7c89ee6ee0cb2239abd2f38" "78972162"
  fetch_locked_file "amazon-train-package" \
    "${base}/almrrc2021-data-training/model_build_inputs/package_data.json" \
    "amazon/train/package_data.json" sha256 \
    "9ac858358e9f43d34cb65198c03601c814328a06dec9b71ee88b40aaec7f0966" "375437806"
  fetch_locked_file "amazon-train-actual" \
    "${base}/almrrc2021-data-training/model_build_inputs/actual_sequences.json" \
    "amazon/train/actual_sequences.json" sha256 \
    "3de44067242e8fa841d4b852db8809afaa23fe852e452ee4a2f54f0f0fb77e3c" "9665078"
  fetch_locked_file "amazon-eval-route" \
    "${base}/almrrc2021-data-evaluation/model_apply_inputs/eval_route_data.json" \
    "amazon/eval/route_data.json" sha256 \
    "dfd8a198b9df5e3b194d3a268f6f442edcd77f028870d54a8d7397a7bc366466" "37777768"
  fetch_locked_file "amazon-eval-package" \
    "${base}/almrrc2021-data-evaluation/model_apply_inputs/eval_package_data.json" \
    "amazon/eval/package_data.json" sha256 \
    "612f9aae67df34e637b106a6e2b1cd627fee11fdc0107937bdb56f8c61ec7848" "166201035"
  fetch_locked_file "amazon-eval-actual" \
    "${base}/almrrc2021-data-evaluation/model_score_inputs/eval_actual_sequences.json" \
    "amazon/eval/actual_sequences.json" sha256 \
    "30e924b5f263bd5e0a4bb58a756c8f6767b49d59a6bfa718ce26d2767a2c90e9" "4625218"
}

download_lade() {
  local target_dir="${raw_dir}/lade"
  require_acknowledgement \
    "ACCEPT_LADE_RESEARCH_TERMS" \
    "LaDe metadata declares Apache-2.0, but its README separately limits the stated use to research and asks users to read terms."
  mkdir -p "${target_dir}"

  if command -v hf >/dev/null 2>&1; then
    log "lade: downloading pinned Hugging Face revision ${lade_revision}"
    hf download "Cainiao-AI/LaDe" \
      --repo-type dataset \
      --revision "${lade_revision}" \
      --local-dir "${target_dir}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    log "lade: downloading pinned Hugging Face revision ${lade_revision}"
    huggingface-cli download "Cainiao-AI/LaDe" \
      --repo-type dataset \
      --revision "${lade_revision}" \
      --local-dir "${target_dir}"
  else
    die "LaDe requires the official 'hf' or 'huggingface-cli' downloader"
  fi
}

download_athens() {
  fetch_locked_file \
    "athens-pharmaceutical-3pl" \
    "https://zenodo.org/api/records/15310106/files/data.zip/content" \
    "athens/data.zip" \
    "sha256" "${athens_sha256}" "2926700"
}

download_road() {
  fetch_locked_file \
    "road-can-ids" \
    "https://zenodo.org/records/10462796/files/road.zip?download=1" \
    "road-can-ids/road.zip" \
    "md5" "${road_md5}" ""
}

download_deviation() {
  fetch_locked_file \
    "route-deviation-mendeley" \
    "https://data.mendeley.com/public-files/datasets/kkwgfvmtxn/files/388ea249-0830-4906-a39e-7bcbd5ee837a/file_downloaded" \
    "route-deviation/routes_performance.xlsx" \
    "sha256" "${deviation_sha256}" "19713337"
}

download_olist() {
  local target_dir="${raw_dir}/olist"
  local completion_marker="${target_dir}/.download-complete"
  require_acknowledgement \
    "ACCEPT_OLIST_DATASET_TERMS" \
    "Review and capture the authoritative license shown on the authenticated Olist Kaggle dataset page."
  require_command kaggle
  mkdir -p "${target_dir}"
  if [[ -f "${completion_marker}" ]]; then
    log "olist: completion marker exists; skipping"
    return 0
  fi
  kaggle datasets download \
    --dataset "olistbr/brazilian-ecommerce" \
    --path "${target_dir}"
  touch "${completion_marker}"
  log "olist: downloaded; capture Kaggle version, file hashes, and displayed license before processing"
}

download_tlc() {
  fetch_locked_file \
    "nyc-tlc-yellow-2024-01" \
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet" \
    "nyc-tlc/yellow_tripdata_2024-01.parquet" \
    "sha256" "${tlc_sha256}" "49961641"
}

download_osm() {
  local target_dir="${raw_dir}/osm-indonesia"
  local target_file="${target_dir}/indonesia-latest.osm.pbf"
  local checksum_file="${target_dir}/indonesia-latest.osm.pbf.md5"
  local expected_digest=""
  require_command curl
  mkdir -p "${target_dir}"

  curl --fail --location --retry 4 --connect-timeout 30 \
    --output "${checksum_file}" \
    "https://download.geofabrik.de/asia/indonesia-latest.osm.pbf.md5"
  expected_digest="$(awk 'NR == 1 {print $1}' "${checksum_file}")"
  [[ "${expected_digest}" =~ ^[0-9a-fA-F]{32}$ ]] || die "osm-indonesia: provider MD5 sidecar is invalid"

  if [[ -f "${target_file}" ]]; then
    if verify_md5 "${target_file}" "${expected_digest}"; then
      log "osm-indonesia: current provider snapshot is already verified; skipping"
      return 0
    fi
    die "osm-indonesia: local 'latest' does not match the current provider sidecar. Archive it under a dated manifest before downloading a new snapshot."
  fi

  curl --fail --location --retry 4 --retry-delay 3 --connect-timeout 30 \
    --output "${target_file}" \
    "https://download.geofabrik.de/asia/indonesia-latest.osm.pbf"
  verify_md5 "${target_file}" "${expected_digest}" || die "osm-indonesia: provider MD5 verification failed"
  log "osm-indonesia: verified against the provider sidecar; record this MD5 and a local SHA-256 in run evidence"
}

list_sources() {
  printf '%s\n' \
    "singapore" \
    "dt-cargo" \
    "vius" \
    "scania" \
    "cfs" \
    "amazon" \
    "lade" \
    "athens" \
    "road" \
    "deviation" \
    "olist" \
    "tlc" \
    "osm"
}

usage() {
  cat <<'USAGE'
Usage: data_sources/download_real_data.sh SOURCE [SOURCE ...]

Sources:
  singapore  Primary commercial-vehicle GPS/OBD/payload/fuel ZIP
  dt-cargo   Pinned 16.4 MB track/fleet tables plus one real speed trace
  vius       2021 U.S. truck inventory/use public-use file
  scania     Real Scania APS data; review UCI CC BY 4.0 and bundled GPLv3 notice
  cfs        2022 U.S. commodity flow microdata (about 678 MB)
  amazon     Real last-mile routes; requires ACCEPT_AMAZON_CC_BY_NC_4_0=1
  lade       Courier/order data; requires ACCEPT_LADE_RESEARCH_TERMS=1
  athens     Nine real pharmaceutical 3PL VRPTW days; external audit only
  road       Real CAN intrusion corpus (about 557 MB)
  deviation  Planned-versus-driven route workbook
  olist      Real parcel freight values; requires ACCEPT_OLIST_DATASET_TERMS=1
  tlc        January 2024 NYC yellow-taxi trips and metered fares
  osm        Current Indonesia OSM extract, verified with Geofabrik sidecar
  small      singapore + vius + deviation
  core       current training stack; Amazon and LaDe acknowledgements required
  all        every source above; large and subject to license acknowledgements
  --list     print source identifiers

Raw files are written beneath data_sources/raw/ by default and are Git-ignored.
Set REAL_DATA_RAW_DIR to use a different explicit storage location.
USAGE
}

if (( $# == 0 )); then
  usage
  exit 2
fi

requested_sources=("$@")
expanded_sources=()
for requested_source in "${requested_sources[@]}"; do
  case "${requested_source}" in
    --list)
      list_sources
      exit 0
      ;;
    small)
      expanded_sources+=(singapore vius deviation)
      ;;
    core)
      expanded_sources+=(singapore dt-cargo vius scania amazon lade tlc)
      ;;
    all)
      expanded_sources+=(singapore dt-cargo vius scania cfs amazon lade athens road deviation olist tlc osm)
      ;;
    *)
      expanded_sources+=("${requested_source}")
      ;;
  esac
done

mkdir -p "${raw_dir}"
for source_name in "${expanded_sources[@]}"; do
  case "${source_name}" in
    singapore) download_singapore ;;
    dt-cargo) download_dt_cargo ;;
    vius) download_vius ;;
    scania) download_scania ;;
    cfs) download_cfs ;;
    amazon) download_amazon ;;
    lade) download_lade ;;
    athens) download_athens ;;
    road) download_road ;;
    deviation) download_deviation ;;
    olist) download_olist ;;
    tlc) download_tlc ;;
    osm) download_osm ;;
    *)
      usage >&2
      die "unknown source: ${source_name}"
      ;;
  esac
done
