const state = { fleet: [], recommendations: [], selectedId: null, map: null, mapLayers: [] };
const api = (path, options = {}) => fetch(`/api/v1${path}`, { headers: { "Content-Type": "application/json" }, ...options }).then(async response => {
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || body.reason || `Request failed (${response.status})`);
  return body;
});

const rupiah = new Intl.NumberFormat("id-ID", { style: "currency", currency: "IDR", maximumFractionDigits: 0 });
const compactRupiah = amount => amount >= 1_000_000 ? `Rp${(amount / 1_000_000).toFixed(1)}m` : rupiah.format(amount);
const mins = value => value === 0 ? "Now" : `${Math.round(value / 60)}h ${value % 60}m`;
const esc = value => String(value).replace(/[&<>"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[char]));

function toast(message) {
  const element = document.getElementById("toast");
  element.textContent = message;
  element.classList.add("show");
  window.clearTimeout(window.__toastTimer);
  window.__toastTimer = window.setTimeout(() => element.classList.remove("show"), 3300);
}

function riskClass(level) { return `risk ${level || "low"}`; }

function initMap() {
  if (!window.L || state.map) return;
  state.map = L.map("map", { zoomControl: false }).setView([-6.75, 109.45], 7);
  L.control.zoom({ position: "bottomright" }).addTo(state.map);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 18, attribution: "© OpenStreetMap contributors" }).addTo(state.map);
}

function clearMapLayers() {
  state.mapLayers.forEach(layer => layer.remove());
  state.mapLayers = [];
}

function renderMap() {
  if (!state.map) return;
  clearMapLayers();
  const selected = state.recommendations.find(plan => plan.id === state.selectedId);
  state.fleet.forEach(truck => {
    const risk = truck.empty_return_risk.level;
    const color = risk === "high" ? "#ff6d7a" : risk === "medium" ? "#ffba45" : "#67d99c";
    const layer = L.circleMarker([truck.position.lat, truck.position.lon], { radius: 7, color, fillColor: color, fillOpacity: 1, weight: 2 })
      .bindTooltip(`${truck.name}<br>${Math.round(truck.empty_return_risk.probability * 100)}% empty-return risk`);
    layer.addTo(state.map); state.mapLayers.push(layer);
  });
  if (selected) {
    const coordinates = selected.geometry.map(point => [point.lat, point.lon]);
    const route = L.polyline(coordinates, { color: "#33d6cd", weight: 4, opacity: .92, dashArray: selected.is_multi_hop ? "8 7" : null }).addTo(state.map);
    state.mapLayers.push(route);
    selected.stops.forEach((stop, index) => {
      const marker = L.circleMarker([stop.lat, stop.lon], { radius: 6, color: "#edf5ff", fillColor: index === 0 ? "#5b95ff" : "#33d6cd", fillOpacity: 1, weight: 1 }).bindTooltip(`${stop.kind}: ${stop.name}`);
      marker.addTo(state.map); state.mapLayers.push(marker);
    });
    state.map.fitBounds(route.getBounds().pad(.16), { maxZoom: 9 });
    document.getElementById("map-title").textContent = selected.is_multi_hop ? "Proposed multi-hop backhaul" : "Proposed backhaul route";
    document.getElementById("selection-label").textContent = `${selected.truck_name} · ${selected.order_ids.join(", ")}`;
  } else {
    document.getElementById("map-title").textContent = "Java logistics network";
    document.getElementById("selection-label").textContent = "Select a recommendation";
  }
}

function renderMetrics(metrics) {
  const rows = [
    ["Fleet", metrics.fleet_total, "vehicles reporting"],
    ["Empty-return exposure", metrics.fleet_at_empty_risk, "risk ≥ 50%"],
    ["Open cargo", metrics.open_orders, "eligible marketplace loads"],
    ["Recoverable margin", compactRupiah(metrics.recoverable_margin_idr), "proposed opportunities"],
    ["Verified telemetry", metrics.telemetry_accepted, `${metrics.dispatcher_decisions} dispatcher decision(s)`],
  ];
  document.getElementById("metric-grid").innerHTML = rows.map(([label, value, subtext]) => `<article class="metric"><div class="metric-label">${label}</div><div class="metric-value">${value}</div><div class="metric-subtext">${subtext}</div></article>`).join("");
  const google = document.getElementById("google-status");
  google.textContent = metrics.google_routes_configured ? "Google Routes key configured" : "Google Routes not configured";
  google.className = `chip ${metrics.google_routes_configured ? "success" : "warning"}`;
  if (metrics.iot_demo_secret_warning) google.title = "Set IOT_SHARED_SECRET before connecting a real device.";
}

function renderFleet() {
  document.getElementById("fleet-count").textContent = `${state.fleet.length} live`;
  document.getElementById("fleet-list").innerHTML = state.fleet.map(truck => {
    const risk = truck.empty_return_risk;
    const anomaly = truck.anomaly;
    return `<article class="fleet-row"><div class="fleet-top"><div><div class="fleet-name">${esc(truck.name)}</div><div class="fleet-meta">${esc(truck.vehicle_type)} · ${truck.capacity_kg.toLocaleString()} kg · fuel ${Math.round(truck.fuel_pct)}%</div></div><span class="${riskClass(risk.level)}">${Math.round(risk.probability * 100)}% risk</span></div><div class="fleet-meta">${esc(truck.status.replaceAll("_", " "))} · ETA ${mins(truck.eta.p50_min)} · ${esc(anomaly.status.replaceAll("_", " "))}</div></article>`;
  }).join("");
}

function renderRecommendations() {
  const list = document.getElementById("recommendation-list");
  document.getElementById("recommendation-count").textContent = `${state.recommendations.length} ranked`;
  if (!state.recommendations.length) { list.innerHTML = `<div class="empty-state">No compatible cargo remains in the current marketplace scenario.</div>`; return; }
  list.innerHTML = state.recommendations.map(plan => `<article class="recommendation-row ${plan.id === state.selectedId ? "selected" : ""}" data-plan="${esc(plan.id)}"><div class="recommendation-top"><div><div class="rec-title">${plan.is_multi_hop ? "Multi-hop · " : ""}${esc(plan.cargo_summary)}</div><div class="rec-meta">${esc(plan.truck_name)} · ${plan.distance_km} km · ${plan.capacity_pct}% capacity</div></div><span class="chip ${plan.status === "accepted" ? "accepted" : "muted"}">${esc(plan.status)}</span></div><div class="rec-bottom"><span class="margin">${compactRupiah(plan.expected_margin_idr)}</span><span class="confidence">${Math.round(plan.confidence * 100)}% confidence</span></div></article>`).join("");
  list.querySelectorAll("[data-plan]").forEach(item => item.addEventListener("click", () => selectPlan(item.dataset.plan)));
}

function renderDetail() {
  const panel = document.getElementById("recommendation-detail");
  const status = document.getElementById("detail-status");
  const plan = state.recommendations.find(candidate => candidate.id === state.selectedId);
  if (!plan) { status.textContent = "No selection"; status.className = "chip muted"; panel.className = "empty-state"; panel.textContent = "Select a recommendation to inspect cargo, route, ETA, cost, margin, and confidence."; return; }
  status.textContent = plan.status; status.className = `chip ${plan.status === "accepted" ? "accepted" : "warning"}`;
  const stops = plan.stops.map(stop => `<span class="stop"><b>${esc(stop.kind)}</b><br>${esc(stop.name)}</span>`).join("");
  panel.className = "detail-content";
  panel.innerHTML = `<div class="detail-summary"><div><div class="detail-cargo">${esc(plan.cargo_summary)}</div><div class="detail-subtitle">${esc(plan.truck_name)} · ${plan.order_ids.map(esc).join(" + ")} · ${plan.is_multi_hop ? "multi-hop" : "direct"}</div></div><div class="detail-actions">${plan.status === "proposed" ? `<button class="primary" data-action="accept">Accept plan</button><button class="danger" data-action="reject">Reject</button>` : ""}<button class="traffic" data-action="traffic">Check live traffic</button></div></div><div class="stat-grid"><div class="stat"><label>Final ETA (P50)</label><strong>${mins(plan.eta_final_delivery_min)}</strong></div><div class="stat"><label>Operating cost</label><strong>${compactRupiah(plan.operating_cost_idr)}</strong></div><div class="stat"><label>Expected margin</label><strong class="margin">${compactRupiah(plan.expected_margin_idr)}</strong></div><div class="stat"><label>Minimum quote</label><strong>${compactRupiah(plan.minimum_viable_quote_idr)}</strong></div></div><div class="route-stops">${stops}</div><ul class="explanation">${plan.explanation.map(item => `<li>${esc(item)}</li>`).join("")}</ul>`;
  panel.querySelector('[data-action="accept"]')?.addEventListener("click", () => decide(plan.id, "accept"));
  panel.querySelector('[data-action="reject"]')?.addEventListener("click", () => decide(plan.id, "reject"));
  panel.querySelector('[data-action="traffic"]')?.addEventListener("click", () => checkTraffic(plan.id));
}

function selectPlan(id) { state.selectedId = id; renderRecommendations(); renderDetail(); renderMap(); document.getElementById("traffic-result").className = "traffic-result"; document.getElementById("traffic-result").textContent = "Choose a recommendation, then request a one-time live traffic confirmation."; }

async function decide(id, action) {
  try {
    const response = await api(`/recommendations/${encodeURIComponent(id)}/decision`, { method: "POST", body: JSON.stringify({ action }) });
    toast(`${response.recommendation.id} ${action}ed by dispatcher.`);
    await refresh();
  } catch (error) { toast(error.message); }
}

async function checkTraffic(id) {
  const output = document.getElementById("traffic-result");
  output.className = "traffic-result"; output.textContent = "Checking Google live traffic…";
  try {
    const data = await api(`/recommendations/${encodeURIComponent(id)}/live-traffic`);
    output.textContent = `${data.provider}: live ETA ${mins(data.live_eta_min)}${data.static_eta_min !== null ? ` (baseline ${mins(data.static_eta_min)}, +${data.traffic_delay_min} min traffic)` : ""}. ${data.notice}`;
  } catch (error) { output.className = "traffic-result error"; output.textContent = error.message; }
}

async function simulate() {
  const button = document.getElementById("simulate-button"); button.disabled = true;
  try { const result = await api("/simulation/tick", { method: "POST", body: "{}" }); toast(`${result.events.filter(event => event.accepted).length} signed IoT event(s) accepted.`); await refresh(); }
  catch (error) { toast(error.message); }
  finally { button.disabled = false; }
}

async function refresh() {
  try {
    const [metrics, fleet, recommendations] = await Promise.all([api("/metrics"), api("/fleet"), api("/recommendations")]);
    state.fleet = fleet.fleet; state.recommendations = recommendations.recommendations;
    if (!state.selectedId || !state.recommendations.some(plan => plan.id === state.selectedId)) state.selectedId = state.recommendations[0]?.id || null;
    renderMetrics(metrics); renderFleet(); renderRecommendations(); renderDetail(); renderMap();
  } catch (error) { toast(`Dashboard failed to refresh: ${error.message}`); }
}

document.getElementById("refresh-button").addEventListener("click", refresh);
document.getElementById("simulate-button").addEventListener("click", simulate);
initMap(); refresh();
