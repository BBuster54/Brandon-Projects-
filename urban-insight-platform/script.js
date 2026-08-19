class UrbanInsightDashboard {
  constructor() {
    this.currentCity = "New York";
    this.currentView = "overview";
    this.charts = [];
    this.cities = {};
  }

  async init() {
    try {
      const response = await fetch("data/sample-data.json");
      if (!response.ok) throw new Error("The sample dataset could not be loaded.");
      const dataset = await response.json();
      this.cities = dataset.cities;
      this.populateCitySelector();
      this.bindEvents();
      this.render();
    } catch (error) {
      document.getElementById("mainContent").innerHTML = `<section class="card"><h2>Unable to load the dashboard data</h2><p>${error.message} Run this project through a local server (for example, VS Code Live Server) rather than opening the file directly.</p></section>`;
    }
  }

  get city() { return this.cities[this.currentCity]; }

  bindEvents() {
    document.getElementById("citySelector").addEventListener("change", (event) => {
      this.currentCity = event.target.value;
      this.render();
    });
    document.getElementById("refreshData").addEventListener("click", () => this.render());
    document.querySelectorAll(".nav-links a").forEach((link) => {
      link.addEventListener("click", (event) => {
        event.preventDefault();
        this.currentView = link.dataset.view;
        document.querySelectorAll(".nav-links a").forEach((item) => item.classList.remove("active"));
        link.classList.add("active");
        this.render();
      });
    });
  }

  populateCitySelector() {
    document.getElementById("citySelector").innerHTML = Object.entries(this.cities)
      .map(([key, city]) => `<option value="${key}">${city.name}, ${city.state}</option>`)
      .join("");
  }

  destroyCharts() { this.charts.forEach((chart) => chart.destroy()); this.charts = []; }

  render() {
    this.destroyCharts();
    const container = document.getElementById("mainContent");
    ({ overview: () => this.renderOverview(container), environment: () => this.renderEnvironment(container), transportation: () => this.renderTransportation(container), housing: () => this.renderHousing(container) })[this.currentView]();
  }

  metricCard(value, label, detail, tone = "positive") {
    return `<article class="metric-card"><div class="metric-value">${value}</div><div class="metric-label">${label}</div><div class="metric-change ${tone}">${detail}</div></article>`;
  }

  chartCard(title, canvasId) { return `<article class="card"><h3>${title}</h3><div class="chart-container"><canvas id="${canvasId}"></canvas></div></article>`; }

  renderOverview(container) {
    const { metrics, name, state } = this.city;
    container.innerHTML = `<div class="dashboard-intro"><p class="eyebrow">Selected city</p><h2>${name}, ${state}</h2><p>Compare environmental, mobility, and housing indicators from the bundled prototype dataset.</p></div><section class="metrics-grid">${this.metricCard(metrics.airQuality.aqi, "Air Quality Index", this.aqiLabel(metrics.airQuality.aqi), this.statusTone(metrics.airQuality.aqi))}${this.metricCard(`${metrics.transportation.transitUsage}%`, "Transit Usage", `${metrics.transportation.publicTransitScore}/100 transit score`)}${this.metricCard(metrics.transportation.evStations, "EV Charging Stations", "Reported station count")}${this.metricCard(`${metrics.environment.greenSpace}%`, "Green Space Coverage", `${metrics.environment.treeCanopy}% tree canopy`)}</section><section class="dashboard-grid">${this.chartCard("City indicator comparison", "overviewChart")}${this.chartCard("Air quality components", "airChart")}<article class="card map-summary"><h3>City profile</h3><dl><div><dt>Population</dt><dd>${formatNumber(this.city.population)}</dd></div><div><dt>Area</dt><dd>${this.city.area.toLocaleString()} sq mi</dd></div><div><dt>Average commute</dt><dd>${metrics.transportation.avgCommute} min</dd></div><div><dt>Water quality</dt><dd>${metrics.environment.waterQuality}/100</dd></div></dl><p class="data-note">Prototype data snapshot: July 14, 2024. This dashboard does not claim to provide live readings.</p></article>${this.chartCard("Metrics across included cities", "cityComparisonChart")}</section>`;
    this.createOverviewCharts();
  }

  renderEnvironment(container) {
    const { airQuality, environment } = this.city.metrics;
    container.innerHTML = `<div class="dashboard-intro"><p class="eyebrow">Environment</p><h2>${this.city.name}</h2><p>Air and environmental indicators from the prototype dataset.</p></div><section class="metrics-grid">${this.metricCard(airQuality.aqi, "Air Quality Index", this.aqiLabel(airQuality.aqi), this.statusTone(airQuality.aqi))}${this.metricCard(`${airQuality.pm25} μg/m³`, "PM2.5", this.pmLabel(airQuality.pm25), this.pmTone(airQuality.pm25))}${this.metricCard(`${environment.noiseLevel} dB`, "Noise Level", "Estimated citywide level", environment.noiseLevel > 70 ? "negative" : "positive")}${this.metricCard(`${environment.energyEfficiency}/100`, "Energy Efficiency", "Composite indicator")}</section><section class="dashboard-grid">${this.chartCard("Air quality components", "environmentAirChart")}${this.chartCard("Environmental scorecard", "environmentChart")}<article class="card"><h3>Measurement notes</h3><div class="data-grid"><table class="data-table"><tbody><tr><th>Ozone</th><td>${airQuality.ozone} ppb</td></tr><tr><th>NO₂</th><td>${airQuality.no2} ppb</td></tr><tr><th>Waste recycling</th><td>${environment.wasteRecycling}%</td></tr><tr><th>Tree canopy</th><td>${environment.treeCanopy}%</td></tr></tbody></table></div></article></section>`;
    this.createEnvironmentCharts();
  }

  renderTransportation(container) {
    const t = this.city.metrics.transportation;
    container.innerHTML = `<div class="dashboard-intro"><p class="eyebrow">Transportation</p><h2>${this.city.name}</h2><p>Mobility and infrastructure indicators from the prototype dataset.</p></div><section class="metrics-grid">${this.metricCard(`${t.transitUsage}%`, "Transit Usage", `${t.publicTransitScore}/100 transit score`)}${this.metricCard(`${t.trafficCongestion}%`, "Traffic Congestion", "Composite congestion indicator", "negative")}${this.metricCard(t.evStations, "EV Charging Stations", "Reported station count")}${this.metricCard(`${t.bikeUsage}%`, "Bike Usage", `${t.walkScore}/100 walk score`)}</section><section class="dashboard-grid">${this.chartCard("Mobility mode profile", "modeChart")}${this.chartCard("Infrastructure by city", "transportChart")}</section>`;
    this.createTransportationCharts();
  }

  renderHousing(container) {
    const h = this.city.metrics.housing;
    container.innerHTML = `<div class="dashboard-intro"><p class="eyebrow">Housing</p><h2>${this.city.name}</h2><p>Housing indicators from the prototype dataset.</p></div><section class="metrics-grid">${this.metricCard(h.density, "Housing Density", "Units per acre")}${this.metricCard(`${h.affordability}/100`, "Affordability Index", "Composite indicator", "negative")}${this.metricCard(h.construction, "New Construction", "Units in dataset")}${this.metricCard(formatCurrency(h.medianPrice), "Median Home Price", `${h.rentBurden}% rent burden`, "negative")}</section><section class="dashboard-grid">${this.chartCard("Housing profile", "housingChart")}${this.chartCard("Affordability across cities", "housingComparisonChart")}</section>`;
    this.createHousingCharts();
  }

  createChart(canvasId, config) { const canvas = document.getElementById(canvasId); if (canvas) this.charts.push(new Chart(canvas, config)); }
  createOverviewCharts() {
    const { metrics } = this.city;
    this.createChart("overviewChart", this.barChart(["AQI", "Transit", "Energy", "Green space"], [metrics.airQuality.aqi, metrics.transportation.transitUsage, metrics.environment.energyEfficiency, metrics.environment.greenSpace], "Selected-city indicator score", "#2563eb"));
    this.createChart("airChart", this.barChart(["PM2.5", "PM10", "Ozone", "NO₂"], [metrics.airQuality.pm25, metrics.airQuality.pm10, metrics.airQuality.ozone, metrics.airQuality.no2], "Pollutant level", "#f59e0b"));
    this.createChart("cityComparisonChart", this.lineChart(Object.values(this.cities).map((city) => city.name), Object.values(this.cities).map((city) => city.metrics.airQuality.aqi), "Air Quality Index", "#ef4444"));
  }
  createEnvironmentCharts() {
    const { airQuality, environment } = this.city.metrics;
    this.createChart("environmentAirChart", this.barChart(["PM2.5", "PM10", "Ozone", "NO₂", "SO₂"], [airQuality.pm25, airQuality.pm10, airQuality.ozone, airQuality.no2, airQuality.so2], "Pollutant level", "#f59e0b"));
    this.createChart("environmentChart", this.barChart(["Energy", "Water", "Recycling", "Green space", "Tree canopy"], [environment.energyEfficiency, environment.waterQuality, environment.wasteRecycling, environment.greenSpace, environment.treeCanopy], "Score / coverage (%)", "#10b981"));
  }
  createTransportationCharts() {
    const t = this.city.metrics.transportation;
    const privateCar = Math.max(0, 100 - t.transitUsage - t.bikeUsage - 10);
    this.createChart("modeChart", { type: "doughnut", data: { labels: ["Transit", "Private car", "Bike / walk", "Other"], datasets: [{ data: [t.transitUsage, privateCar, t.bikeUsage, 10], backgroundColor: ["#2563eb", "#ef4444", "#10b981", "#f59e0b"], borderWidth: 0 }] }, options: this.chartOptions() });
    this.createChart("transportChart", this.barChart(Object.values(this.cities).map((city) => city.name), Object.values(this.cities).map((city) => city.metrics.transportation.evStations), "EV charging stations", "#8b5cf6"));
  }
  createHousingCharts() {
    const h = this.city.metrics.housing;
    this.createChart("housingChart", this.barChart(["Density", "Affordability", "Construction", "Rent burden"], [h.density, h.affordability, h.construction / 2, h.rentBurden], "Normalized indicator", "#06b6d4"));
    this.createChart("housingComparisonChart", this.barChart(Object.values(this.cities).map((city) => city.name), Object.values(this.cities).map((city) => city.metrics.housing.affordability), "Affordability index", "#2563eb"));
  }
  barChart(labels, data, label, color) { return { type: "bar", data: { labels, datasets: [{ label, data, backgroundColor: color, borderRadius: 6 }] }, options: this.chartOptions() }; }
  lineChart(labels, data, label, color) { return { type: "line", data: { labels, datasets: [{ label, data, borderColor: color, backgroundColor: `${color}22`, fill: true, tension: 0.35 }] }, options: this.chartOptions() }; }
  chartOptions() { return { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, grid: { color: "#e5e7eb" } }, x: { grid: { display: false } } } }; }
  aqiLabel(aqi) { return aqi <= 50 ? "Good" : aqi <= 100 ? "Moderate" : "Unhealthy for sensitive groups"; }
  statusTone(aqi) { return aqi <= 50 ? "positive" : aqi <= 100 ? "neutral" : "negative"; }
  pmLabel(pm) { return pm <= 35 ? "Lower concentration" : pm <= 75 ? "Moderate concentration" : "Higher concentration"; }
  pmTone(pm) { return pm <= 35 ? "positive" : pm <= 75 ? "neutral" : "negative"; }
}

function formatNumber(value) { return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value); }
function formatCurrency(value) { return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value); }

document.addEventListener("DOMContentLoaded", () => {
  if (typeof Chart === "undefined") {
    document.getElementById("mainContent").innerHTML = "<p>Chart.js did not load. Check your internet connection and reload.</p>";
    return;
  }
  new UrbanInsightDashboard().init();
});
