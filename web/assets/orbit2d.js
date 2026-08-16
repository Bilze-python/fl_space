// Natural Earth map renderer for the primary 2D orbit view.
// It receives the same replay data object that the Cesium viewer consumes.

(() => {
  const renderer = {
    world: null,
    loading: null,
    orbit: null,
    slot: 0,
    svg: null,
    projection: null,
    path: null,
  };

  function getSvg(id) {
    const element = document.getElementById(id);
    if (!element || typeof d3 === "undefined" || typeof topojson === "undefined") return null;
    if (renderer.svg?.node() !== element) renderer.svg = d3.select(element);
    return renderer.svg;
  }

  async function loadWorld() {
    if (renderer.world) return renderer.world;
    if (!renderer.loading) {
      renderer.loading = d3.json("/vendor/world-atlas/countries-110m.json")
        .then((world) => {
          renderer.world = world;
          return world;
        })
        .catch((error) => {
          console.error("Unable to load local Natural Earth map", error);
          return null;
        });
    }
    return renderer.loading;
  }

  function lineFeature(points) {
    const segments = [];
    let segment = [];
    for (const point of points || []) {
      if (!point) continue;
      const previous = segment[segment.length - 1];
      if (previous && Math.abs(point.lon - previous[0]) > 180) {
        if (segment.length > 1) segments.push(segment);
        segment = [];
      }
      segment.push([point.lon, point.lat]);
    }
    if (segment.length > 1) segments.push(segment);
    return { type: "MultiLineString", coordinates: segments };
  }

  function resize() {
    const svg = renderer.svg;
    if (!svg) return false;
    const node = svg.node();
    const rect = node.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width || node.clientWidth || 960));
    const height = Math.max(1, Math.round(rect.height || node.clientHeight || 540));
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("preserveAspectRatio", "xMidYMid meet");
    renderer.projection = d3.geoNaturalEarth1().fitExtent([[18, 28], [width - 18, height - 28]], { type: "Sphere" });
    renderer.path = d3.geoPath(renderer.projection);
    return true;
  }

  function renderBase() {
    const svg = renderer.svg;
    if (!svg || !renderer.world || !renderer.path) return;
    svg.selectAll(".map-base").remove();
    const base = svg.append("g").attr("class", "map-base").attr("pointer-events", "none");
    base.append("path").datum({ type: "Sphere" }).attr("class", "map-ocean").attr("d", renderer.path);
    base.append("path").datum(d3.geoGraticule10()).attr("class", "map-graticule").attr("d", renderer.path);
    base.append("g").selectAll("path")
      .data(topojson.feature(renderer.world, renderer.world.objects.countries).features)
      .join("path").attr("class", "map-country").attr("d", renderer.path);
    base.append("path").datum({ type: "Sphere" }).attr("class", "map-border").attr("d", renderer.path);
  }

  function point(point) {
    return renderer.projection([point.lon, point.lat]);
  }

  function drawLink(selection, source, target) {
    if (!source || !target || Math.abs(source.lon - target.lon) > 180) {
      selection.attr("display", "none");
      return;
    }
    const a = point(source);
    const b = point(target);
    selection.attr("display", null).attr("x1", a[0]).attr("y1", a[1]).attr("x2", b[0]).attr("y2", b[1]);
  }

  function render() {
    const svg = renderer.svg;
    const orbit = renderer.orbit;
    if (!svg || !orbit?.timeslots?.length || !renderer.projection) return;
    svg.selectAll(".orbit-layer").remove();
    const slot = orbit.timeslots[renderer.slot] || orbit.timeslots[0];
    const positions = slot.positions || [];
    const contacts = slot.contacts || [];
    const satelliteById = new Map(positions.map((satellite) => [satellite.sat_id, satellite]));
    const activeIds = new Set(contacts.map((contact) => contact.sat_id));
    const layer = svg.append("g").attr("class", "orbit-layer");

    layer.append("g").attr("class", "trajectory-group").selectAll("path")
      .data(orbit.trajectories || []).join("path")
      .attr("class", (trajectory) => `orbit-trail plane-${(trajectory.sat_id || 0) % 6}`)
      .attr("d", (trajectory) => renderer.path(lineFeature(trajectory.positions)));

    const links = layer.append("g").attr("class", "link-group");
    links.selectAll("line.gsl").data(contacts).join("line").attr("class", "orbit-link gsl")
      .each(function (contact) { drawLink(d3.select(this), satelliteById.get(contact.sat_id), orbit.ground_stations?.[contact.gs_id]); });
    if (orbit.isl_enabled) {
      links.selectAll("line.isl").data(slot.isl_links || []).join("line").attr("class", "orbit-link isl")
        .each(function (link) { drawLink(d3.select(this), satelliteById.get(link.a_id), satelliteById.get(link.b_id)); });
    }
    links.selectAll("line.offload").data(slot.experiment?.offload_actions || []).join("line").attr("class", "orbit-link offload")
      .each(function (action) { drawLink(d3.select(this), satelliteById.get(Number(action.from_sat)), satelliteById.get(Number(action.to_sat))); });

    const stations = layer.append("g").attr("class", "station-group");
    stations.selectAll("g").data(orbit.ground_stations || []).join("g").attr("class", "ground-station")
      .each(function (station, index) {
        const [x, y] = point(station);
        d3.select(this).attr("transform", `translate(${x},${y})`)
          .html(`<path d="M0 -7 L6 5 L-6 5 Z"></path><text y="17">GS${index + 1}</text>`);
      });

    const satellites = layer.append("g").attr("class", "satellite-group");
    satellites.selectAll("g").data(positions).join("g").attr("class", "satellite-node")
      .classed("active", (satellite) => activeIds.has(satellite.sat_id))
      .each(function (satellite) {
        const [x, y] = point(satellite);
        d3.select(this).attr("transform", `translate(${x},${y})`)
          .html(`<rect class="sat-body" x="-4" y="-4" width="8" height="8"></rect><rect class="sat-panel" x="-13" y="-1.5" width="7" height="3"></rect><rect class="sat-panel" x="6" y="-1.5" width="7" height="3"></rect><text y="-9">S${satellite.sat_id + 1}</text>`);
      });
  }

  window.draw2DOrbit = (canvasId, orbit, requestedSlot = 0) => {
    const svg = getSvg(canvasId);
    if (!svg || !orbit) return;
    renderer.orbit = orbit;
    renderer.slot = Math.max(0, Math.min(Number(requestedSlot || 0), orbit.timeslots.length - 1));
    resize();
    if (renderer.world) {
      renderBase();
      render();
      return;
    }
    loadWorld().then((world) => {
      if (!world || renderer.orbit !== orbit) return;
      resize();
      renderBase();
      render();
    });
  };
})();
