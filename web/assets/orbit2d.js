// Global Plate Carree (equirectangular) orbit view for the 2D mode.

function draw2DOrbit(canvasId, orbitData) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const rect = canvas.getBoundingClientRect();
  const cssWidth = Math.max(1, Math.round(rect.width || canvas.clientWidth || 960));
  const cssHeight = Math.max(1, Math.round(rect.height || canvas.clientHeight || 540));
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(cssWidth * ratio);
  canvas.height = Math.round(cssHeight * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const w = cssWidth;
  const h = cssHeight;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#08131b";
  ctx.fillRect(0, 0, w, h);

  const mapPoint = (lat, lon) => ({
    x: ((Number(lon) + 180) / 360) * w,
    y: ((90 - Number(lat)) / 180) * h,
  });
  const wrapped = (a, b) => Math.abs(Number(a) - Number(b)) > 180;

  // Ocean, latitude/longitude grid and a restrained land silhouette keep the
  // projection readable without relying on a remote tile provider.
  ctx.fillStyle = "#0d3447";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(155, 205, 220, 0.18)";
  ctx.lineWidth = 1;
  for (let lon = -180; lon <= 180; lon += 30) {
    const x = mapPoint(0, lon).x;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  for (let lat = -90; lat <= 90; lat += 15) {
    const y = mapPoint(lat, 0).y;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  ctx.strokeStyle = "rgba(220, 240, 242, 0.45)";
  ctx.lineWidth = 1.2;
  [
    [[-168, 72], [-150, 60], [-132, 52], [-124, 38], [-112, 28], [-96, 20], [-84, 27], [-76, 43], [-62, 50], [-52, 62], [-42, 70], [-70, 76], [-110, 78], [-145, 76]],
    [[-82, 12], [-74, 4], [-68, -12], [-63, -28], [-58, -48], [-52, -55], [-44, -32], [-38, -8], [-48, 8], [-63, 17]],
    [[-10, 36], [8, 44], [25, 60], [48, 68], [78, 66], [105, 58], [130, 48], [145, 33], [140, 12], [118, 3], [105, -7], [80, 8], [55, 18], [35, 30], [15, 28]],
    [[-18, 30], [5, 35], [33, 28], [48, 12], [40, -8], [27, -30], [12, -35], [-4, -20], [-15, 2]],
    [[112, -10], [132, -12], [153, -23], [166, -39], [145, -45], [123, -34]],
  ].forEach((polygon) => {
    ctx.beginPath();
    polygon.forEach(([lon, lat], index) => {
      const p = mapPoint(lat, lon);
      if (index === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
    });
    ctx.closePath();
    ctx.fillStyle = "#1c4a4d";
    ctx.fill();
    ctx.stroke();
  });

  if (!orbitData?.timeslots?.length) return;
  const requestedSlot = typeof state !== "undefined" ? Number(state.orbitSlot) : 0;
  const slotIndex = Math.max(0, Math.min(Number.isFinite(requestedSlot) ? requestedSlot : 0, orbitData.timeslots.length - 1));
  const currentSlot = orbitData.timeslots[slotIndex] || orbitData.timeslots[0];
  const positions = currentSlot.positions || [];
  const contacts = currentSlot.contacts || [];
  const islLinks = currentSlot.isl_links || [];
  const offloadActions = currentSlot.experiment?.offload_actions || [];

  // Draw complete sampled trajectories, breaking only at the date line.
  (orbitData.trajectories || []).forEach((trajectory, index) => {
    const samples = trajectory.positions || [];
    ctx.strokeStyle = `hsla(${(index * 47) % 360}, 76%, 68%, 0.42)`;
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    let previous = null;
    samples.forEach((sample) => {
      if (sample == null) return;
      const p = mapPoint(sample.lat, sample.lon);
      if (!previous || wrapped(sample.lon, previous.lon)) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
      previous = sample;
    });
    ctx.stroke();
  });

  const drawConnection = (first, second, color, width, dash = []) => {
    if (!first || !second || wrapped(first.lon, second.lon)) return;
    const a = mapPoint(first.lat, first.lon);
    const b = mapPoint(second.lat, second.lon);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    ctx.setLineDash([]);
  };
  const stationByIndex = orbitData.ground_stations || [];
  contacts.forEach((contact) => {
    const sat = positions.find((item) => item.sat_id === contact.sat_id);
    drawConnection(sat, stationByIndex[contact.gs_id], "rgba(255, 209, 102, 0.58)", 1.2);
  });
  if (orbitData.isl_enabled) islLinks.forEach((link) => {
    const a = positions.find((item) => item.sat_id === link.a_id);
    const b = positions.find((item) => item.sat_id === link.b_id);
    drawConnection(a, b, "rgba(100, 210, 255, 0.64)", 1.4);
  });
  offloadActions.forEach((action) => {
    const a = positions.find((item) => item.sat_id === Number(action.from_sat));
    const b = positions.find((item) => item.sat_id === Number(action.to_sat));
    drawConnection(a, b, "rgba(255, 209, 102, 0.95)", 2.5, [7, 4]);
  });

  stationByIndex.forEach((station, index) => {
    const p = mapPoint(station.lat, station.lon);
    ctx.fillStyle = "#ff8b70";
    ctx.fillRect(p.x - 3, p.y - 3, 6, 6);
    ctx.fillStyle = "#ffd4c8";
    ctx.font = "10px monospace";
    ctx.fillText(`GS${index}`, p.x + 6, p.y - 6);
  });
  positions.forEach((satellite) => {
    const p = mapPoint(satellite.lat, satellite.lon);
    const active = contacts.some((contact) => contact.sat_id === satellite.sat_id);
    ctx.fillStyle = active ? "#67f59a" : "#73c9ff";
    ctx.beginPath(); ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = active ? "#c8ffda" : "#d6f2ff";
    ctx.lineWidth = 1.5; ctx.stroke();
    ctx.fillStyle = "#e9f3f5";
    ctx.font = "10px monospace";
    ctx.fillText(`S${satellite.sat_id}`, p.x + 7, p.y + 3);
  });

  ctx.fillStyle = "rgba(240, 250, 252, 0.9)";
  ctx.font = "12px monospace";
  ctx.fillText(`时隙: ${slotIndex + 1}/${orbitData.timeslots.length}`, 12, 20);
  ctx.fillText(`时间: ${currentSlot.time?.substring(11, 19) || "--"}`, 12, 39);
  ctx.fillText(`卫星: ${positions.length}  连接: ${contacts.length + islLinks.length}`, 12, 58);
}

if (typeof window !== "undefined") {
  window.draw2DOrbit = draw2DOrbit;
}
