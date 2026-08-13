// 2D 轨道视图绘制模块

function draw2DOrbit(canvasId, orbitData) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const width = canvas.width = canvas.offsetWidth * 2;
  const height = canvas.height = canvas.offsetHeight * 2;
  ctx.scale(2, 2);

  const w = width / 2;
  const h = height / 2;

  // 清空画布
  ctx.clearRect(0, 0, w, h);

  // 绘制地球
  ctx.fillStyle = '#1a1f3a';
  ctx.fillRect(0, 0, w, h);

  // 绘制网格
  ctx.strokeStyle = 'rgba(100, 120, 150, 0.2)';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 12; i++) {
    const x = (w / 12) * i;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let i = 0; i <= 6; i++) {
    const y = (h / 6) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // 绘制地球（中心圆）
  const centerX = w / 2;
  const centerY = h / 2;
  const earthRadius = Math.min(w, h) * 0.08;

  ctx.fillStyle = '#4a90e2';
  ctx.beginPath();
  ctx.arc(centerX, centerY, earthRadius, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = '#6ba3e8';
  ctx.lineWidth = 2;
  ctx.stroke();

  if (!orbitData || !orbitData.timeslots || orbitData.timeslots.length === 0) {
    return;
  }

  const currentSlot = orbitData.timeslots[state.orbitSlot] || orbitData.timeslots[0];
  const positions = currentSlot.positions || [];
  const contacts = currentSlot.contacts || [];
  const islLinks = currentSlot.isl_links || [];
  const offloadActions = currentSlot.experiment?.offload_actions || [];

  // 计算缩放比例（轨道高度到像素）
  const orbitRadius = Math.min(w, h) * 0.35;

  // 经纬度转平面坐标
  function latLonToXY(lat, lon) {
    const x = centerX + (lon / 180) * orbitRadius * Math.cos((lat * Math.PI) / 180);
    const y = centerY - (lat / 90) * orbitRadius;
    return { x, y };
  }

  // Complete projected orbit per satellite, independent of replay duration.
  (orbitData.trajectories || []).forEach((trajectory, index) => {
    ctx.strokeStyle = `hsla(${(index * 47) % 360}, 72%, 67%, 0.28)`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    let previous = null;
    for (const sample of trajectory.positions || []) {
      const point = latLonToXY(sample.lat, sample.lon);
      if (!previous || Math.abs(Number(sample.lon) - Number(previous.lon)) > 180) {
        ctx.moveTo(point.x, point.y);
      } else {
        ctx.lineTo(point.x, point.y);
      }
      previous = sample;
    }
    ctx.stroke();
  });

  // 绘制卫星轨道圆
  ctx.strokeStyle = 'rgba(100, 150, 200, 0.3)';
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 5]);
  if (!orbitData.trajectories?.length) {
    ctx.beginPath();
    ctx.arc(centerX, centerY, orbitRadius, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // 绘制地面站
  const stations = orbitData.ground_stations || [];
  stations.forEach((station, idx) => {
    const pos = latLonToXY(station.lat, station.lon);
    ctx.fillStyle = '#ff6b6b';
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#ff6b6b';
    ctx.font = '10px monospace';
    ctx.fillText(`GS${idx}`, pos.x + 6, pos.y - 6);
  });

  // 绘制卫星-地面站连接
  ctx.strokeStyle = 'rgba(255, 200, 100, 0.4)';
  ctx.lineWidth = 1;
  contacts.forEach(contact => {
    const sat = positions.find(p => p.sat_id === contact.sat_id);
    const station = stations[contact.gs_id];
    if (sat && station) {
      const satPos = latLonToXY(sat.lat, sat.lon);
      const gsPos = latLonToXY(station.lat, station.lon);
      ctx.beginPath();
      ctx.moveTo(satPos.x, satPos.y);
      ctx.lineTo(gsPos.x, gsPos.y);
      ctx.stroke();
    }
  });

  // 绘制ISL链路
  if (orbitData.isl_enabled && islLinks.length > 0) {
    ctx.strokeStyle = 'rgba(100, 200, 255, 0.5)';
    ctx.lineWidth = 1.5;
    islLinks.forEach(link => {
      const satA = positions.find(p => p.sat_id === link.a_id);
      const satB = positions.find(p => p.sat_id === link.b_id);
      if (satA && satB) {
        const posA = latLonToXY(satA.lat, satA.lon);
        const posB = latLonToXY(satB.lat, satB.lon);
        ctx.beginPath();
        ctx.moveTo(posA.x, posA.y);
        ctx.lineTo(posB.x, posB.y);
        ctx.stroke();
      }
    });
  }

  // Highlight archived FedLEO sample transfers for the current training round.
  offloadActions.forEach(action => {
    const from = positions.find(p => p.sat_id === Number(action.from_sat));
    const to = positions.find(p => p.sat_id === Number(action.to_sat));
    if (!from || !to) return;
    const fromPos = latLonToXY(from.lat, from.lon);
    const toPos = latLonToXY(to.lat, to.lon);
    ctx.strokeStyle = 'rgba(255, 209, 102, 0.95)';
    ctx.lineWidth = 3;
    ctx.setLineDash([7, 4]);
    ctx.beginPath();
    ctx.moveTo(fromPos.x, fromPos.y);
    ctx.lineTo(toPos.x, toPos.y);
    ctx.stroke();
    ctx.setLineDash([]);
  });

  // 绘制卫星
  positions.forEach(sat => {
    const pos = latLonToXY(sat.lat, sat.lon);
    const hasContact = contacts.some(c => c.sat_id === sat.sat_id);

    // 卫星圆点
    ctx.fillStyle = hasContact ? '#4ade80' : '#60a5fa';
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 5, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = hasContact ? '#22c55e' : '#3b82f6';
    ctx.lineWidth = 2;
    ctx.stroke();

    // 卫星标签
    ctx.fillStyle = '#e5e7eb';
    ctx.font = '9px monospace';
    ctx.fillText(`S${sat.sat_id}`, pos.x + 8, pos.y + 3);
  });

  // 绘制时隙信息
  ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
  ctx.font = '12px monospace';
  ctx.fillText(`时隙: ${state.orbitSlot}/${orbitData.timeslots.length - 1}`, 10, 20);
  ctx.fillText(`时间: ${currentSlot.time?.substring(11, 19) || '--'}`, 10, 40);
  ctx.fillText(`卫星: ${positions.length}`, 10, 60);
  ctx.fillText(`连接: ${contacts.length}`, 10, 80);
  if (orbitData.isl_enabled) {
    ctx.fillText(`ISL: ${islLinks.length}`, 10, 100);
  }
}

// 导出给主应用使用
if (typeof window !== 'undefined') {
  window.draw2DOrbit = draw2DOrbit;
}
