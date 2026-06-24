(() => {
  "use strict";

  const POLL_INTERVAL = 5000;
  const ARC_LEN = Math.PI * 62; // semicircle radius 62
  let pollTimer = null;
  let clockTimer = null;
  let currentFilter = "all";
  let currentScope = "today";
  let pendingAction = null;

  const params = new URLSearchParams(window.location.search);
  const urlToken = params.get("token");
  if (urlToken) {
    localStorage.setItem("dashboard_token", urlToken);
    params.delete("token");
    const clean = window.location.pathname + (params.toString() ? `?${params}` : "") + window.location.hash;
    window.history.replaceState({}, "", clean);
  }
  const authToken = localStorage.getItem("dashboard_token") || "";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  // --- Fetch helpers ---

  async function api(path) {
    try {
      const headers = authToken ? { "X-Trading-Agent-Web-Token": authToken } : {};
      const res = await fetch(path, { cache: "no-store", headers });
      return await res.json();
    } catch (e) {
      return { ok: false, error: e.message };
    }
  }

  async function apiPost(path, body) {
    try {
      const res = await fetch(path, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(authToken ? { "X-Trading-Agent-Web-Token": authToken } : {}),
        },
        body: JSON.stringify(body),
      });
      return await res.json();
    } catch (e) {
      return { ok: false, error: e.message };
    }
  }

  // --- Formatting ---

  function num(v) {
    const n = typeof v === "number" ? v : parseFloat(v);
    return isNaN(n) ? null : n;
  }
  function fmtNum(v, d = 2) {
    const n = num(v);
    return n == null ? "--" : n.toFixed(d);
  }
  function fmtMoney(v, d = 0) {
    const n = num(v);
    if (n == null) return "--";
    const s = n > 0 ? "+" : n < 0 ? "−" : "";
    return `${s}$${Math.abs(n).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d })}`;
  }
  function fmtTime(iso) {
    if (!iso) return "--";
    try { return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }); }
    catch { return "--"; }
  }
  function fmtAge(s) {
    if (s == null) return "--";
    if (s < 60) return `${s}s ago`;
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m ago`;
  }
  function pnlClass(v) {
    const n = num(v);
    return n == null ? "" : n >= 0 ? "green" : "red";
  }
  function statusFor(frac) {
    if (frac == null) return "teal";
    if (frac < 0.7) return "green";
    if (frac < 0.9) return "amber";
    return "red";
  }
  function escapeHtml(s) {
    const el = document.createElement("span");
    el.textContent = s;
    return el.innerHTML;
  }
  function card(label, value, cls = "") {
    return `<div class="card"><div class="label">${label}</div><div class="value ${cls}">${value}</div></div>`;
  }

  // --- SVG radial gauge (semicircle) ---

  function gauge({ label, zh, value, frac, cap, capText }) {
    const hasCap = cap != null && cap > 0;
    const st = hasCap ? statusFor(frac) : "teal";
    const offset = hasCap ? ARC_LEN * (1 - Math.max(0, Math.min(1, frac))) : ARC_LEN;
    const arc = hasCap
      ? `<path class="arc-fill ${st}" d="M13 88 A62 62 0 0 1 137 88" fill="none" stroke="currentColor"
           stroke-width="9" stroke-linecap="round" stroke-dasharray="${ARC_LEN.toFixed(2)}" stroke-dashoffset="${offset.toFixed(2)}"/>`
      : "";
    const pct = hasCap ? `${Math.round(frac * 100)}% of cap` : "no cap set";
    return `<div class="gauge ${hasCap ? "" : "nocap"}">
      <svg viewBox="0 0 150 100" aria-hidden="true">
        <path class="track" d="M13 88 A62 62 0 0 1 137 88" fill="none" stroke-width="9" stroke-linecap="round"/>
        ${arc}
      </svg>
      <div class="g-readout">
        <div class="g-value ${st}">${value}</div>
        <div class="g-pct">${pct}</div>
      </div>
      <div class="g-label">${label}</div>
      <div class="g-zh">${zh}</div>
      <div class="g-cap">${capText || ""}</div>
    </div>`;
  }

  function renderCluster(risk, report, positions, mandate) {
    const r = risk.data || {};
    const totals = r.totals || {};
    const rep = report.data || {};
    const orders = rep.orders || {};
    const port = ((mandate.data || {}).portfolio) || {};
    const posCount = (positions.data || {}).count;

    const premium = num(r.premium_at_risk_usd) || 0;
    const openPos = posCount != null ? posCount : (r.open_positions || 0);
    const filled = orders.filled || 0;
    const delta = num(totals.delta) || 0;
    const theta = num(totals.theta_usd_per_day) || 0;
    const pnl = num(rep.realized_pnl_usd);
    const loss = pnl != null && pnl < 0 ? -pnl : 0;

    const premCap = num(port.max_total_premium_at_risk_usd);
    const posCap = num(port.max_open_positions);
    const newCap = num(port.max_new_positions_per_day);
    const deltaCap = num(port.max_portfolio_abs_delta);
    const thetaCap = num(port.max_portfolio_theta_decay_usd_per_day);
    const lossCap = num(port.daily_loss_stop_usd);

    let html = "";
    html += gauge({
      label: "Premium at Risk", zh: "在险权利金",
      value: `$${premium.toLocaleString()}`,
      frac: premCap ? premium / premCap : null, cap: premCap,
      capText: premCap ? `/ $${premCap.toLocaleString()}` : "",
    });
    html += gauge({
      label: "Positions", zh: "持仓占用",
      value: openPos, frac: posCap ? openPos / posCap : null, cap: posCap,
      capText: posCap ? `/ ${posCap}` : "",
    });
    html += gauge({
      label: "New Today", zh: "当日新仓",
      value: filled, frac: newCap ? filled / newCap : null, cap: newCap,
      capText: newCap ? `/ ${newCap}` : "",
    });
    html += gauge({
      label: "Net Delta", zh: "组合净 Delta",
      value: delta.toFixed(2), frac: deltaCap ? Math.abs(delta) / deltaCap : null, cap: deltaCap,
      capText: deltaCap ? `/ ±${deltaCap}` : "",
    });
    html += gauge({
      label: "Theta / Day", zh: "每日时间损耗",
      value: `$${theta.toFixed(0)}`, frac: thetaCap ? Math.abs(theta) / thetaCap : null, cap: thetaCap,
      capText: thetaCap ? `/ $${thetaCap}` : "",
    });
    html += gauge({
      label: "Daily Loss", zh: "当日亏损 vs 止损",
      value: pnl != null ? fmtMoney(pnl, 0) : "$0",
      frac: lossCap ? loss / lossCap : null, cap: lossCap,
      capText: lossCap ? `stop $${lossCap.toLocaleString()}` : "",
    });

    $("#cluster").innerHTML = html;
  }

  // --- Top bar ---

  function renderTopBar(status, positions, report) {
    const data = status.data || {};
    const posData = positions.data || {};
    const rep = report.data || {};

    const modeBadge = $("#mode-badge");
    const mode = localStorage.getItem("trading_mode") || "paper";
    modeBadge.textContent = mode.toUpperCase();
    modeBadge.className = `badge badge-${mode}`;

    const haltBadge = $("#halt-badge");
    if (data.halted) {
      haltBadge.textContent = "HALT ACTIVE";
      haltBadge.className = "badge badge-halt";
      haltBadge.title = data.halt_reason || "";
    } else {
      haltBadge.textContent = "HALT CLEAR";
      haltBadge.className = "badge badge-clear";
      haltBadge.title = "";
    }

    const stale = !!data.heartbeat_stale;
    $("#hb-dot").className = stale ? "dot stale" : "dot";
    const hbInfo = $("#heartbeat-info");
    hbInfo.textContent = `heartbeat ${fmtAge(data.heartbeat_age_seconds)}`;
    hbInfo.style.color = stale ? "var(--red)" : "";

    $("#cycle-info").innerHTML = `last cycle: <b>${data.heartbeat_note ? escapeHtml(String(data.heartbeat_note)) : "--"}</b>`;
    $("#position-count").innerHTML = `positions: <b>${posData.count || 0}</b>`;

    const pnl = num(rep.realized_pnl_usd);
    const pc = pnlClass(pnl);
    const color = pc === "green" ? "var(--green)" : pc === "red" ? "var(--red)" : "var(--text)";
    $("#daily-pnl").innerHTML = `daily P/L: <b style="color:${color}">${pnl != null ? fmtMoney(pnl, 2) : "--"}</b>`;

    $("#btn-halt").disabled = data.halted;
    $("#btn-resume").disabled = !data.halted;
    $("#btn-run-cycle").disabled = false;
  }

  // --- Greeks & stress ---

  function renderRisk(riskData) {
    const d = riskData.data || {};
    const totals = d.totals || {};
    let html = "";
    html += card("Delta", fmtNum(totals.delta, 4));
    html += card("Gamma", fmtNum(totals.gamma, 4));
    html += card("Vega", fmtNum(totals.vega, 4));
    html += card("Theta / Day", `$${fmtNum(totals.theta_usd_per_day, 2)}`, "amber");

    const scenarios = d.scenario_revaluation_pnl_usd || {};
    for (const [label, pnl] of Object.entries(scenarios)) {
      const name = label.replace("underlying_", "").replace("pct", "%").replace(/_/g, " ");
      html += card(`Scenario ${name}`, `$${fmtNum(pnl, 0)}`, pnlClass(pnl));
    }
    $("#risk-cards").innerHTML = html || '<div class="empty">No risk data · 暂无风险数据</div>';
  }

  // --- Decision pipeline ---

  function stage(label, zh, value, opts = {}) {
    const cls = opts.terminal ? "stage terminal" : "stage";
    let drops = "";
    if (opts.drop) drops += `<div class="s-drop">${opts.drop}</div>`;
    if (opts.note) drops += `<div class="s-drop muted">${opts.note}</div>`;
    return `<div class="${cls}"><div class="s-label">${label}</div><div class="s-zh">${zh}</div><div class="s-value">${value}</div>${drops}</div>`;
  }

  function renderFunnel(funnelData) {
    const d = funnelData.data || {};
    const dec = d.committee_decisions || {};
    const liq = d.liquidity_at_entry || {};
    const gate = d.risk_gate || {};
    const hasData = d.universe_selected || d.event_seeds || d.reached_committee || d.orders_filled;

    let html = '<div class="pipeline">';
    html += stage("Universe", "扫描标的", d.universe_selected || 0, { note: `+${d.event_seeds || 0} seeds` });
    html += stage("Contracts", "候选合约", d.reached_committee || 0, {
      drop: d.dropped_no_eligible_contract ? `${d.dropped_no_eligible_contract} no contract` : "",
      note: d.served_from_cache ? `${d.served_from_cache} cached` : "",
    });
    html += stage("Committee", "委员会通过", dec.open_position || 0, {
      drop: dec.reject ? `${dec.reject} reject` : "", note: dec.hold ? `${dec.hold} hold` : "",
    });
    html += stage("Monte Carlo", "蒙特卡洛", d.monte_carlo_checked || 0, {
      drop: d.monte_carlo_rejected ? `${d.monte_carlo_rejected} rejected` : "",
    });
    html += stage("Liquidity", "流动性", liq.passed || 0, { drop: liq.failed ? `${liq.failed} failed` : "" });
    html += stage("Risk Gate", "风控闸门", gate.approved || 0, { drop: gate.rejected ? `${gate.rejected} rejected` : "" });
    html += stage("Filled", "成交", d.orders_filled || 0, { terminal: true });
    html += "</div>";

    $("#funnel-cards").innerHTML = hasData ? html : '<div class="empty">No funnel activity today · 今日暂无漏斗数据</div>';
  }

  // --- Positions ---

  function daysTo(dateStr) {
    if (!dateStr) return null;
    const t = Date.parse(dateStr);
    if (isNaN(t)) return null;
    return Math.round((t - Date.now()) / 86400000);
  }

  function renderPositions(posData) {
    const positions = (posData.data || {}).positions || [];
    const container = $("#positions-table");
    if (positions.length === 0) {
      container.innerHTML = '<div class="empty">No open positions · 当前无持仓</div>';
      return;
    }

    let html = '<div class="scroll-table"><table><thead><tr>';
    html += "<th>Ticker</th><th>Option Code</th><th>Side</th><th>Qty</th><th>Entry</th><th>Spot</th>";
    html += "<th>IV</th><th>Delta</th><th>Theta/Day</th><th>TP%</th><th>SL%</th><th>Time Stop</th><th>Expiry</th>";
    html += "</tr></thead><tbody>";

    for (const p of positions) {
      const dte = daysTo(p.expiry);
      const flag = dte != null && dte <= 5;
      html += `<tr class="${flag ? "flag" : ""}">`;
      html += `<td class="mono-accent">${escapeHtml(p.ticker || "")}</td>`;
      html += `<td>${escapeHtml(p.option_code || "")}</td>`;
      html += `<td>${escapeHtml(p.option_side || "")}</td>`;
      html += `<td>${p.contracts || ""}</td>`;
      html += `<td>${fmtNum(p.entry_price)}</td>`;
      html += `<td>${fmtNum(p.entry_spot)}</td>`;
      html += `<td>${p.entry_iv != null ? (p.entry_iv * 100).toFixed(1) + "%" : "--"}</td>`;
      html += `<td>${fmtNum(p.entry_delta, 4)}</td>`;
      html += `<td class="text-amber">${fmtNum(p.entry_theta)}</td>`;
      html += `<td>${p.take_profit_pct != null ? p.take_profit_pct + "%" : "--"}</td>`;
      html += `<td>${p.stop_loss_pct != null ? p.stop_loss_pct + "%" : "--"}</td>`;
      html += `<td>${escapeHtml(p.time_stop || "--")}</td>`;
      html += `<td class="${flag ? "text-amber" : ""}">${escapeHtml(p.expiry || "--")}${flag ? ` · ${dte}d` : ""}</td>`;
      html += "</tr>";
    }
    html += "</tbody></table></div>";
    container.innerHTML = html;
  }

  // --- Report ---

  function renderReport(reportData) {
    const d = reportData.data || {};
    const orders = d.orders || {};
    const llm = d.llm_usage || {};
    let html = "";
    html += card("Date", d.date || "--");
    html += card("Events", d.events || 0);
    html += card("Proposals", d.proposals_processed || 0);
    html += card("Placed", orders.placed || 0);
    html += card("Filled", orders.filled || 0, "green");
    html += card("Cancelled", orders.cancelled || 0, "red");
    html += card("Closed", d.positions_closed || 0);
    html += card("Realized P/L", `$${fmtNum(d.realized_pnl_usd, 2)}`, pnlClass(d.realized_pnl_usd));
    html += card("Failures", d.failures || 0, d.failures > 0 ? "red" : "");
    html += card("LLM Tokens", llm.total_tokens ? llm.total_tokens.toLocaleString() : "0");
    $("#report-cards").innerHTML = html || '<div class="empty">No report today · 今日暂无报告</div>';
  }

  // --- IV sparklines ---

  function sparkline(values) {
    const W = 100, H = 30, pad = 3;
    if (!values.length) return `<svg class="iv-spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"></svg>`;
    if (values.length === 1) {
      const y = H / 2;
      return `<svg class="iv-spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><circle cx="${W - pad}" cy="${y}" r="2" fill="var(--teal)"/></svg>`;
    }
    let min = Math.min(...values), max = Math.max(...values);
    if (min === max) { min -= 1; max += 1; }
    const n = values.length;
    const xs = (i) => pad + (i / (n - 1)) * (W - 2 * pad);
    const ys = (v) => H - pad - ((v - min) / (max - min)) * (H - 2 * pad);
    const pts = values.map((v, i) => `${xs(i).toFixed(1)},${ys(v).toFixed(1)}`);
    const line = `M${pts.join(" L")}`;
    const area = `M${xs(0).toFixed(1)},${H} L${pts.join(" L")} L${xs(n - 1).toFixed(1)},${H} Z`;
    const last = values[n - 1];
    const lx = xs(n - 1).toFixed(1), ly = ys(last).toFixed(1);
    return `<svg class="iv-spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <path d="${area}" fill="rgba(var(--teal-rgb),0.10)"/>
      <path d="${line}" fill="none" stroke="var(--teal)" stroke-width="1.5" vector-effect="non-scaling-stroke" stroke-linejoin="round"/>
      <circle cx="${lx}" cy="${ly}" r="1.8" fill="var(--teal)"/>
    </svg>`;
  }

  function renderIV(ivData) {
    const data = ivData.data || {};
    const container = $("#iv-table");
    const tickers = Object.keys(data).sort();
    if (tickers.length === 0) {
      container.innerHTML = '<div class="empty">No IV30 data · 暂无波动率数据</div>';
      return;
    }
    let html = '<div class="iv-grid">';
    for (const tk of tickers) {
      const entries = data[tk] || [];
      const vals = entries.map((e) => num(e.iv30)).filter((v) => v != null);
      const latest = vals.length ? vals[vals.length - 1] : null;
      html += `<div class="iv-row">
        <span class="iv-tk">${escapeHtml(tk)}</span>
        ${sparkline(vals)}
        <span class="iv-last">${latest != null ? latest + "%" : "--"}<br><span class="pts">${vals.length} pts</span></span>
      </div>`;
    }
    html += "</div>";
    container.innerHTML = html;
  }

  // --- Events ---

  function classifyEvent(t) {
    if (t.includes("fail") || t.includes("error") || t.includes("crash")) return "error";
    if (t.includes("order")) return "order";
    if (t.includes("committee")) return "committee";
    return "";
  }

  function renderEvents(auditData) {
    const events = (auditData.data || {}).events || [];
    const container = $("#events-list");
    const filtered = events.filter((e) => currentFilter === "all" || classifyEvent(e.event_type || "") === currentFilter);
    if (filtered.length === 0) {
      container.innerHTML = '<div class="empty">No events · 暂无事件</div>';
      return;
    }
    let html = "";
    for (const e of filtered.slice().reverse().slice(0, 100)) {
      const t = e.event_type || "";
      const cls = classifyEvent(t);
      const detail = e.payload ? JSON.stringify(e.payload).substring(0, 140) : "";
      html += `<div class="event-item ${cls ? "is-" + cls : ""}">`;
      html += `<span class="event-time">${fmtTime(e.recorded_at)}</span>`;
      html += `<span class="event-type ${cls}">${escapeHtml(t)}</span>`;
      html += `<span class="event-detail" title="${escapeHtml(JSON.stringify(e.payload || {}))}">${escapeHtml(detail)}</span>`;
      html += `</div>`;
    }
    container.innerHTML = html;
  }

  // --- Polling ---

  async function refresh() {
    const auditPath = currentScope === "24h" ? "/api/audit/24h?limit=200" : "/api/audit/recent?limit=80";
    const [status, positions, risk, funnel, report, audit, iv, mandate] = await Promise.all([
      api("/api/status"),
      api("/api/positions"),
      api("/api/portfolio-risk"),
      api("/api/funnel/today"),
      api("/api/report/today"),
      api(auditPath),
      api("/api/iv"),
      api("/api/config/mandate"),
    ]);

    renderTopBar(status, positions, report);
    renderCluster(risk, report, positions, mandate);
    renderRisk(risk);
    renderFunnel(funnel);
    renderPositions(positions);
    renderReport(report);
    renderIV(iv);
    renderEvents(audit);

    $("#last-update").textContent = `updated ${new Date().toLocaleTimeString()}`;
  }

  function startPolling() {
    refresh();
    pollTimer = setInterval(refresh, POLL_INTERVAL);
  }

  function startClock() {
    const tick = () => { $("#clock").textContent = new Date().toLocaleTimeString([], { hour12: false }); };
    tick();
    clockTimer = setInterval(tick, 1000);
  }

  // --- Controls / UI ---

  function setupFilters() {
    $$(".filter-btn[data-filter]").forEach((btn) => {
      btn.addEventListener("click", () => {
        $$(".filter-btn[data-filter]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        currentFilter = btn.dataset.filter;
        refresh();
      });
    });
    $$(".filter-btn[data-scope]").forEach((btn) => {
      btn.addEventListener("click", () => {
        $$(".filter-btn[data-scope]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        currentScope = btn.dataset.scope;
        refresh();
      });
    });
  }

  function setupTheme() {
    const saved = localStorage.getItem("dashboard_theme") || "dark";
    document.documentElement.setAttribute("data-theme", saved);
    $("#theme-toggle").addEventListener("click", () => {
      const cur = document.documentElement.getAttribute("data-theme");
      const next = cur === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("dashboard_theme", next);
    });
  }

  function setupRefresh() {
    $("#refresh-btn").addEventListener("click", () => {
      clearInterval(pollTimer);
      refresh();
      startPolling();
    });
  }

  function setupModal() {
    const modal = $("#confirm-modal");
    const text = $("#confirm-text");

    function openModal(msg, action) {
      text.textContent = msg;
      pendingAction = action;
      modal.classList.remove("hidden");
    }
    function closeModal() {
      modal.classList.add("hidden");
      pendingAction = null;
    }

    $("#confirm-yes").addEventListener("click", async () => {
      if (pendingAction) await pendingAction();
      closeModal();
    });
    $("#confirm-no").addEventListener("click", closeModal);
    modal.addEventListener("click", (e) => { if (e.target === modal) closeModal(); });

    $("#btn-halt").addEventListener("click", () => {
      openModal("确认 HALT 全部交易？这会停止 run-loop。", async () => {
        const res = await apiPost("/api/halt", { confirm: true, reason: "web dashboard halt" });
        $("#control-status").textContent = res.ok ? "✓ Halt activated · 已停机" : `✗ Error: ${res.error}`;
        setTimeout(refresh, 500);
      });
    });
    $("#btn-resume").addEventListener("click", () => {
      openModal("确认 RESUME 恢复交易？", async () => {
        const res = await apiPost("/api/resume", { confirm: true });
        $("#control-status").textContent = res.ok ? "✓ Resumed · 已恢复" : `✗ Error: ${res.error}`;
        setTimeout(refresh, 500);
      });
    });
    $("#btn-run-cycle").addEventListener("click", () => {
      openModal("运行一个纸盘周期？需要 OpenD，可能耗时较长。", async () => {
        const mandate = await api("/api/config/mandate");
        const tickers = ((mandate.data && mandate.data.universe && mandate.data.universe.watchlist) || []).slice(0, 5);
        if (tickers.length === 0) {
          $("#control-status").textContent = "✗ No tickers in mandate watchlist";
          return;
        }
        const res = await apiPost("/api/run-cycle", { confirm: true, tickers });
        $("#control-status").textContent = res.ok ? "▶ Cycle started in background · 周期已后台启动" : `✗ Error: ${res.error}`;
      });
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    setupTheme();
    setupFilters();
    setupRefresh();
    setupModal();
    startClock();
    startPolling();
  });
})();
