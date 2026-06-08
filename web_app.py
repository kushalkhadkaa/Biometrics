from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from flask import Flask, jsonify, render_template_string, request, send_from_directory
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename

from fingerprint_pipeline import (
    FingerprintConfig,
    as_uint8,
    discover_images,
    mask_overlay,
    provenance_overlay,
    run_pipeline,
)

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parent
WEB_OUTPUTS = ROOT / "outputs" / "web"
UPLOADS = WEB_OUTPUTS / "uploads"
HISTORY_FILE = WEB_OUTPUTS / "history.json"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fingerprint Cleanroom</title>
  <style>
    :root {
      --bg: linear-gradient(180deg, #eef4f1 0%, #f6f8f7 58%, #edf1ee 100%);
      --surface: rgba(255,255,255,0.96);
      --surface-2: #f7fbf8;
      --ink: #132128;
      --muted: #61717a;
      --line: #d5e0dd;
      --teal: #0e6775;
      --teal-2: #0a5160;
      --sage: #d9ebe3;
      --blue: #2a7bd7;
      --red: #b4343d;
      --yellow: #d8af26;
      --shadow: 0 18px 40px rgba(19, 33, 40, 0.08);
      --radius: 10px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font: 15px/1.45 "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .shell {
      max-width: 1500px;
      margin: 0 auto;
      padding: 22px 24px 40px;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 18px;
      margin-bottom: 18px;
    }
    .brand h1 {
      margin: 0;
      font-size: 24px;
      font-weight: 760;
      letter-spacing: 0;
    }
    .brand p {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .chip {
      border: 1px solid #bfd9d0;
      background: rgba(226, 243, 236, 0.94);
      color: var(--teal-2);
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 730;
      white-space: nowrap;
    }
    .grid {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .panel, .canvas, .metrics, .details {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }
    .panel {
      padding: 18px;
      position: sticky;
      top: 18px;
    }
    .section-title {
      margin: 0 0 14px;
      font-size: 15px;
      font-weight: 740;
    }
    .dropzone {
      padding: 18px;
      border: 1px dashed #98b7ac;
      border-radius: 9px;
      background:
        radial-gradient(circle at top left, rgba(183, 220, 206, 0.24), transparent 42%),
        linear-gradient(180deg, #fbfdfc 0%, #f5faf7 100%);
    }
    .dropzone strong {
      display: block;
      font-size: 15px;
      margin-bottom: 5px;
    }
    .dropzone span {
      color: var(--muted);
      font-size: 13px;
    }
    input[type="file"] {
      width: 100%;
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 10px;
      font: inherit;
    }
    button {
      border: 0;
      border-radius: 8px;
      min-height: 44px;
      padding: 10px 14px;
      font: inherit;
      font-weight: 760;
      cursor: pointer;
      transition: transform 140ms ease, background 140ms ease, opacity 140ms ease;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.55; cursor: not-allowed; transform: none; }
    .primary {
      width: 100%;
      margin-top: 14px;
      background: linear-gradient(135deg, var(--teal) 0%, #14869a 100%);
      color: white;
    }
    .primary:hover { background: linear-gradient(135deg, var(--teal-2) 0%, #126d7e 100%); }
    .sample-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 12px;
    }
    .sample-button {
      background: var(--surface-2);
      border: 1px solid var(--line);
      color: var(--ink);
      min-height: 38px;
      font-size: 13px;
    }
    .sample-button:hover { background: #eef6f1; }
    .helper {
      color: var(--muted);
      font-size: 13px;
      margin: 10px 0 0;
    }
    .mini-link {
      color: var(--teal-2);
      font-weight: 700;
      text-decoration: none;
    }
    .message {
      margin-top: 12px;
      font-size: 13px;
      font-weight: 700;
      color: var(--red);
    }
    .status-board {
      margin-top: 16px;
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }
    .status-board h3 {
      margin: 0 0 8px;
      font-size: 13px;
      font-weight: 740;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .progress-track {
      height: 8px;
      border-radius: 999px;
      background: #e5ece8;
      overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      width: 0%;
      border-radius: 999px;
      background: linear-gradient(90deg, #0e6775 0%, #38a0b2 100%);
      transition: width 240ms ease;
    }
    .status-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin: 10px 0 12px;
      font-size: 13px;
      color: var(--muted);
    }
    .activity {
      display: grid;
      gap: 8px;
      max-height: 220px;
      overflow: auto;
      padding-right: 4px;
    }
    .activity-item {
      display: flex;
      gap: 10px;
      align-items: flex-start;
      padding: 9px 10px;
      border-radius: 8px;
      background: #f8fbf9;
      border: 1px solid #e3ebe7;
      font-size: 13px;
    }
    .dot {
      flex: 0 0 10px;
      width: 10px;
      height: 10px;
      margin-top: 4px;
      border-radius: 999px;
      background: #98a9b1;
    }
    .dot.live {
      background: #1f91a6;
      box-shadow: 0 0 0 6px rgba(31, 145, 166, 0.12);
      animation: pulse 1.3s ease infinite;
    }
    .history-list {
      display: grid;
      gap: 8px;
      max-height: 320px;
      overflow: auto;
      margin-top: 10px;
    }
    .history-card {
      background: #f8fbf9;
      border: 1px solid #e3ebe7;
      border-radius: 8px;
      padding: 10px;
      display: grid;
      gap: 4px;
    }
    .history-card strong {
      font-size: 13px;
    }
    .history-card span {
      color: var(--muted);
      font-size: 12px;
    }
    .history-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 4px;
    }
    @keyframes pulse {
      0% { transform: scale(0.92); }
      50% { transform: scale(1.05); }
      100% { transform: scale(0.92); }
    }
    .empty {
      min-height: 460px;
      display: grid;
      place-items: center;
      text-align: center;
      padding: 34px;
      color: var(--muted);
    }
    .empty h2 {
      margin: 0 0 8px;
      color: var(--ink);
      font-size: 22px;
      font-weight: 740;
    }
    .results {
      display: none;
      gap: 16px;
    }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(300px, 0.65fr);
      gap: 16px;
    }
    .stack {
      display: grid;
      gap: 16px;
    }
    .canvas {
      overflow: hidden;
      min-width: 0;
    }
    .canvas-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
      font-weight: 740;
    }
    .canvas img {
      display: block;
      width: 100%;
      height: auto;
      background: white;
    }
    .download-link {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 32px;
      padding: 6px 10px;
      border-radius: 7px;
      background: #edf5f3;
      color: var(--teal-2);
      border: 1px solid #d2e2dd;
      text-decoration: none;
      font-size: 12px;
      font-weight: 740;
    }
    .metrics {
      padding: 14px;
    }
    .report-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    .report-box {
      background: #f7faf8;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    .report-box h3 {
      margin: 0 0 10px;
      font-size: 14px;
      font-weight: 740;
    }
    .report-box p {
      margin: 0;
      white-space: pre-wrap;
      color: #213038;
      font-size: 14px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      background: #f7faf8;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px;
    }
    .metric b {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      margin-bottom: 6px;
    }
    .metric span {
      font-size: 18px;
      font-weight: 770;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 12px;
    }
    .swatch {
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 2px;
      margin-right: 5px;
      vertical-align: -1px;
    }
    .blue { background: var(--blue); }
    .red { background: var(--red); }
    .yellow { background: var(--yellow); }
    .warnings {
      margin: 12px 0 0;
      padding-left: 18px;
      color: #8a6000;
      font-size: 13px;
    }
    .details {
      padding: 14px;
    }
    .details summary {
      cursor: pointer;
      font-size: 14px;
      font-weight: 740;
    }
    .stage-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .loading-overlay {
      position: fixed;
      inset: 0;
      background: rgba(13, 26, 31, 0.18);
      backdrop-filter: blur(4px);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 18px;
      z-index: 50;
    }
    .loading-card {
      width: min(520px, 100%);
      background: rgba(255,255,255,0.98);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 26px 60px rgba(18, 32, 38, 0.18);
      padding: 22px;
    }
    .loading-head {
      display: flex;
      gap: 14px;
      align-items: center;
      margin-bottom: 12px;
    }
    .spinner {
      width: 42px;
      height: 42px;
      border-radius: 999px;
      border: 4px solid #d9ece6;
      border-top-color: #12859a;
      animation: spin 0.9s linear infinite;
      flex: 0 0 42px;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    .loading-head h3 {
      margin: 0;
      font-size: 18px;
      font-weight: 760;
    }
    .loading-head p {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    @media (max-width: 1080px) {
      .grid, .hero { grid-template-columns: 1fr; }
      .panel { position: static; }
      .report-grid,
      .metric-grid, .stage-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 640px) {
      .shell { padding: 16px; }
      .topbar { flex-direction: column; align-items: flex-start; }
      .report-grid, .metric-grid, .stage-grid, .sample-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div class="brand">
        <h1>Fingerprint Cleanroom</h1>
        <p>Automatic forensic preprocessing, artifact removal, reconstruction, and noise suppression.</p>
      </div>
      <div class="chip">Production auto mode</div>
    </div>

    <div class="grid">
      <section class="panel">
        <h2 class="section-title">Input</h2>
        <div class="dropzone">
          <strong>Upload fingerprint image</strong>
          <span>The app chooses preprocessing, text removal, denoising, and reconstruction automatically.</span>
          <input id="fingerprintInput" name="fingerprint" type="file" accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff,image/*">
        </div>
        <button id="processButton" class="primary" type="button">Process image</button>
        <div class="helper">No manual parameter tuning required.</div>

        <div style="margin-top: 16px;">
          <h2 class="section-title">Quick samples</h2>
          <div class="sample-grid">
            {% for sample in samples[:8] %}
              <button class="sample-button" type="button" data-sample="{{ sample.path }}">{{ sample.name }}</button>
            {% endfor %}
          </div>
        </div>

        <div class="message" id="messageBox"></div>

        <div class="status-board">
          <h3>Live activity</h3>
          <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
          <div class="status-meta">
            <span id="statusText">Waiting for input</span>
            <span id="statusPercent">0%</span>
          </div>
          <div class="activity" id="activityList">
            <div class="activity-item"><span class="dot"></span><div>Upload an image or choose a sample to begin.</div></div>
          </div>
        </div>

        <div class="status-board">
          <h3>Case history</h3>
          <div class="history-list" id="historyList">
            {% if history_cases %}
              {% for item in history_cases %}
                <div class="history-card">
                  <strong>{{ item.source_name }}</strong>
                  <span>{{ item.created_label }}</span>
                  <div class="history-actions">
                    <a class="mini-link" href="{{ item.clean_url }}" target="_blank">Clean image</a>
                    <a class="mini-link" href="{{ item.report_pdf }}" target="_blank">PDF report</a>
                  </div>
                </div>
              {% endfor %}
            {% else %}
              <div class="activity-item"><span class="dot"></span><div>No completed cases yet.</div></div>
            {% endif %}
          </div>
        </div>
      </section>

      <section>
        <div class="empty" id="emptyState">
          <div>
            <h2>Ready for processing</h2>
            <div>The final panel will show the cleaned fingerprint, original reference, provenance, and detailed stages.</div>
          </div>
        </div>

        <div class="results" id="resultsArea">
          <div class="hero">
            <article class="canvas">
              <div class="canvas-header">
                <span>Clean output</span>
                <a class="download-link" id="downloadLink" href="#" download>Download</a>
              </div>
              <img id="cleanImage" alt="Clean fingerprint output">
            </article>
            <div class="stack">
              <article class="canvas">
                <div class="canvas-header"><span>Original</span></div>
                <img id="originalImage" alt="Original fingerprint">
              </article>
              <article class="canvas">
                <div class="canvas-header"><span>Artifact map</span></div>
                <img id="artifactImage" alt="Artifact provenance">
              </article>
            </div>
          </div>

          <div class="metrics">
            <div class="report-grid">
              <div class="report-box">
                <h3>Scientific assessment</h3>
                <p id="scientificSummary"></p>
              </div>
              <div class="report-box">
                <h3>Stakeholder note</h3>
                <p id="stakeholderSummary"></p>
              </div>
            </div>
            <div style="height:12px;"></div>
            <div class="metric-grid" id="metricGrid"></div>
            <div class="legend">
              <span><i class="swatch blue"></i>reconstructed</span>
              <span><i class="swatch red"></i>blocked</span>
              <span><i class="swatch yellow"></i>reviewed artifact</span>
            </div>
            <ul class="warnings" id="warningsList"></ul>
          </div>

          <details class="details">
            <summary>Intermediate stages</summary>
            <div class="stage-grid" id="stageGrid"></div>
          </details>
        </div>
      </section>
    </div>
  </div>

  <div class="loading-overlay" id="loadingOverlay">
    <div class="loading-card">
      <div class="loading-head">
        <div class="spinner"></div>
        <div>
          <h3>Processing fingerprint</h3>
          <p id="overlayMessage">Preparing cleanup pipeline</p>
        </div>
      </div>
      <div class="progress-track"><div class="progress-fill" id="overlayProgress"></div></div>
      <div class="status-meta" style="margin-top:10px;">
        <span id="overlayStatus">Starting</span>
        <span id="overlayPercent">0%</span>
      </div>
      <div class="activity" id="overlayActivity"></div>
    </div>
  </div>

  <script>
    const processButton = document.getElementById("processButton");
    const fingerprintInput = document.getElementById("fingerprintInput");
    const messageBox = document.getElementById("messageBox");
    const emptyState = document.getElementById("emptyState");
    const resultsArea = document.getElementById("resultsArea");
    const cleanImage = document.getElementById("cleanImage");
    const originalImage = document.getElementById("originalImage");
    const artifactImage = document.getElementById("artifactImage");
    const downloadLink = document.getElementById("downloadLink");
    const metricGrid = document.getElementById("metricGrid");
    const warningsList = document.getElementById("warningsList");
    const stageGrid = document.getElementById("stageGrid");
    const historyList = document.getElementById("historyList");
    const scientificSummary = document.getElementById("scientificSummary");
    const stakeholderSummary = document.getElementById("stakeholderSummary");
    const progressFill = document.getElementById("progressFill");
    const statusText = document.getElementById("statusText");
    const statusPercent = document.getElementById("statusPercent");
    const activityList = document.getElementById("activityList");
    const loadingOverlay = document.getElementById("loadingOverlay");
    const overlayProgress = document.getElementById("overlayProgress");
    const overlayStatus = document.getElementById("overlayStatus");
    const overlayPercent = document.getElementById("overlayPercent");
    const overlayMessage = document.getElementById("overlayMessage");
    const overlayActivity = document.getElementById("overlayActivity");

    let activeJobId = null;
    let pollHandle = null;

    function setMessage(text) {
      messageBox.textContent = text || "";
    }

    function escapeHtml(text) {
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function renderActivities(items, target, liveMessage) {
      if (!items.length) {
        target.innerHTML = '<div class="activity-item"><span class="dot"></span><div>Waiting for work.</div></div>';
        return;
      }
      target.innerHTML = items.map((item, index) => {
        const live = index === items.length - 1 ? " live" : "";
        return `<div class="activity-item"><span class="dot${live}"></span><div>${escapeHtml(item.message)}</div></div>`;
      }).join("");
    }

    function updateProgressView(job) {
      const pct = Math.max(0, Math.min(100, job.progress || 0));
      progressFill.style.width = `${pct}%`;
      overlayProgress.style.width = `${pct}%`;
      statusText.textContent = job.message || "Working";
      overlayStatus.textContent = job.message || "Working";
      overlayMessage.textContent = job.message || "Working";
      statusPercent.textContent = `${pct}%`;
      overlayPercent.textContent = `${pct}%`;
      renderActivities(job.activities || [], activityList);
      renderActivities(job.activities || [], overlayActivity);
    }

    function showLoading(show) {
      loadingOverlay.style.display = show ? "flex" : "none";
      processButton.disabled = show;
      fingerprintInput.disabled = show;
      document.querySelectorAll("[data-sample]").forEach((button) => {
        button.disabled = show;
      });
    }

    function renderMetrics(metrics) {
      metricGrid.innerHTML = metrics.map((item) => (
        `<div class="metric"><b>${escapeHtml(item.label)}</b><span>${escapeHtml(item.value)}</span></div>`
      )).join("");
    }

    function renderWarnings(warnings) {
      if (!warnings || !warnings.length) {
        warningsList.innerHTML = "";
        warningsList.style.display = "none";
        return;
      }
      warningsList.style.display = "block";
      warningsList.innerHTML = warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("");
    }

    function renderStages(stages) {
      stageGrid.innerHTML = stages.map((item) => `
        <article class="canvas">
          <div class="canvas-header"><span>${escapeHtml(item.label)}</span></div>
          <img src="${item.url}" alt="${escapeHtml(item.label)}" loading="lazy">
        </article>
      `).join("");
    }

    function renderHistory(items) {
      if (!items || !items.length) {
        historyList.innerHTML = '<div class="activity-item"><span class="dot"></span><div>No completed cases yet.</div></div>';
        return;
      }
      historyList.innerHTML = items.map((item) => `
        <div class="history-card">
          <strong>${escapeHtml(item.source_name)}</strong>
          <span>${escapeHtml(item.created_label)}</span>
          <div class="history-actions">
            <a class="mini-link" href="${item.clean_url}" target="_blank">Clean image</a>
            <a class="mini-link" href="${item.report_pdf}" target="_blank">PDF report</a>
          </div>
        </div>
      `).join("");
    }

    function waitForImages(urls) {
      return Promise.all(urls.map((url) => new Promise((resolve) => {
        const probe = new Image();
        probe.onload = () => resolve(true);
        probe.onerror = () => resolve(false);
        probe.src = url;
      })));
    }

    async function renderResult(result) {
      const currentActivities = Array.from(activityList.querySelectorAll(".activity-item div")).map((node) => ({ message: node.textContent }));
      updateProgressView({
        progress: 100,
        message: "Loading generated images",
        activities: [...currentActivities, { message: "Loading generated images" }]
      });
      await waitForImages([result.clean_url, result.original_url, result.artifact_url]);
      cleanImage.src = result.clean_url;
      originalImage.src = result.original_url;
      artifactImage.src = result.artifact_url;
      downloadLink.href = result.download;
      scientificSummary.textContent = result.scientific_summary || "";
      stakeholderSummary.textContent = result.stakeholder_summary || "";
      renderMetrics(result.metrics);
      renderWarnings(result.warnings);
      renderStages(result.stages);
      renderHistory(result.history || []);
      emptyState.style.display = "none";
      resultsArea.style.display = "grid";
    }

    async function startProcessing(formData) {
      setMessage("");
      showLoading(true);
      const response = await fetch("/api/process", { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok) {
        showLoading(false);
        setMessage(data.error || "Processing could not be started.");
        return;
      }
      activeJobId = data.job_id;
      updateProgressView({ progress: 1, message: "Queued for processing", activities: [{ message: "Queued for processing" }] });
      if (pollHandle) {
        clearInterval(pollHandle);
      }
      pollHandle = setInterval(() => pollJob(activeJobId), 900);
      await pollJob(activeJobId);
    }

    async function pollJob(jobId) {
      const response = await fetch(`/api/job/${jobId}`);
      const job = await response.json();
      updateProgressView(job);

      if (job.status === "complete") {
        if (pollHandle) {
          clearInterval(pollHandle);
          pollHandle = null;
        }
        await renderResult(job.result);
        showLoading(false);
        setMessage("");
      } else if (job.status === "error") {
        if (pollHandle) {
          clearInterval(pollHandle);
          pollHandle = null;
        }
        showLoading(false);
        setMessage(job.error || "Processing failed.");
      }
    }

    processButton.addEventListener("click", async () => {
      if (!fingerprintInput.files.length) {
        setMessage("Choose an image before processing.");
        return;
      }
      const formData = new FormData();
      formData.append("fingerprint", fingerprintInput.files[0]);
      await startProcessing(formData);
    });

    document.querySelectorAll("[data-sample]").forEach((button) => {
      button.addEventListener("click", async () => {
        const formData = new FormData();
        formData.append("sample", button.dataset.sample);
        await startProcessing(formData);
      });
    });
  </script>
</body>
</html>
"""


def allowed_file(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTENSIONS


def image_url(path: Path) -> str:
    rel = path.relative_to(WEB_OUTPUTS).as_posix()
    return f"/outputs/{rel}"


def save_image(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        Image.fromarray(as_uint8(image)).save(path)
    else:
        Image.fromarray((image * 255).clip(0, 255).astype("uint8")).save(path)
    return path


def odd(value: int) -> int:
    return value if value % 2 else value + 1


def update_job(job_id: str, **fields: object) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(fields)


def append_activity(job_id: str, progress: int, message: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["progress"] = progress
        job["message"] = message
        job.setdefault("activities", []).append(
            {"progress": progress, "message": message, "timestamp": time.time()}
        )


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_history(items: list[dict]) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")


def created_label(timestamp_iso: str) -> str:
    try:
        dt = datetime.fromisoformat(timestamp_iso)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return timestamp_iso


def summarize_case(metrics: dict[str, float | int | str], warnings: list[str], source_name: str) -> tuple[str, str]:
    roi = float(metrics["roi_coverage_pct"])
    coherence = float(metrics["mean_ridge_coherence_roi"])
    noise = float(metrics["noise_residual_mad"])
    rebuilt = float(metrics["reconstructed_pct_roi"])
    blocked = float(metrics["blocked_artifact_pct_roi"])
    contrast = float(metrics["contrast_std_roi"])

    quality = "strong" if coherence >= 0.5 and contrast >= 0.18 and blocked < 5.0 else "moderate" if coherence >= 0.3 else "limited"
    noise_state = "low residual noise" if noise < 0.01 else "moderate residual noise" if noise < 0.03 else "elevated residual noise"
    reconstruction_state = (
        "minimal reconstruction influence"
        if rebuilt < 2.0
        else "moderate reconstructed area"
        if rebuilt < 10.0
        else "substantial reconstructed area"
    )

    scientific = (
        f"Case {source_name} produced {roi:.2f}% foreground ROI coverage with {coherence:.4f} mean ridge coherence "
        f"and {contrast:.4f} ROI contrast. The final display shows {noise_state}, {reconstruction_state}, "
        f"and {blocked:.3f}% blocked artifact area after cleanup. "
        f"Overall interpretive quality is {quality}; provenance and intermediate stages should be reviewed "
        f"whenever reconstruction exceeds routine cleanup expectations."
    )
    if warnings:
        scientific += " Review flags: " + " ".join(warnings)

    stakeholder = (
        f"The processed fingerprint for {source_name} has been cleaned for easier review, with visible annotation "
        f"and noise suppression applied automatically. The result is best suited for triage, briefing, and collaborative "
        f"forensic review rather than replacing the original capture. Current confidence reads as {quality}, and the "
        f"artifact/provenance panel remains available to support defensible interpretation."
    )
    return scientific, stakeholder


def create_pdf_report(case_dir: Path, source_name: str, result_payload: dict) -> Path:
    pdf_path = case_dir / "forensic_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        fig.text(0.08, 0.95, "Fingerprint Cleanroom Report", fontsize=20, fontweight="bold", color="#14333b")
        fig.text(0.08, 0.92, f"Source: {source_name}", fontsize=11, color="#35505b")
        fig.text(0.08, 0.89, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=11, color="#35505b")
        fig.text(0.08, 0.83, "Scientific assessment", fontsize=13, fontweight="bold", color="#14333b")
        fig.text(0.08, 0.79, result_payload["scientific_summary"], fontsize=10.5, color="#1f3036", wrap=True)
        fig.text(0.08, 0.61, "Stakeholder note", fontsize=13, fontweight="bold", color="#14333b")
        fig.text(0.08, 0.57, result_payload["stakeholder_summary"], fontsize=10.5, color="#1f3036", wrap=True)
        fig.text(0.08, 0.38, "Metrics", fontsize=13, fontweight="bold", color="#14333b")
        metric_lines = "\n".join(f"{item['label']}: {item['value']}" for item in result_payload["metrics"])
        fig.text(0.08, 0.35, metric_lines, fontsize=10.5, color="#1f3036", family="monospace")
        if result_payload["warnings"]:
            fig.text(0.08, 0.16, "Warnings", fontsize=13, fontweight="bold", color="#14333b")
            warning_lines = "\n".join(f"- {warning}" for warning in result_payload["warnings"])
            fig.text(0.08, 0.13, warning_lines, fontsize=10.5, color="#775100")
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        axes[0].imshow(Image.open(case_dir / "original.png"))
        axes[0].set_title("Original")
        axes[1].imshow(Image.open(case_dir / "clean_output.png"), cmap="gray")
        axes[1].set_title("Clean output")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.imshow(Image.open(case_dir / "artifact_map.png"))
        ax.set_title("Artifact provenance map")
        ax.axis("off")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    return pdf_path


def append_case_history(entry: dict) -> list[dict]:
    history = load_history()
    history.insert(0, entry)
    history = history[:40]
    save_history(history)
    return history


def auto_config(image_path: Path) -> FingerprintConfig:
    image = ImageOps.exif_transpose(Image.open(image_path).convert("L"))
    arr = np.asarray(image).astype(np.float32) / 255.0
    h, w = arr.shape
    min_dim = max(1, min(h, w))
    contrast = float(np.std(arr))
    brightness = float(np.mean(arr))
    lap_var = float(cv2.Laplacian((arr * 255).astype("uint8"), cv2.CV_32F).var())

    illumination_sigma = float(np.clip(min_dim / 24.0, 12.0, 48.0))
    if brightness < 0.42 or brightness > 0.72:
        illumination_sigma = float(np.clip(illumination_sigma * 1.25, 14.0, 56.0))

    clahe_clip = 2.1
    if contrast < 0.16:
        clahe_clip = 3.0
    elif contrast > 0.28:
        clahe_clip = 1.6

    block_size = odd(int(np.clip(min_dim / 40.0, 17, 37)))
    bilateral_diameter = 5 if lap_var > 220 else 7
    post_denoise = 0.22 if lap_var > 420 else 0.4
    inpaint_radius = 6 if min_dim >= 700 else 5 if min_dim >= 420 else 4
    final_cleanup_dilate = 2 if min_dim >= 420 else 1
    final_cleanup_radius = 6 if min_dim >= 700 else 5 if min_dim >= 420 else 4

    return FingerprintConfig(
        output_dir=str(WEB_OUTPUTS),
        illumination_sigma=illumination_sigma,
        clahe_clip_limit=clahe_clip,
        segmentation_block_size=block_size,
        denoise_median_size=3,
        denoise_bilateral_diameter=bilateral_diameter,
        denoise_bilateral_sigma_color=42.0,
        denoise_bilateral_sigma_space=6.0,
        color_artifact_saturation=30,
        color_artifact_value_delta=14,
        dark_component_max_area_ratio=0.04,
        remove_ambiguous_dark_inside_roi=True,
        artifact_mask_dilate=2,
        include_review_artifacts_in_reconstruction=True,
        force_reconstruct_artifacts=True,
        max_inpaint_component_area=400000,
        max_inpaint_component_width=900,
        max_inpaint_component_height=900,
        min_inpaint_support_ratio=0.0,
        inpaint_radius=inpaint_radius,
        gabor_blend=0.34,
        post_denoise_strength=post_denoise,
        final_cleanup_dilate=final_cleanup_dilate,
        final_cleanup_radius=final_cleanup_radius,
        suppress_background=True,
    )


def build_result(image_path: Path, progress: object | None = None) -> dict:
    cfg = auto_config(image_path)
    result = run_pipeline(image_path, cfg, progress_callback=progress)
    case_dir = WEB_OUTPUTS / "cases" / f"{image_path.stem}-{uuid.uuid4().hex[:8]}"
    case_dir.mkdir(parents=True, exist_ok=True)

    original_path = case_dir / "original.png"
    ImageOps.exif_transpose(Image.open(image_path).convert("RGB")).save(original_path)
    clean_path = save_image(case_dir / "clean_output.png", result.stages["07_ridge_enhanced_analysis_view"])
    artifact_path = save_image(case_dir / "artifact_map.png", provenance_overlay(result))

    stage_map = [
        ("Normalized", "01_normalized"),
        ("Illumination", "02_illumination_corrected"),
        ("Local contrast", "03_local_contrast"),
        ("ROI isolated", "04_roi_isolated"),
        ("Denoised", "05_ridge_preserving_denoise"),
        ("Reconstruction", "06_artifact_suppressed_reconstruction"),
        ("Ridge enhanced", "07_ridge_enhanced_analysis_view"),
        ("Binary preview", "08_preview_binary_ridge_map"),
    ]
    stages = []
    for label, key in stage_map:
        stage_path = save_image(case_dir / f"{key}.png", result.stages[key])
        stages.append({"label": label, "url": image_url(stage_path)})

    remove_overlay = save_image(
        case_dir / "artifact_remove_overlay.png",
        mask_overlay(result.stages["05_ridge_preserving_denoise"], result.masks["artifact_remove"]),
    )
    review_overlay = save_image(
        case_dir / "artifact_review_overlay.png",
        mask_overlay(result.stages["05_ridge_preserving_denoise"], result.masks["artifact_review"], (1.0, 0.85, 0.05)),
    )
    stages.extend(
        [
            {"label": "Remove mask", "url": image_url(remove_overlay)},
            {"label": "Review mask", "url": image_url(review_overlay)},
        ]
    )

    metrics = [
        {"label": "ROI", "value": f"{result.metrics['roi_coverage_pct']}%"},
        {"label": "Coherence", "value": str(result.metrics["mean_ridge_coherence_roi"])},
        {"label": "Sharpness", "value": str(result.metrics["blur_laplacian_var"])},
        {"label": "Noise", "value": str(result.metrics["noise_residual_mad"])},
        {"label": "Artifacts", "value": f"{result.metrics['artifact_remove_pct_roi']}%"},
        {"label": "Rebuilt", "value": f"{result.metrics['reconstructed_pct_roi']}%"},
        {"label": "Blocked", "value": f"{result.metrics['blocked_artifact_pct_roi']}%"},
        {"label": "Contrast", "value": str(result.metrics["contrast_std_roi"])},
    ]

    warnings = [
        warning
        for warning in result.warnings
        if "blocked" in warning.lower() or "low orientation" in warning.lower()
    ]

    scientific_summary, stakeholder_summary = summarize_case(result.metrics, warnings, image_path.name)

    payload = {
        "source_name": image_path.name,
        "clean_url": image_url(clean_path),
        "original_url": image_url(original_path),
        "artifact_url": image_url(artifact_path),
        "download": image_url(clean_path),
        "stages": stages,
        "metrics": metrics,
        "warnings": warnings,
        "scientific_summary": scientific_summary,
        "stakeholder_summary": stakeholder_summary,
    }

    pdf_path = create_pdf_report(case_dir, image_path.name, payload)
    payload["report_pdf"] = image_url(pdf_path)

    now_iso = datetime.now().isoformat(timespec="seconds")
    history_entry = {
        "case_id": case_dir.name,
        "source_name": image_path.name,
        "clean_url": payload["clean_url"],
        "report_pdf": payload["report_pdf"],
        "created_at": now_iso,
        "created_label": created_label(now_iso),
    }
    payload["history"] = append_case_history(history_entry)
    return payload


def sample_options() -> list[dict[str, str]]:
    return [{"name": p.name, "path": str(p)} for p in discover_images()]


def resolve_input_image() -> tuple[Path | None, str]:
    upload = request.files.get("fingerprint")
    if upload and upload.filename:
        filename = secure_filename(upload.filename)
        candidate = UPLOADS / f"{uuid.uuid4().hex[:8]}-{filename}"
        if not allowed_file(candidate):
            return None, "Unsupported image type."
        upload.save(candidate)
        return candidate, ""

    sample = request.form.get("sample", "")
    if sample:
        candidate = (ROOT / sample).resolve()
        try:
            candidate.relative_to(ROOT)
        except ValueError:
            return None, "Invalid sample path."
        if candidate.exists() and allowed_file(candidate):
            return candidate, ""
        return None, "Selected sample was not found."

    return None, "Upload an image first."


def process_job(job_id: str, image_path: Path) -> None:
    try:
        update_job(job_id, status="running")
        append_activity(job_id, 2, "Queued for processing")

        def progress(progress_value: int, message: str) -> None:
            append_activity(job_id, progress_value, message)

        with app.app_context():
            result = build_result(image_path, progress)
        update_job(job_id, status="complete", progress=100, message="Processing complete", result=result)
        append_activity(job_id, 100, "Clean output ready")
    except Exception as exc:
        update_job(job_id, status="error", error=str(exc), message="Processing failed")
        append_activity(job_id, 100, f"Processing failed: {exc}")


@app.route("/outputs/<path:filename>")
def web_outputs(filename: str):
    return send_from_directory(WEB_OUTPUTS, filename)


@app.route("/")
def index():
    WEB_OUTPUTS.mkdir(parents=True, exist_ok=True)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    history = load_history()
    return render_template_string(HTML, samples=sample_options(), history_cases=history)


@app.route("/api/process", methods=["POST"])
def api_process():
    WEB_OUTPUTS.mkdir(parents=True, exist_ok=True)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    image_path, error = resolve_input_image()
    if image_path is None:
        return jsonify({"error": error}), 400

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Queued for processing",
            "activities": [],
            "result": None,
            "error": None,
            "created_at": time.time(),
        }

    worker = threading.Thread(target=process_job, args=(job_id, image_path), daemon=True)
    worker.start()
    return jsonify({"job_id": job_id})


@app.route("/api/job/<job_id>")
def api_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(job)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
