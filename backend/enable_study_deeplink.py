# 1. Add study_id to n8n_orchestrator.py payload
p0 = "./app/services/n8n_orchestrator.py"
with open(p0, "r", encoding="utf-8") as f:
    s0 = f.read()
if '"study_id":' not in s0:
    s0 = s0.replace(
        '"event": "CRITICAL_RADIOLOGY_FINDING",',
        '"event": "CRITICAL_RADIOLOGY_FINDING",\n            "study_id": vision_result.get("study_id", ""),'
    )
    with open(p0, "w", encoding="utf-8") as f:
        f.write(s0)

# 2. Add STUDY_CACHE + GET /api/v1/studies/{study_id} + pass study_id in app/main.py
p1 = "./app/main.py"
with open(p1, "r", encoding="utf-8") as f:
    m = f.read()

if "STUDY_CACHE" not in m:
    m = m.replace(
        "DPO_FEEDBACK_VAULT: List[dict] = []",
        "DPO_FEEDBACK_VAULT: List[dict] = []\nSTUDY_CACHE: dict = {}"
    )

if '"study_id": study_id,' in m and '"study_id": study_id,\n                "primary_finding": primary,' not in m:
    m = m.replace(
        '"primary_finding": primary,\n                "confidence_score": top_conf,',
        '"study_id": study_id,\n                "primary_finding": primary,\n                "confidence_score": top_conf,',
        1
    )

if "study_id: str = \"\"" not in m:
    m = m.replace(
        'patient_info: str = ""',
        'patient_info: str = ""\n    study_id: str = ""'
    )
    m = m.replace(
        'vision_result={"primary_finding": req.primary_finding,',
        'vision_result={"study_id": req.study_id, "primary_finding": req.primary_finding,'
    )

# Store payload in STUDY_CACHE before returning in upload_and_analyze_xray
old_ret = '    return {\n        "study_id": study_id,'
new_ret = '    payload_out = {\n        "study_id": study_id,'
if old_ret in m:
    m = m.replace(old_ret, new_ret, 1)
    m = m.replace(
        '        "n8n_automation": n8n_dispatch,\n    }',
        '        "n8n_automation": n8n_dispatch,\n    }\n    STUDY_CACHE[study_id] = payload_out\n    return payload_out',
        1
    )

# Add GET /api/v1/studies/{study_id} endpoint right after /api/v1/studies/history
if '@app.get("/api/v1/studies/{study_id}")' not in m:
    m = m.replace(
        '@app.get("/api/v1/studies/history")\nasync def get_studies_history(q: str = "", page: int = 1, page_size: int = 10):\n    return local_db.list_studies(q=q, page=page, page_size=page_size)',
        '@app.get("/api/v1/studies/history")\nasync def get_studies_history(q: str = "", page: int = 1, page_size: int = 10):\n    return local_db.list_studies(q=q, page=page, page_size=page_size)\n\n\n@app.get("/api/v1/studies/{study_id}")\nasync def get_study_by_id(study_id: str):\n    if study_id in STUDY_CACHE:\n        return STUDY_CACHE[study_id]\n    raise HTTPException(status_code=404, detail="Study session expired or not found in active memory.")'
    )

with open(p1, "w", encoding="utf-8") as f:
    f.write(m)

# 3. Update index.html: pass study_id on manual-alert & auto-load ?study_id=... on page load
p2 = "./app/static/index.html"
with open(p2, "r", encoding="utf-8") as f:
    h = f.read()

if 'study_id: currentStudy.study_id,\n              patient_info:' not in h:
    h = h.replace(
        'patient_info: `${document.getElementById("inAge").value}',
        'study_id: currentStudy.study_id,\n              patient_info: `${document.getElementById("inAge").value}'
    )

deeplink_js = '''
      // Deep-Link Auto-Loader: opens exact patient study from n8n email (?study_id=...)
      (async () => {
        const sid = new URLSearchParams(window.location.search).get("study_id");
        if (!sid) return;
        try {
          const r = await fetch(`http://127.0.0.1:8000/api/v1/studies/${encodeURIComponent(sid)}`);
          if (!r.ok) { showToast("Study link expired (server restarted)."); return; }
          const data = await r.json();
          currentStudy = data;
          rawAiDraft = data.clinical_impression || "";
          document.getElementById("aiOutputReadOnly").textContent = rawAiDraft;
          document.getElementById("origXrayImg").src = data.original_image_b64;
          document.getElementById("heatmapOverlayImg").src = data.heatmap_image_b64;
          document.getElementById("origXrayImg").classList.remove("hidden");
          document.getElementById("heatmapOverlayImg").classList.remove("hidden");
          const ph = document.getElementById("viewportPlaceholder");
          if (ph) ph.classList.add("hidden");
          const ctrTxt = data.ctr_ratio ? ` | CTR: ${data.ctr_ratio} (${data.ctr_ratio > 0.50 ? "Enlarged" : "Normal"})` : "";
          document.getElementById("scanTelemetryMeta").textContent = `${data.filename} | ${data.latency_ms}ms${ctrTxt}`;
          const urg = document.getElementById("urgencyBadge");
          urg.textContent = `${data.urgency} TRIAGE`;
          urg.className = (data.urgency === "STAT" || data.urgency === "CRITICAL")
            ? "px-2.5 py-1 rounded-full text-[11px] font-mono font-semibold bg-rose-500/20 text-rose-300 border border-rose-500/40"
            : "px-2.5 py-1 rounded-full text-[11px] font-mono font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/40";
          if (data.clinical_risk_score !== undefined) {
            const rb = document.getElementById("riskScoreBanner");
            const rBadge = document.getElementById("riskScoreBadge");
            const rList = document.getElementById("riskRationaleList");
            if (rb && rBadge && rList) {
              rb.classList.remove("hidden");
              rBadge.textContent = `${Number(data.clinical_risk_score).toFixed(1)} / 100 (${data.triage_tier || data.urgency})`;
              rBadge.className = Number(data.clinical_risk_score) >= 65 ? "font-mono font-bold text-rose-400" : "font-mono font-bold text-amber-300";
              rList.innerHTML = (data.risk_rationale || []).map(x => `<div>&bull; ${String(x).replace(/</g,"&lt;")}</div>`).join("");
            }
          }
          if (typeof renderProbs === "function" && data.class_probabilities) renderProbs(data.class_probabilities);
          document.getElementById("saveDpoPairBtn").disabled = false;
          document.getElementById("dispatchN8nTriageBtn").disabled = false;
          document.getElementById("downloadReportBtn").disabled = false;
          showToast(`Loaded Deep-Linked Study: ${data.filename}`);
        } catch (e) {}
      })();
'''
if "Deep-Link Auto-Loader" not in h:
    h = h.replace("</script>", deeplink_js + "\n    </script>", 1)

with open(p2, "w", encoding="utf-8") as f:
    f.write(h)

print("DEEP-LINK SUPPORT (?study_id=...) ENABLED ACROSS n8n, main.py & index.html!")
