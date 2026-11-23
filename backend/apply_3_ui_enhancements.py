p = "./app/static/index.html"
with open(p, "r", encoding="utf-8") as f:
    s = f.read()

# 1. Add Operating Mode selector right before Upload button
old_up = '<input type="file" id="realXrayFileInput" accept="image/*,.dcm" class="hidden" />'
new_up = (
    '<select id="operatingModeSelect" title="Clinical Operating Mode" '
    'class="px-2.5 py-2 rounded-xl text-xs font-mono bg-slate-950 text-cyan-300 border border-slate-700/80 focus:border-cyan-400 focus:outline-none">'
    '<option value="triage">Mode: ER Triage (High Sens)</option>'
    '<option value="balanced">Mode: Balanced (Low Alarm)</option>'
    '</select>\n            ' + old_up
)
if 'id="operatingModeSelect"' not in s:
    assert old_up in s, "upload input not found"
    s = s.replace(old_up, new_up, 1)

# 2. Add Clinical Risk Score & Rationale strip right after n8nStatusPill
old_n8n = '''            <span id="n8nStatusPill" class="px-2.5 py-1 rounded-lg text-[11px] font-mono bg-slate-950 text-slate-400 border border-slate-800">
              n8n Triage: Idle
            </span>'''
new_n8n = old_n8n + '''
          </div>
          <div id="riskScoreBanner" class="hidden mb-3 rounded-xl bg-slate-950/90 border border-slate-700/80 px-3.5 py-2.5 text-xs">
            <div class="flex items-center justify-between gap-2">
              <span class="font-semibold text-slate-200">Multimodal Clinical Risk Engine:</span>
              <span id="riskScoreBadge" class="font-mono font-bold text-cyan-300">-- / 100</span>
            </div>
            <div id="riskRationaleList" class="mt-1 text-[11px] font-mono text-slate-400 leading-relaxed"></div>'''

# Inject cleanly right below the header flex container containing n8nStatusPill
if 'id="riskScoreBanner"' not in s:
    idx_pill = s.index('id="n8nStatusPill"')
    close_span = s.index('</span>', idx_pill) + len('</span>')
    close_div = s.index('</div>', close_span) + len('</div>')
    risk_div = (
        '\n          <div id="riskScoreBanner" class="hidden mt-2 mb-3 rounded-xl bg-slate-950/90 border border-slate-700/80 px-3.5 py-2.5 text-xs">\n'
        '            <div class="flex items-center justify-between gap-2">\n'
        '              <span class="font-semibold text-slate-200">Multimodal Clinical Risk Engine</span>\n'
        '              <span id="riskScoreBadge" class="font-mono font-bold text-cyan-300">-- / 100</span>\n'
        '            </div>\n'
        '            <div id="riskRationaleList" class="mt-1 text-[11px] font-mono text-slate-400 leading-relaxed"></div>\n'
        '          </div>'
    )
    s = s[:close_div] + risk_div + s[close_div:]

# 3. Add 175k RAG Evidence container right below followupAnswerBox
old_fbox = '<div id="followupAnswerBox" class="hidden rounded-xl bg-slate-950 border border-cyan-500/30 p-3 text-xs font-mono text-cyan-100 leading-relaxed"></div>'
if 'id="ragEvidenceGrid"' not in s:
    assert old_fbox in s, "followupAnswerBox not found"
    s = s.replace(
        old_fbox,
        old_fbox + '\n            <div id="ragEvidenceGrid" class="hidden grid grid-cols-1 md:grid-cols-3 gap-2 mt-2"></div>',
        1
    )

# 4. Pass real age, spo2, clinical_context, and operating_mode in FormData
old_fd = 'fd.append("patient_mrn", `SpO2 ${spo2} | ${hist}`);'
new_fd = (
    'fd.append("patient_mrn", `SpO2 ${spo2} | ${hist}`);\n'
    '      fd.append("age", String(age || ""));\n'
    '      fd.append("spo2", String(spo2 || ""));\n'
    '      fd.append("clinical_context", String(hist || ""));\n'
    '      const modeEl = document.getElementById("operatingModeSelect");\n'
    '      fd.append("operating_mode", modeEl ? modeEl.value : "triage");'
)
if 'fd.append("operating_mode"' not in s:
    assert old_fd in s, "fd.append patient_mrn not found"
    s = s.replace(old_fd, new_fd, 1)

# 5. Render clinical_risk_score & risk_rationale on upload response
old_n8n_js = 'if (data.n8n_automation) {'
risk_js = '''if (data.clinical_risk_score !== undefined) {
          const rb = document.getElementById("riskScoreBanner");
          const rBadge = document.getElementById("riskScoreBadge");
          const rList = document.getElementById("riskRationaleList");
          if (rb && rBadge && rList) {
            rb.classList.remove("hidden");
            const sc = Number(data.clinical_risk_score);
            const tier = data.triage_tier || data.urgency || "";
            rBadge.textContent = `${sc.toFixed(1)} / 100 (${tier})`;
            rBadge.className = sc >= 65 ? "font-mono font-bold text-rose-400" : (sc >= 40 ? "font-mono font-bold text-amber-300" : "font-mono font-bold text-emerald-400");
            const items = Array.isArray(data.risk_rationale) ? data.risk_rationale : [];
            rList.innerHTML = items.map(x => `<div>&bull; ${String(x).replace(/</g,"&lt;")}</div>`).join("");
          }
        }
        '''
if 'data.clinical_risk_score !== undefined' not in s:
    assert old_n8n_js in s, "data.n8n_automation block not found"
    s = s.replace(old_n8n_js, risk_js + old_n8n_js, 1)

# 6. Render 175k RAG Evidence Cards in Ask 7B response
old_f_res = 'box.textContent = d.answer || d.detail || "No response";'
new_f_res = '''box.textContent = d.answer || d.detail || "No response";
          const evGrid = document.getElementById("ragEvidenceGrid");
          if (evGrid && Array.isArray(d.evidence) && d.evidence.length > 0) {
            evGrid.classList.remove("hidden");
            evGrid.innerHTML = d.evidence.map(ev =>
              `<div class="rounded-lg bg-slate-950/90 border border-slate-800 p-2.5 text-[11px] text-slate-300 leading-snug">` +
              `<div class="font-mono text-cyan-400 font-semibold mb-1">[${ev.rank}] 175k RAG &bull; Score ${ev.rerank_score}</div>` +
              `<div>${String(ev.snippet || "").slice(0, 180).replace(/</g,"&lt;")}...</div></div>`
            ).join("");
          } else if (evGrid) { evGrid.classList.add("hidden"); }'''

if 'const evGrid = document.getElementById("ragEvidenceGrid")' not in s:
    if old_f_res in s:
        s = s.replace(old_f_res, new_f_res, 1)
    else:
        # Fallback match for box.textContent inside askFollowupBtn
        idx_ask = s.index('id="askFollowupBtn"') if 'id="askFollowupBtn"' in s else s.index('getElementById("askFollowupBtn")')
        idx_bt = s.index('box.textContent =', idx_ask + 120)
        end_ln = s.index('\n', idx_bt)
        s = s[:end_ln] + '\n          ' + new_f_res.split('\n', 1)[1] + s[end_ln:]

with open(p, "w", encoding="utf-8") as f:
    f.write(s)

print("ALL 3 UI ENHANCEMENTS APPLIED CLEANLY TO index.html!")
