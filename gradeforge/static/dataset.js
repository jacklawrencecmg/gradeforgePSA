// Dataset, annotation, model-lifecycle and integrity views. Relies on $, el, api, pretty, statBox from index.html.
(() => {
const root = $('#ds');
const J = (method, path, body) => api(path, {method, headers:{'Content-Type':'application/json'}, body: body === undefined ? undefined : JSON.stringify(body)});
const say = (n, t, bad) => { n.textContent = t; n.style.color = bad ? 'var(--bad)' : 'var(--ok)'; };
const rate = r => !r || r.rate == null ? 'n/a' : `${(r.rate*100).toFixed(0)}% (${r.k}/${r.n}, 95% CI ${r.ci95 ? (r.ci95[0]*100).toFixed(0) + '–' + (r.ci95[1]*100).toFixed(0) + '%' : 'n/a'})`;
const field = (label, input) => el('div', {}, el('label', {}, label), input);
const opts = (obj, blank) => [blank ? el('option', {value:''}, blank) : null, ...(Array.isArray(obj) ? obj.map(v => [v, v]) : Object.entries(obj)).map(([k, v]) => el('option', {value:k}, Array.isArray(obj) ? pretty(k) : `${pretty(k)}`))];
const kv = (rows) => el('table', {}, rows.map(([k, v]) => el('tr', {}, el('td', {style:'color:var(--muted)'}, k), el('td', {}, v ?? 'n/a'))));
const note = t => el('p', {class:'note'}, t);
let vocab = null, view = 'overview', cur = null;

async function boot() {
  vocab = vocab || await api('/api/dataset/vocab');
  root.replaceChildren(el('div', {class:'panel'}, el('b', {}, 'Data-validated pipeline. '),
    'Every number on these pages is computed from cards you record. With no real outcomes recorded, nothing here is validated and no model is used.'),
    el('div', {style:'display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px'},
      ['overview', 'cards', 'annotate', 'models', 'integrity'].map(v => el('button', {class: 'ghost' + (v === view ? ' on' : ''), type:'button', onclick: () => { view = v; boot(); }}, pretty(v)))),
    el('div', {id:'dsv'}));
  const box = $('#dsv');
  try { await ({overview, cards, annotate, models, integrity}[view])(box); } catch (e) { box.append(el('p', {style:'color:var(--bad)'}, e.message)); }
}
document.querySelector('#tabs button[data-tab=dataset]').addEventListener('click', boot);

// ---------------------------------------------------------------- overview
async function overview(box) {
  const s = await api('/api/dataset/summary'), f = s.funnel, r = s.readiness;
  box.append(el('div', {class:'panel'}, el('h3', {style:'margin-top:0'}, 'Collection funnel'),
    el('div', {class:'stats'}, Object.entries(f).map(([k, v]) => statBox(pretty(k), v))),
    note('Only cards that are graded, VERIFIED against the PSA cert page, photographed before submission, and not excluded count as training or evaluation data.')));
  const t = el('table', {}, el('tr', {}, ['Partition', 'Cards', 'Eligible', 'PSA 10', 'Not 10'].map(h => el('th', {}, h))));
  for (const [p, d] of Object.entries(s.partitions)) t.append(el('tr', {}, el('td', {}, p), el('td', {}, d.cards), el('td', {}, d.eligible), el('td', {}, d.psa10), el('td', {}, d.grade_le9)));
  box.append(el('div', {class:'panel'}, el('h3', {style:'margin-top:0'}, 'Partitions (assigned once, by salted hash of the card group; never edited)'), t,
    el('h4', {}, r.ready_to_train ? 'Enough data to train a first model' : 'Not enough data to train yet'),
    el('ul', {}, Object.entries(r.requirements).map(([k, v]) => el('li', {}, `${k}: ${v} — ${r.satisfied[k] ? 'met' : 'not met'}`))),
    Object.keys(s.ineligible_reasons).length ? el('div', {}, el('h4', {}, 'Why cards are not eligible'), el('ul', {}, Object.entries(s.ineligible_reasons).map(([k, v]) => el('li', {}, `${v} × ${k}`)))) : null,
    s.synthetic_in_dataset ? el('p', {style:'color:var(--bad)'}, `${s.synthetic_in_dataset} synthetic test cards are present.`) : null,
    note('Active model: ' + (s.champion || 'none. Predictions stay rules-only until a model passes the untouched test partition.'))));
}

// ---------------------------------------------------------------- cards
async function cards(box) {
  const typeSel = el('select', {}, opts(vocab.card_types)), fin = el('select', {}, opts(vocab.finishes)), basis = el('select', {}, opts(vocab.selection_bases));
  const inputs = {label: el('input', {type:'text'}), set_name: el('input', {type:'text'}), year: el('input', {type:'number'}), subject: el('input', {type:'text'}), card_number: el('input', {type:'text'}), variant: el('input', {type:'text'})};
  const rights = el('input', {type:'checkbox'}), out = el('span');
  box.append(el('div', {class:'panel'}, el('h3', {style:'margin-top:0'}, 'Register a physical card'),
    el('div', {class:'grid g2'}, field('Card type', typeSel), field('Finish', fin), field('Label', inputs.label), field('Set', inputs.set_name), field('Year', inputs.year),
      field('Subject / player / character', inputs.subject), field('Card number', inputs.card_number), field('Variant', inputs.variant),
      field('Why did/will you submit this card? (selection bias is reported from this)', basis)),
    note('Copies of the same set + year + number are kept in the same partition so near-identical cards cannot sit on both sides of the split.'),
    el('label', {style:'display:flex;gap:6px;align-items:center'}, rights, 'I took these photographs (or have permission) and may use them to train and evaluate GradeForge.'),
    el('button', {class:'primary', type:'button', onclick: async () => {
      try { const b = {card_type:typeSel.value, finish:fin.value, selection_basis:basis.value, rights_attested:rights.checked};
        for (const [k, i] of Object.entries(inputs)) b[k] = k === 'year' ? (i.value ? +i.value : null) : i.value;
        const c = await J('POST', '/api/dataset/cards', b); say(out, ` registered ${c.card_uid} (partition: ${c.partition})`); cur = c.card_uid; boot();
      } catch (e) { say(out, ' ' + e.message, true); } }}, 'Register'), out));
  const list = await api('/api/dataset/cards');
  const t = el('table', {}, el('tr', {}, ['Card', 'Type', 'Partition', 'Status', 'Grade', 'Verified', 'Eligible', ''].map(h => el('th', {}, h))));
  list.items.forEach(c => t.append(el('tr', {}, el('td', {}, c.label || c.card_uid), el('td', {}, pretty(c.card_type)), el('td', {}, c.partition), el('td', {}, c.status), el('td', {}, c.grade ?? '—'),
    el('td', {}, pretty(c.verification || '—')), el('td', {}, c.eligible_for_training ? 'yes' : 'no'), el('td', {}, el('button', {class:'ghost', type:'button', onclick: () => { cur = c.card_uid; boot(); }}, 'Open')))));
  box.append(el('div', {class:'panel'}, el('h3', {style:'margin-top:0'}, `Cards (${list.items.length})`), list.items.length ? t : el('p', {class:'muted'}, 'No cards yet.')));
  if (cur) box.append(await cardPanel(cur));
}

async function cardPanel(uid) {
  const c = await api('/api/dataset/cards/' + uid), hist = (await api('/api/history')).items, sub = c.submission || {};
  const p = el('div', {class:'panel'}, el('h3', {style:'margin-top:0'}, `${c.label || c.uid || uid} — ${c.status}`),
    kv([['Card id', uid], ['Partition', c.partition], ['Eligible for training', c.eligible_for_training ? 'yes' : 'no: ' + c.ineligible_reasons.join('; ')],
        ['Photo-quality tier', c.frozen?.quality_tier], ['Images linked', c.images.map(i => `${i.analysis_id} (${pretty(i.phase)}${i.captured_on ? ', ' + i.captured_on : ''})`).join('; ') || 'none']]));
  const out = el('div'), run = async fn => { try { await fn(); boot(); } catch (e) { say(out, e.message, true); } };
  // link image
  const an = el('select', {}, hist.map(h => el('option', {value:h.id}, `${h.label || h.id} — ${new Date(h.created * 1000).toLocaleDateString()}`)));
  const ph = el('select', {}, opts(vocab.image_phases, 'Choose phase…')), cap = el('input', {type:'date'});
  p.append(el('h4', {}, '1. Link analysed photographs'), el('div', {class:'grid g2'}, field('Analysis', an), field('Phase (be honest: slab photos are never training data)', ph), field('Date photographed', cap)),
    el('button', {class:'ghost', type:'button', onclick: () => run(() => J('POST', `/api/dataset/cards/${uid}/images`, {analysis_id:an.value, phase:ph.value, captured_on:cap.value || null}))}, 'Link photographs'));
  // submission
  const sd = el('input', {type:'date', value:sub.submitted_on || ''}), sv = el('input', {type:'text', value:sub.service || '', placeholder:'e.g. Regular'});
  p.append(el('h4', {}, '2. Submission'), el('div', {class:'grid g2'}, field('Date submitted to PSA', sd), field('Service level', sv)),
    el('button', {class:'ghost', type:'button', onclick: () => run(() => J('POST', `/api/dataset/cards/${uid}/submission`, {submitted_on:sd.value, service:sv.value}))}, 'Save submission'));
  // result
  if (!sub.grade) {
    const g = el('select', {}, [...Array(10)].map((_, k) => el('option', {value:10 - k}, 'PSA ' + (10 - k)))), q = el('select', {}, vocab.qualifiers.map(x => el('option', {value:x}, x || 'no qualifier'))), cert = el('input', {type:'text', placeholder:'cert number'});
    p.append(el('h4', {}, '3. Result from PSA'), el('div', {class:'grid g2'}, field('Grade', g), field('Qualifier', q), field('Cert number', cert)),
      note('Recording the result freezes the features GradeForge saw for the most recent pre-grading photographs. Later reviews cannot change them.'),
      el('button', {class:'primary', type:'button', onclick: () => run(() => J('POST', `/api/dataset/cards/${uid}/result`, {grade:+g.value, qualifier:q.value, cert:cert.value}))}, 'Record result'));
  } else {
    const seen = el('input', {type:'text', value:sub.cert || ''}), pg = el('input', {type:'number', min:1, max:10}), by = el('input', {type:'text', placeholder:'your name'});
    p.append(el('h4', {}, '3. Result and verification'), kv([['Grade', `${sub.grade}${sub.qualifier ? ' ' + sub.qualifier : ''}`], ['Cert', sub.cert || 'not recorded'], ['Verification', pretty(sub.verification_status)]]),
      c.cert_page ? el('p', {}, 'Look the cert up on PSA\'s page, then type what it shows: ', el('a', {href:c.cert_page, target:'_blank', rel:'noopener'}, c.cert_page)) : el('p', {class:'note'}, 'Record the cert number to enable verification.'),
      el('div', {class:'grid g2'}, field('Cert number shown on the PSA page', seen), field('Grade shown on the PSA page', pg), field('Verified by', by)),
      el('button', {class:'ghost', type:'button', onclick: () => run(() => J('POST', `/api/dataset/cards/${uid}/verify`, {verified_by:by.value, cert_number_seen:seen.value, cert_page_grade:+pg.value}))}, 'Record verification'),
      note('GradeForge does not contact PSA and never marks a grade verified by itself; a mismatch is recorded as a conflict and the card is excluded.'));
  }
  const why = el('input', {type:'text', placeholder:'reason, e.g. re-holdered before grading'});
  p.append(el('div', {style:'margin-top:10px'}, why, ' ', el('button', {class:'ghost', type:'button', onclick: () => run(() => J('POST', `/api/dataset/cards/${uid}/exclude`, {reason:why.value}))}, 'Exclude this card')), out);
  return p;
}

// ---------------------------------------------------------------- annotation
async function annotate(box) {
  const list = (await api('/api/dataset/cards')).items.filter(c => c.n_images);
  if (!list.length) { box.append(el('p', {class:'muted'}, 'Link pre-grading photographs to a card first.')); return; }
  const cs = el('select', {}, list.map(c => el('option', {value:c.card_uid}, c.label || c.card_uid)));
  if (cur && list.some(c => c.card_uid === cur)) cs.value = cur;
  const holder = el('div'); box.append(el('div', {class:'panel'}, field('Card', cs), holder,
    note('Annotate BEFORE you know the PSA grade. Annotations made after the result are saved but marked not blind and excluded from detector evaluation.')));
  const draw = async () => {
    const c = await api('/api/dataset/cards/' + cs.value), imgs = c.images.filter(i => i.phase === 'pre_grading');
    holder.replaceChildren();
    if (!imgs.length) { holder.append(el('p', {class:'muted'}, 'No pre-grading photographs linked.')); return; }
    const ia = el('select', {}, imgs.map(i => el('option', {value:i.analysis_id}, i.analysis_id))), side = el('select', {}, ['front', 'back'].map(s => el('option', {value:s}, s)));
    const kind = el('select', {}, vocab.annotation_kinds.map(k => el('option', {value:k}, pretty(k)))), sev = el('select', {}, ['minor', 'moderate', 'severe'].map(s => el('option', {value:s}, s)));
    const who = el('input', {type:'text', placeholder:'annotator name', value:localStorage.getItem('gf_annotator') || ''}), msg = el('span');
    const img = el('img', {style:'width:100%;display:block;user-select:none', draggable:'false'}), cv = el('canvas', {style:'position:absolute;left:0;top:0;width:100%;height:100%;cursor:crosshair'});
    const wrap = el('div', {style:'position:relative;max-width:520px;margin:8px 0'}, img, cv);
    let anns = [], pend = null, start = null;
    const paint = () => { cv.width = img.naturalWidth || 1; cv.height = img.naturalHeight || 1; const x = cv.getContext('2d'); x.clearRect(0, 0, cv.width, cv.height); x.lineWidth = Math.max(2, cv.width / 400); x.font = `${Math.max(12, cv.width / 60)}px sans-serif`;
      anns.filter(a => a.side === side.value).forEach(a => { x.strokeStyle = a.blind ? '#4cc2ff' : '#e6a01e'; x.strokeRect(...a.box); x.fillStyle = x.strokeStyle; x.fillText(pretty(a.kind), a.box[0], a.box[1] - 4); });
      if (pend) { x.strokeStyle = '#3cb371'; x.strokeRect(...pend); } };
    const load = async () => { anns = (await api(`/api/dataset/annotations?analysis_id=${ia.value}`)).items; img.src = `/api/history/${ia.value}/image/${side.value}`; };
    img.onload = () => { paint(); renderList(); };
    const pos = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height]; };
    cv.onmousedown = e => { start = pos(e); pend = null; }; cv.onmousemove = e => { if (!start) return; const p = pos(e); pend = [Math.min(start[0], p[0]), Math.min(start[1], p[1]), Math.abs(p[0] - start[0]), Math.abs(p[1] - start[1])]; paint(); };
    cv.onmouseup = () => { start = null; };
    const lst = el('div');
    const renderList = () => { lst.replaceChildren(el('ul', {}, anns.filter(a => a.side === side.value).map(a => el('li', {}, `${pretty(a.kind)} (${a.severity}) by ${a.annotator}${a.blind ? '' : ' — NOT blind'} `, el('button', {class:'ghost', type:'button', onclick: async () => { await api('/api/dataset/annotations/' + a.id, {method:'DELETE'}); await load(); }}, 'withdraw'))))); };
    ia.onchange = side.onchange = load;
    holder.append(el('div', {class:'grid g2'}, field('Photograph', ia), field('Side', side)), wrap,
      el('div', {class:'grid g2'}, field('Defect kind', kind), field('Severity', sev), field('Annotator', who)),
      el('button', {class:'primary', type:'button', onclick: async () => {
        if (!pend) return say(msg, ' drag a box on the image first', true);
        try { localStorage.setItem('gf_annotator', who.value); } catch {}
        try { const r = await J('POST', '/api/dataset/annotations', {card_uid:cs.value, analysis_id:ia.value, side:side.value, kind:kind.value, box:pend, severity:sev.value, annotator:who.value});
          pend = null; say(msg, r.warning || ' saved (blind)', !!r.warning); await load(); } catch (e) { say(msg, ' ' + e.message, true); } }}, 'Save annotation'), ' ',
      el('button', {class:'ghost', type:'button', onclick: async () => { await J('POST', '/api/dataset/annotations/complete', {analysis_id:ia.value, side:side.value, annotator:who.value}); say(msg, ' marked: this side is completely annotated'); }}, 'This side is completely annotated'), msg, lst,
      note('Marking a side complete lets GradeForge count detector findings you did not mark as false alarms. Only do it if you really checked the whole side.'));
    await load();
  };
  cs.onchange = draw; await draw();
}

// ---------------------------------------------------------------- models
function perfBlock(p) {
  if (!p || !p.n) return el('p', {class:'muted'}, 'No eligible cards.');
  const d = el('div'), m = p.model, ga = p.grade_accuracy;
  d.append(kv([['Cards', `${p.n} (${p.class_balance.psa10} PSA 10, ${p.class_balance.not_psa10} not)`]]));
  if (m) d.append(el('h4', {}, `Model at threshold ${Number(m.threshold).toFixed(3)}`), kv([['PSA 10 precision', rate(m.psa10_precision)], ['PSA 10 recall', rate(m.psa10_recall)], ['False-positive rate', rate(m.false_positive_rate)],
    ['AUC (95% CI)', `${m.auc} (${m.auc_ci95.join('–')})`], ['Brier vs base rate', `${m.brier} vs ${m.baseline_brier}`], ['Calibration slope', p.calibration ? `${p.calibration.slope} (95% CI ${(p.calibration.slope_ci95 || []).join('–')})` : 'n/a']]));
  if (p.nine_vs_ten) { const n = p.nine_vs_ten; d.append(el('h4', {}, 'PSA 9 vs PSA 10 only'), kv([['Cards', `${n.n_psa10} tens, ${n.n_psa9} nines`], ['AUC', n.auc], ['Precision', rate(n.psa10_precision)], ['Recall', rate(n.psa10_recall)], ['False-positive rate (9 called 10)', rate(n.false_positive_rate)]])); }
  const rows = [['majority grade baseline', ga.majority_grade_baseline], ['always predict 9', ga.always_9_baseline], ['rules ceiling baseline', ga.conservative_ceiling_baseline], ['system (model + ceiling)', ga.system]].filter(r => r[1] && r[1].n);
  if (rows.length) d.append(el('h4', {}, 'Exact and within-one-grade accuracy'), el('table', {}, el('tr', {}, ['Predictor', 'Cards', 'Exact', 'Within one'].map(h => el('th', {}, h))), rows.map(([k, v]) => el('tr', {}, el('td', {}, k), el('td', {}, v.n), el('td', {}, rate(v.exact)), el('td', {}, rate(v.within_one))))), note(ga.note));
  d.append(el('h4', {}, 'Rules shortlist (credible or uncertain)'), kv([['Precision', rate(p.rules_shortlist.psa10_precision)], ['Recall', rate(p.rules_shortlist.psa10_recall)], ['False-positive rate', rate(p.rules_shortlist.false_positive_rate)]]));
  return d;
}
function groupBlock(g) {
  return el('div', {}, el('h4', {}, 'By ' + pretty(g.group_by)), el('table', {}, el('tr', {}, ['Group', 'Cards', 'PSA 10 rate', 'Model precision', 'Model recall', 'Note'].map(h => el('th', {}, h))),
    g.groups.map(x => el('tr', {}, el('td', {}, pretty(x.group)), el('td', {}, x.n), el('td', {}, rate(x.psa10)), el('td', {}, x.metrics?.model ? rate(x.metrics.model.psa10_precision) : '—'), el('td', {}, x.metrics?.model ? rate(x.metrics.model.psa10_recall) : '—'), el('td', {}, x.suppressed || '')))), note(g.warning));
}
async function models(box) {
  const m = await api('/api/dataset/models'), out = el('div'), res = el('div');
  const tp = el('input', {type:'number', step:'0.05', min:'0.5', max:'0.99', placeholder:'e.g. 0.85 (optional)', style:'width:200px'});
  box.append(el('div', {class:'panel'}, el('h3', {style:'margin-top:0'}, 'Retrain'),
    el('p', {}, m.readiness.ready_to_train ? 'Enough eligible cards to train a candidate.' : 'Not enough eligible cards yet: ' + Object.entries(m.readiness.requirements).filter(([k]) => !m.readiness.satisfied[k]).map(([k, v]) => `${k} needs ${v}`).join('; ')),
    field('Target PSA 10 precision for the decision threshold (chosen on validation data)', tp),
    el('button', {class:'primary', type:'button', onclick: async () => { try { const r = await J('POST', '/api/dataset/models/retrain', {note:'ui', target_precision: tp.value ? +tp.value : null}); res.replaceChildren(r.trained ? el('div', {}, el('b', {}, `Candidate ${r.version} trained.`), note(r.validation_note), perfBlock(r.validation_performance)) : el('p', {}, r.reason)); } catch (e) { say(out, e.message, true); } }}, 'Train a candidate'), out, res));
  const t = el('table', {}, el('tr', {}, ['Version', 'Status', 'Test AUC', 'Test Brier gain', 'Synthetic', ''].map(h => el('th', {}, h))));
  const ack = el('input', {type:'checkbox'});
  m.items.forEach(v => t.append(el('tr', {}, el('td', {}, v.version), el('td', {}, pretty(v.status)), el('td', {}, v.test?.auc ?? '—'), el('td', {}, v.test?.brier_improvement ?? '—'), el('td', {}, v.synthetic ? 'YES' : 'no'), el('td', {},
    v.status === 'candidate' ? el('button', {class:'ghost', type:'button', onclick: async () => { try { const r = await J('POST', `/api/dataset/models/${v.version}/test`, {confirm:ack.checked}); res.replaceChildren(r.evaluated ? el('div', {}, el('b', {}, `${pretty(r.status)}: ${r.meaning}`), perfBlock(r.performance), kv([['Pre-declared checks', JSON.stringify(r.validity.checks)], ['Slope 95% CI', (r.validity.slope_ci95 || []).join('–')]])) : el('p', {}, r.reason)); } catch (e) { say(out, e.message, true); } }}, 'Run test (once)') : null,
    v.status === 'passed_test' ? el('button', {class:'ghost', type:'button', onclick: async () => { try { const r = await J('POST', `/api/dataset/models/${v.version}/promote`); say(out, ' promoted ' + r.promoted); boot(); } catch (e) { say(out, e.message, true); } }}, 'Promote') : null, ' ',
    el('button', {class:'ghost', type:'button', onclick: async () => { const r = await api(`/api/dataset/models/${v.version}/report?group_by=card_type`); res.replaceChildren(el('b', {}, `Report on the ${r.partition} partition`), r.caveat ? note(r.caveat) : null, perfBlock(r.overall), groupBlock(r.by_group)); }}, 'Report')))));
  box.append(el('div', {class:'panel'}, el('h3', {style:'margin-top:0'}, 'Model versions (active: ' + (m.champion || 'none') + ')'),
    m.items.length ? t : el('p', {class:'muted'}, 'No models yet.'),
    el('label', {style:'display:flex;gap:6px;align-items:center'}, ack, 'I understand the test partition can be used only once per model version.'),
    el('button', {class:'ghost', type:'button', onclick: async () => { try { const r = await J('POST', '/api/dataset/models/rollback'); say(out, ' now active: ' + r.now_active); boot(); } catch (e) { say(out, e.message, true); } }}, 'Roll back active model'),
    el('button', {class:'ghost', type:'button', onclick: async () => { const r = await api('/api/dataset/monitor'); res.replaceChildren(r.n ? el('div', {}, el('b', {}, `Prospective check on ${r.n} newer cards`), perfBlock(r.performance), note(r.note)) : el('p', {}, r.message)); }}, 'Check newer cards')));
}

// ---------------------------------------------------------------- integrity
async function integrity(box) {
  const [a, sel, er, ann, rr] = await Promise.all(['leakage-audit', 'selection-bias', 'errors', 'annotations/evaluation', 'rules-report'].map(p => api('/api/dataset/' + p)));
  box.append(el('div', {class:'panel'}, el('h3', {style:'margin-top:0'}, 'Leakage audit: ' + (a.passed ? 'all checks pass' : 'FAILED')),
    el('ul', {}, a.checks.map(c => el('li', {style:`color:${c.passed ? 'var(--ok)' : 'var(--bad)'}`}, `${c.passed ? '✓' : '✗'} ${c.check}${c.passed ? '' : ' — ' + c.detail + ': ' + c.offenders.join(', ')}`))),
    el('p', {class:'note'}, 'Not checkable automatically: ' + a.not_checkable.join(' '))));
  box.append(el('div', {class:'panel'}, el('h3', {style:'margin-top:0'}, 'Selection and photo-quality bias'),
    kv([['Photographed but never submitted', sel.not_submitted_vs_graded.n_not_submitted_with_photos], ['Reading', sel.not_submitted_vs_graded.reading],
        ['Pipeline abstentions among graded', `${sel.pipeline_abstentions_among_graded.n}; PSA 10 rate ${rate(sel.pipeline_abstentions_among_graded.psa10_rate)}`]]),
    el('h4', {}, 'PSA 10 rate by declared reason for submitting'), kv(Object.entries(sel.psa10_rate_by_declared_selection_basis).map(([k, v]) => [pretty(k), rate(v)])),
    el('h4', {}, 'PSA 10 rate by photo-quality tier'), kv(Object.entries(sel.psa10_rate_by_photo_quality_tier).map(([k, v]) => [k, rate(v)])),
    el('h4', {}, 'What cannot be corrected'), el('ul', {}, sel.what_cannot_be_corrected.map(x => el('li', {}, x)))));
  box.append(el('div', {class:'panel'}, el('h3', {style:'margin-top:0'}, 'Prediction error tracking'),
    er.n ? el('div', {}, kv([['Graded cards with a recorded prediction', er.n], ['PSA above the rules ceiling', rate(er.ceiling_violations)]]), er.probability_errors ? kv(Object.entries(er.probability_errors.by_model_version).map(([k, v]) => [k, `n=${v.n}, Brier ${v.brier}, false positives ${v.false_positives}, false negatives ${v.false_negatives}`])) : note('No model probabilities have been logged yet.')) : el('p', {class:'muted'}, er.message)));
  box.append(el('div', {class:'panel'}, el('h3', {style:'margin-top:0'}, 'Rules-only performance by card type'), rr.n ? el('div', {}, perfBlock(rr.overall), groupBlock(rr.by_group)) : el('p', {class:'muted'}, 'No eligible cards.')));
  box.append(el('div', {class:'panel'}, el('h3', {style:'margin-top:0'}, 'Detector accuracy against your blind annotations'),
    ann.n_blind_annotations ? el('div', {}, el('table', {}, el('tr', {}, ['Defect kind', 'Annotated', 'Detector recall', 'Precision (fully annotated sides)'].map(h => el('th', {}, h))),
      Object.entries(ann.by_kind).map(([k, v]) => el('tr', {}, el('td', {}, pretty(k)), el('td', {}, v.annotated), el('td', {}, v.annotated ? `${v.recall} (${v.found}/${v.annotated})` : '—'), el('td', {}, v.detections ? `${v.precision_on_fully_annotated_sides} (${v.detections_matched}/${v.detections})` : '—')))), note(ann.matching_rule)) : el('p', {class:'muted'}, 'No blind annotations yet.')));
}
})();
