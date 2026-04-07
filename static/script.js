// ── THEME ──────────────────────────────────────────────────
function toggleTheme(){ document.body.classList.toggle('light'); }

// ── SPEAKER COLORS ─────────────────────────────────────────
const SP_COLORS = ['#4f8ef7','#a78bfa','#34d399','#fb7185','#fbbf24','#22d3ee','#f97316','#ec4899'];

function spColor(index){ return SP_COLORS[index % SP_COLORS.length]; }

function spIndexMap(metrics){
  const map = {};
  Object.keys(metrics).forEach((sp,i) => { map[sp] = i; });
  return map;
}

// ── BEST / WORST HELPERS ───────────────────────────────────
function computeBestWorst(metrics, spScores){
  const speakers = Object.keys(metrics);

  const get = (sp, key) => {
    const m = metrics[sp] || {};
    const s = spScores[sp] || {};
    switch(key){
      case 'speaking':     return m.speaking_share_percent;
      case 'turns':        return m.num_turns;
      case 'words':        return m.avg_words_per_turn;
      case 'duration':     return m.avg_duration_per_turn_sec;
      case 'questions':    return m.questions_asked;
      case 'vocab':        return m.vocabulary_richness;
      case 'filler':       return m.filler_rate;   // LOWER is best
      case 'align':        return m.agenda_alignment_percent;
      case 'coverage':     return m.topic_coverage_percent;
      case 'sentiment':    return m.sentiment_score;
      case 'confidence':   return m.confidence_score;
      case 'contrib':      return s.contribution_quality ?? 0;
      case 'interact':     return s.interaction_score ?? 0;
      case 'decision':     return s.decision_impact ?? 0;
      default: return 0;
    }
  };

  const higherIsBest = ['speaking','turns','words','duration','questions','vocab','align','coverage','sentiment','confidence','contrib','interact','decision'];
  const lowerIsBest  = ['filler'];
  const allKeys = [...higherIsBest, ...lowerIsBest];

  const result = {};
  allKeys.forEach(key => {
    const vals = speakers.map(sp => get(sp, key));
    const best  = lowerIsBest.includes(key) ? Math.min(...vals) : Math.max(...vals);
    const worst = lowerIsBest.includes(key) ? Math.max(...vals) : Math.min(...vals);
    const bestSps  = speakers.filter(sp => get(sp, key) === best);
    const worstSps = speakers.filter(sp => get(sp, key) === worst);
    result[key] = { best, worst, bestSps, worstSps };
  });
  return result;
}

function metricBoxClass(sp, key, bw){
  const { bestSps, worstSps } = bw[key] || {};
  if(!bestSps || !worstSps) return '';
  if(bestSps.includes(sp))  return bestSps.length  > 1 ? 'is-tie-best'  : 'is-best';
  if(worstSps.includes(sp)) return worstSps.length > 1 ? 'is-tie-worst' : 'is-worst';
  return '';
}

function badgeLabel(cls){
  if(cls === 'is-best')       return '<span class="metric-badge">Best</span>';
  if(cls === 'is-worst')      return '<span class="metric-badge">Worst</span>';
  if(cls === 'is-tie-best')   return '<span class="metric-badge">Tied Best</span>';
  if(cls === 'is-tie-worst')  return '<span class="metric-badge">Tied Worst</span>';
  return '';
}

function mbox(label, displayVal, cls){
  return `<div class="metric-box ${cls}">${badgeLabel(cls)}<div class="metric-label">${label}</div><div class="metric-value">${displayVal}</div></div>`;
}

// ── CHART DEFAULTS ─────────────────────────────────────────
Chart.defaults.color = '#6b7a99';
Chart.defaults.font.family = "'DM Sans', sans-serif";

function chartBg(){ return document.body.classList.contains('light') ? '#f0f3f9' : '#060a12'; }

const CHART_REGISTRY = {};

function makeChart(id, config){
  if(CHART_REGISTRY[id]) { CHART_REGISTRY[id].destroy(); }
  const ctx = document.getElementById(id);
  if(!ctx) return;
  CHART_REGISTRY[id] = new Chart(ctx.getContext('2d'), config);
}

// ── CHARTS SECTION ─────────────────────────────────────────
function buildChartsSection(data){
  const metrics   = data.metrics;
  const spScores  = data.analysis?.speaker_scores || {};
  const speakers  = Object.keys(metrics);
  const siMap     = spIndexMap(metrics);

  // After DOM insertion, build all charts
  setTimeout(() => {
    buildPieChart('chartSpeakingPie',    metrics, speakers, siMap);
    buildTurnsBar('chartTurnsBar',       metrics, speakers, siMap);
    buildWordsDurationBar('chartWordsBar', metrics, speakers, siMap);
    buildTopicAlignBar('chartAlignBar',  metrics, speakers, siMap);
    buildFillerVocabBar('chartFillerBar',metrics, speakers, siMap);
    buildSentConfBar('chartSentConf',    metrics, speakers, siMap, spScores);
    buildRadar('chartRadar',             metrics, speakers, siMap, spScores);
    buildScoreBar('chartScoreBar',       data.final_scores?.ranking || [], siMap);
  }, 80);

  const colorLegend = speakers.map((sp,i) => `
    <span style="display:inline-flex;align-items:center;gap:5px;margin:0 10px 6px 0;">
      <span style="width:10px;height:10px;border-radius:50%;background:${spColor(i)};display:inline-block;"></span>
      <span style="font-size:12px;color:var(--text);">${sp}</span>
    </span>`).join('');

  return `
  <div class="compare-section fade-in fade-in-d2">
    <div class="glass">
      <div class="section-label">Visual Comparison</div>
      <div class="section-title">Speaker Analytics Dashboard</div>
      <div style="margin-bottom:18px;display:flex;flex-wrap:wrap;">${colorLegend}</div>

      <div class="charts-mastergrid">

        <!-- PIE: Speaking Share -->
        <div class="chart-card">
          <div class="chart-title">Speaking Share
            <div class="chart-sub">% of total speaking time per speaker</div>
          </div>
          <div class="chart-canvas-wrap" style="max-width:280px;margin:auto;">
            <canvas id="chartSpeakingPie"></canvas>
          </div>
        </div>

        <!-- BAR: Number of Turns -->
        <div class="chart-card">
          <div class="chart-title">Number of Turns
            <div class="chart-sub">How many times each speaker spoke</div>
          </div>
          <div class="chart-canvas-wrap">
            <canvas id="chartTurnsBar" height="170"></canvas>
          </div>
        </div>

        <!-- BAR: Avg Words & Duration -->
        <div class="chart-card">
          <div class="chart-title">Avg Words & Duration per Turn
            <div class="chart-sub">Words per turn (bar) · Seconds per turn (line)</div>
          </div>
          <div class="chart-canvas-wrap">
            <canvas id="chartWordsBar" height="170"></canvas>
          </div>
        </div>

        <!-- BAR: Topic Alignment & Coverage -->
        <div class="chart-card">
          <div class="chart-title">Topic Alignment & Coverage
            <div class="chart-sub">Agenda alignment % · Topic coverage %</div>
          </div>
          <div class="chart-canvas-wrap">
            <canvas id="chartAlignBar" height="170"></canvas>
          </div>
        </div>

        <!-- BAR: Filler Rate & Vocab Richness -->
        <div class="chart-card">
          <div class="chart-title">Filler Rate vs Vocabulary Richness
            <div class="chart-sub">Lower filler = better · Higher vocab = better</div>
          </div>
          <div class="chart-canvas-wrap">
            <canvas id="chartFillerBar" height="170"></canvas>
          </div>
        </div>

        <!-- BAR: Sentiment + Confidence + LLM Scores -->
        <div class="chart-card">
          <div class="chart-title">Sentiment · Confidence · Contribution · Interaction · Decision
            <div class="chart-sub">Multi-dimension LLM + heuristic scores</div>
          </div>
          <div class="chart-canvas-wrap">
            <canvas id="chartSentConf" height="170"></canvas>
          </div>
        </div>

        <!-- RADAR: Full multi-metric -->
        <div class="chart-card full-w">
          <div class="chart-title">Multi-Metric Radar — All Speakers
            <div class="chart-sub">Normalised across 9 dimensions: speaking share · turns · avg words · agenda alignment · topic coverage · vocabulary richness · confidence · contribution quality · interaction score</div>
          </div>
          <div class="chart-canvas-wrap" style="max-width:520px;margin:auto;">
            <canvas id="chartRadar"></canvas>
          </div>
        </div>

        <!-- BAR: Final Score -->
        <div class="chart-card full-w">
          <div class="chart-title">Final Composite Score
            <div class="chart-sub">Weighted overall performance score (0–100)</div>
          </div>
          <div class="chart-canvas-wrap">
            <canvas id="chartScoreBar" height="100"></canvas>
          </div>
        </div>

      </div>
    </div>
  </div>`;
}

// ── CHART BUILDERS ─────────────────────────────────────────
function buildPieChart(id, metrics, speakers, siMap){
  const vals   = speakers.map(sp => metrics[sp].speaking_share_percent);
  const colors = speakers.map((sp,i) => spColor(i));
  makeChart(id, {
    type: 'doughnut',
    data: {
      labels: speakers,
      datasets: [{ data: vals, backgroundColor: colors, borderColor: 'rgba(0,0,0,0)', hoverOffset: 6 }]
    },
    options: {
      cutout: '62%',
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 12, padding: 16 } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw}%` } }
      }
    }
  });
}

function buildTurnsBar(id, metrics, speakers, siMap){
  makeChart(id, {
    type: 'bar',
    data: {
      labels: speakers,
      datasets: [{
        label: 'Turns',
        data: speakers.map(sp => metrics[sp].num_turns),
        backgroundColor: speakers.map((sp,i) => spColor(i)+'cc'),
        borderColor:     speakers.map((sp,i) => spColor(i)),
        borderWidth: 1.5,
        borderRadius: 6,
      }]
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b7a99' } },
        y: { grid: { display: false }, ticks: { color: '#6b7a99' } }
      }
    }
  });
}

function buildWordsDurationBar(id, metrics, speakers, siMap){
  makeChart(id, {
    type: 'bar',
    data: {
      labels: speakers,
      datasets: [
        {
          label: 'Avg Words / Turn',
          type: 'bar',
          data: speakers.map(sp => metrics[sp].avg_words_per_turn),
          backgroundColor: speakers.map((sp,i) => spColor(i)+'99'),
          borderColor:     speakers.map((sp,i) => spColor(i)),
          borderWidth: 1.5,
          borderRadius: 5,
          yAxisID: 'y',
        },
        {
          label: 'Avg Duration (s)',
          type: 'line',
          data: speakers.map(sp => metrics[sp].avg_duration_per_turn_sec),
          borderColor: '#fbbf24',
          backgroundColor: 'rgba(251,191,36,0.12)',
          borderWidth: 2,
          pointRadius: 5,
          pointBackgroundColor: '#fbbf24',
          tension: 0.35,
          fill: true,
          yAxisID: 'y2',
        }
      ]
    },
    options: {
      plugins: { legend: { labels: { boxWidth: 12 } } },
      scales: {
        y:  { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#6b7a99' } },
        y2: { position: 'right', grid: { display: false }, ticks: { color: '#fbbf24' } },
        x:  { grid: { display: false }, ticks: { color: '#6b7a99' } }
      }
    }
  });
}

function buildTopicAlignBar(id, metrics, speakers, siMap){
  makeChart(id, {
    type: 'bar',
    data: {
      labels: speakers,
      datasets: [
        {
          label: 'Agenda Alignment %',
          data: speakers.map(sp => metrics[sp].agenda_alignment_percent),
          backgroundColor: 'rgba(79,142,247,0.55)',
          borderColor:     '#4f8ef7',
          borderWidth: 1.5,
          borderRadius: 5,
        },
        {
          label: 'Topic Coverage %',
          data: speakers.map(sp => metrics[sp].topic_coverage_percent),
          backgroundColor: 'rgba(167,139,250,0.55)',
          borderColor:     '#a78bfa',
          borderWidth: 1.5,
          borderRadius: 5,
        }
      ]
    },
    options: {
      plugins: { legend: { labels: { boxWidth: 12 } } },
      scales: {
        y: { max: 100, grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#6b7a99' } },
        x: { grid: { display: false }, ticks: { color: '#6b7a99' } }
      }
    }
  });
}

function buildFillerVocabBar(id, metrics, speakers, siMap){
  makeChart(id, {
    type: 'bar',
    data: {
      labels: speakers,
      datasets: [
        {
          label: 'Filler Rate',
          data: speakers.map(sp => metrics[sp].filler_rate),
          backgroundColor: 'rgba(251,113,133,0.55)',
          borderColor: '#fb7185',
          borderWidth: 1.5,
          borderRadius: 5,
          yAxisID: 'y',
        },
        {
          label: 'Vocabulary Richness',
          data: speakers.map(sp => metrics[sp].vocabulary_richness),
          backgroundColor: 'rgba(52,211,153,0.55)',
          borderColor: '#34d399',
          borderWidth: 1.5,
          borderRadius: 5,
          yAxisID: 'y2',
        }
      ]
    },
    options: {
      plugins: { legend: { labels: { boxWidth: 12 } } },
      scales: {
        y:  { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#fb7185' } },
        y2: { position: 'right', grid: { display: false }, ticks: { color: '#34d399' } },
        x:  { grid: { display: false }, ticks: { color: '#6b7a99' } }
      }
    }
  });
}

function buildSentConfBar(id, metrics, speakers, siMap, spScores){
  makeChart(id, {
    type: 'bar',
    data: {
      labels: speakers,
      datasets: [
        {
          label: 'Sentiment',
          data: speakers.map(sp => metrics[sp].sentiment_score),
          backgroundColor: 'rgba(52,211,153,0.55)',
          borderColor: '#34d399',
          borderWidth: 1.5,
          borderRadius: 4,
        },
        {
          label: 'Confidence',
          data: speakers.map(sp => metrics[sp].confidence_score),
          backgroundColor: 'rgba(251,191,36,0.55)',
          borderColor: '#fbbf24',
          borderWidth: 1.5,
          borderRadius: 4,
        },
        {
          label: 'Contribution Quality',
          data: speakers.map(sp => spScores[sp]?.contribution_quality ?? 0),
          backgroundColor: 'rgba(79,142,247,0.55)',
          borderColor: '#4f8ef7',
          borderWidth: 1.5,
          borderRadius: 4,
        },
        {
          label: 'Interaction Score',
          data: speakers.map(sp => spScores[sp]?.interaction_score ?? 0),
          backgroundColor: 'rgba(167,139,250,0.55)',
          borderColor: '#a78bfa',
          borderWidth: 1.5,
          borderRadius: 4,
        },
        {
          label: 'Decision Impact',
          data: speakers.map(sp => spScores[sp]?.decision_impact ?? 0),
          backgroundColor: 'rgba(251,113,133,0.55)',
          borderColor: '#fb7185',
          borderWidth: 1.5,
          borderRadius: 4,
        },
      ]
    },
    options: {
      plugins: { legend: { labels: { boxWidth: 12 } } },
      scales: {
        y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#6b7a99' } },
        x: { grid: { display: false }, ticks: { color: '#6b7a99' } }
      }
    }
  });
}

function buildRadar(id, metrics, speakers, siMap, spScores){
  const labels = [
    'Speaking %', 'Turns', 'Avg Words', 'Agenda Align %',
    'Topic Cover %', 'Vocab Richness', 'Confidence', 'Contrib Quality', 'Interaction'
  ];

  // Normalise each dimension 0–1 across speakers
  function normalise(arr){
    const mn = Math.min(...arr), mx = Math.max(...arr);
    if(mx === mn) return arr.map(() => 0.5);
    return arr.map(v => (v - mn) / (mx - mn));
  }

  const raw = {
    speaking:  speakers.map(sp => metrics[sp].speaking_share_percent),
    turns:     speakers.map(sp => metrics[sp].num_turns),
    words:     speakers.map(sp => metrics[sp].avg_words_per_turn),
    align:     speakers.map(sp => metrics[sp].agenda_alignment_percent),
    coverage:  speakers.map(sp => metrics[sp].topic_coverage_percent),
    vocab:     speakers.map(sp => metrics[sp].vocabulary_richness),
    conf:      speakers.map(sp => metrics[sp].confidence_score),
    contrib:   speakers.map(sp => spScores[sp]?.contribution_quality ?? 0),
    interact:  speakers.map(sp => spScores[sp]?.interaction_score ?? 0),
  };

  const normed = {
    speaking:  normalise(raw.speaking),
    turns:     normalise(raw.turns),
    words:     normalise(raw.words),
    align:     normalise(raw.align),
    coverage:  normalise(raw.coverage),
    vocab:     normalise(raw.vocab),
    conf:      normalise(raw.conf),
    contrib:   normalise(raw.contrib),
    interact:  normalise(raw.interact),
  };

  const datasets = speakers.map((sp, i) => {
    const color = spColor(i);
    return {
      label: sp,
      data: [
        normed.speaking[i], normed.turns[i], normed.words[i],
        normed.align[i], normed.coverage[i], normed.vocab[i],
        normed.conf[i], normed.contrib[i], normed.interact[i]
      ],
      borderColor: color,
      backgroundColor: color + '22',
      pointBackgroundColor: color,
      pointRadius: 4,
      borderWidth: 2,
    };
  });

  makeChart(id, {
    type: 'radar',
    data: { labels, datasets },
    options: {
      plugins: { legend: { labels: { boxWidth: 12 } } },
      scales: {
        r: {
          min: 0, max: 1,
          ticks: { display: false },
          grid:  { color: 'rgba(255,255,255,0.07)' },
          pointLabels: { font: { size: 12 }, color: '#6b7a99' }
        }
      }
    }
  });
}

function buildScoreBar(id, ranking, siMap){
  const labels  = ranking.map(r => r.speaker);
  const scores  = ranking.map(r => r.score);
  const colors  = labels.map(sp => {
    const i = siMap[sp] ?? 0;
    return spColor(i);
  });

  makeChart(id, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Final Score',
        data: scores,
        backgroundColor: colors.map(c => c + 'aa'),
        borderColor: colors,
        borderWidth: 2,
        borderRadius: 7,
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { max: 100, grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#6b7a99' } },
        x: { grid: { display: false }, ticks: { color: '#6b7a99', font: { size: 13 } } }
      }
    }
  });
}

// ── ANALYSIS BLOCK ─────────────────────────────────────────
function analysisBlock(data){
  const a = data.analysis || {};
  return `
  <div class="glass fade-in fade-in-d1">
    <div class="section-label">AI Analysis</div>
    <div class="section-title">Meeting Intelligence</div>

    <div class="analysis-full">
      <div class="analysis-item-label">Summary</div>
      <div class="analysis-item-value">${a.summary || '—'}</div>
    </div>

    <div class="analysis-grid">
      <div class="analysis-item">
        <div class="analysis-item-label">Intent</div>
        <div class="analysis-item-value">${a.intent || '—'}</div>
      </div>
      <div class="analysis-item">
        <div class="analysis-item-label">Decision Impact</div>
        <div class="analysis-item-value">${a.decision_impact || '—'}</div>
      </div>
    </div>

    <div class="analysis-full" style="margin-bottom:0">
      <div class="analysis-item-label">Action Items</div>
      <div class="analysis-item-value">${a.action_items || '—'}</div>
    </div>
  </div>`;
}

// ── SPEAKER TABS ───────────────────────────────────────────
function speakerTabsBlock(data){
  const metrics  = data.metrics;
  const spScores = data.analysis?.speaker_scores || {};
  const speakers = Object.keys(metrics);
  const bw       = computeBestWorst(metrics, spScores);
  const siMap    = spIndexMap(metrics);

  let tabBtns = '';
  let tabPanes = '';

  speakers.forEach((sp, i) => {
    const m   = metrics[sp];
    const s   = spScores[sp] || {};
    const col = spColor(i);
    const active = i === 0;

    tabBtns += `
    <button class="sp-tab-btn ${active?'active':''}" onclick="switchSpTab('${sp}', this)">
      <span class="sp-dot" style="background:${col}"></span>
      ${sp}
    </button>`;

    const mkbox = (label, val, key, extra) => {
      const cls = metricBoxClass(sp, key, bw);
      return mbox(label, val !== undefined && val !== null ? val + (extra||'') : '—', cls);
    };

    tabPanes += `
    <div class="sp-pane" id="spPane_${sp.replace(' ','_')}" style="display:${active?'block':'none'}">
      <div class="metrics-grid">
        ${mkbox('Speaking Share',       m.speaking_share_percent,         'speaking',  '%')}
        ${mkbox('Turns',                m.num_turns,                      'turns',     '')}
        ${mkbox('Avg Words / Turn',     m.avg_words_per_turn,             'words',     '')}
        ${mkbox('Avg Duration / Turn',  m.avg_duration_per_turn_sec,      'duration',  's')}
        ${mkbox('Questions Asked',      m.questions_asked,                'questions', '')}
        ${mkbox('Vocabulary Richness',  m.vocabulary_richness,            'vocab',     '')}
        ${mkbox('Filler Rate',          m.filler_rate,                    'filler',    '')}
        ${mkbox('Agenda Alignment',     m.agenda_alignment_percent,       'align',     '%')}
        ${mkbox('Topic Coverage',       m.topic_coverage_percent,         'coverage',  '%')}
        ${mkbox('Sentiment Score',      m.sentiment_score,                'sentiment', '')}
        ${mkbox('Confidence Score',     m.confidence_score,               'confidence','')}
        ${mkbox('Contribution Quality', s.contribution_quality ?? '—',    'contrib',   '')}
        ${mkbox('Interaction Score',    s.interaction_score ?? '—',       'interact',  '')}
        ${mkbox('Decision Impact',      s.decision_impact ?? '—',         'decision',  '')}
      </div>
    </div>`;
  });

  return `
  <div class="glass fade-in fade-in-d3">
    <div class="section-label">Per-Speaker Breakdown</div>
    <div class="section-title">Detailed Metrics</div>
    <div class="sp-nav">${tabBtns}</div>
    ${tabPanes}
  </div>`;
}

function switchSpTab(sp, btn){
  document.querySelectorAll('.sp-pane').forEach(p => p.style.display = 'none');
  document.querySelectorAll('.sp-tab-btn').forEach(b => b.classList.remove('active'));
  const pane = document.getElementById('spPane_' + sp.replace(' ','_'));
  if(pane) pane.style.display = 'block';
  btn.classList.add('active');
}

// ── LEADERBOARD ────────────────────────────────────────────
function leaderboardBlock(data){
  const ranking = data.final_scores?.ranking || [];
  const siMap   = spIndexMap(data.metrics);
  const rankLabels = ['🥇','🥈','🥉'];

  let rows = ranking.map((r, i) => {
    const col = spColor(siMap[r.speaker] ?? i);
    const rCls = i < 3 ? `r${i+1}` : '';
    return `
    <div class="lb-item">
      <div class="lb-rank ${rCls}">${rankLabels[i] || (i+1)}</div>
      <div class="lb-info">
        <div class="lb-name" style="color:${col}">${r.speaker}</div>
        <div class="lb-bar-wrap">
          <div class="lb-bar-fill" style="width:${r.score}%;background:${col}"></div>
        </div>
      </div>
      <div class="lb-score" style="color:${col}">${r.score}</div>
    </div>`;
  }).join('');

  return `
  <div class="glass fade-in fade-in-d2">
    <div class="section-label">Rankings</div>
    <div class="section-title">Leaderboard</div>
    ${rows}
  </div>`;
}

// ── INSIGHTS ───────────────────────────────────────────────
function insightsBlock(data){
  const explanations = data.explanations || {};
  const siMap        = spIndexMap(data.metrics);

  let cards = Object.entries(explanations).map(([sp, e]) => {
    const col = spColor(siMap[sp] ?? 0);
    const strengths = (e.strengths || []).map(s => `<span class="insight-pill pill-strength">${s}</span>`).join('');
    const weaknesses = (e.weaknesses || []).map(w => `<span class="insight-pill pill-weakness">${w}</span>`).join('');
    return `
    <div class="insight-card">
      <div class="insight-sp">
        <span class="insight-sp-dot" style="background:${col}"></span>
        ${sp}
      </div>
      <div style="margin-bottom:6px">
        <span style="font-size:11px;text-transform:uppercase;letter-spacing:0.7px;color:var(--muted);">Strengths</span>
      </div>
      <div class="insight-row" style="margin-bottom:10px">${strengths || '<span class="insight-pill pill-strength">No strong signals</span>'}</div>
      <div style="margin-bottom:6px">
        <span style="font-size:11px;text-transform:uppercase;letter-spacing:0.7px;color:var(--muted);">Weaknesses</span>
      </div>
      <div class="insight-row">${weaknesses || '<span class="insight-pill pill-weakness">No major issues</span>'}</div>
    </div>`;
  }).join('');

  return `
  <div class="glass fade-in fade-in-d4">
    <div class="section-label">AI Insights</div>
    <div class="section-title">Strengths & Gaps</div>
    ${cards}
  </div>`;
}

// ── TRANSCRIPT ─────────────────────────────────────────────
function transcriptBlock(data){
  const raw      = data.transcript || '';
  const spColors = {};
  Object.keys(data.metrics).forEach((sp,i) => { spColors[sp] = spColor(i); });

  // Parse flat transcript string into turn objects
  // Handles both inline "SPEAKER 1 0:00:03 text SPEAKER 2 …" and newline-separated formats
  const entries = [];
  const regex   = /(SPEAKER\s+(\d+))\s+([\d:]+)/g;
  const parts   = [];
  let match;

  while ((match = regex.exec(raw)) !== null){
    parts.push({ speaker: match[1], num: match[2], time: match[3], end: regex.lastIndex });
  }

  parts.forEach((p, i) => {
    const textEnd = i + 1 < parts.length ? parts[i+1].end - parts[i+1].speaker.length - parts[i+1].time.length - 2 : raw.length;
    // simpler: just slice from end of this match header to start of next match
    const nextStart = i + 1 < parts.length
      ? raw.indexOf(parts[i+1].speaker, p.end)
      : raw.length;
    const text = raw.slice(p.end, nextStart).trim();
    if(text) entries.push({ speaker: p.speaker, num: p.num, time: p.time, text });
  });

  const rows = entries.map(entry => {
    const col = spColors[entry.speaker] || '#6b7a99';
    return `
    <div class="tlog-row">
      <div class="tlog-avatar" style="background:${col}1a;border:1.5px solid ${col}80;color:${col}">
        ${entry.num}
      </div>
      <div class="tlog-body">
        <div class="tlog-meta">
          <span class="tlog-sp" style="color:${col}">SPEAKER ${entry.num}</span>
          <span class="tlog-dot" style="background:${col}"></span>
          <span class="tlog-time">${entry.time}</span>
        </div>
        <div class="tlog-text">${entry.text}</div>
      </div>
    </div>`;
  }).join('');

  const empty = `<div style="color:var(--muted);font-size:13px;padding:12px 0;">No transcript available.</div>`;

  return `
  <div class="glass fade-in fade-in-d5">
    <div class="section-label">Output</div>
    <div class="section-title">
      <details>
        <summary>Transcript Log</summary>
        <div class="tlog-wrap mt-3">${rows || empty}</div>
      </details>
    </div>
  </div>`;
}

// ── STAT PILL SUMMARY ──────────────────────────────────────
function statPillsSummary(data){
  const metrics  = data.metrics;
  const speakers = Object.keys(metrics);
  const totalTime = speakers.reduce((a,sp) => a + (metrics[sp].speaking_share_percent || 0), 0);
  const totalSeg  = speakers.reduce((a,sp) => a + (metrics[sp].num_turns || 0), 0);
  const topSpScore= (data.final_scores?.ranking || [])[0]?.score || '—';
  const topSpName = (data.final_scores?.ranking || [])[0]?.speaker || '—';

  return `
  <div class="stat-pills-row fade-in">
    <div class="stat-pill">
      <span class="stat-pill-label">Speakers</span>
      <span class="stat-pill-value">${speakers.length}</span>
    </div>
    <div class="stat-pill">
      <span class="stat-pill-label">Total Turns</span>
      <span class="stat-pill-value">${totalSeg}</span>
    </div>
    <div class="stat-pill">
      <span class="stat-pill-label">Top Performer</span>
      <span class="stat-pill-value" style="font-size:14px;font-weight:700">${topSpName}</span>
    </div>
    <div class="stat-pill">
      <span class="stat-pill-label">Top Score</span>
      <span class="stat-pill-value" style="color:var(--accent3)">${topSpScore}</span>
    </div>
  </div>`;
}

// ── LEGEND CHIPS ───────────────────────────────────────────
function legendChips(){
  return `
  <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px;">
    <span style="font-size:11px;display:flex;align-items:center;gap:5px;">
      <span style="width:12px;height:12px;border-radius:3px;border:1px solid var(--best-col);background:rgba(52,211,153,0.1);display:inline-block;"></span>
      <span style="color:var(--muted);">Best</span>
    </span>
    <span style="font-size:11px;display:flex;align-items:center;gap:5px;">
      <span style="width:12px;height:12px;border-radius:3px;border:1px solid var(--worst-col);background:rgba(251,113,133,0.1);display:inline-block;"></span>
      <span style="color:var(--muted);">Worst</span>
    </span>
    <span style="font-size:11px;display:flex;align-items:center;gap:5px;">
      <span style="width:12px;height:12px;border-radius:3px;border:1px solid var(--tie-best-col);background:rgba(103,232,249,0.1);display:inline-block;"></span>
      <span style="color:var(--muted);">Tied Best</span>
    </span>
    <span style="font-size:11px;display:flex;align-items:center;gap:5px;">
      <span style="width:12px;height:12px;border-radius:3px;border:1px solid var(--tie-worst-col);background:rgba(253,230,138,0.1);display:inline-block;"></span>
      <span style="color:var(--muted);">Tied Worst</span>
    </span>
  </div>`;
}

// ── FORM SUBMIT ────────────────────────────────────────────
$('#uploadForm').submit(function(e){
  e.preventDefault();

  $('#resultSection').html(`
    <div class="loader-wrap">
      <div class="spinner-ring"></div>
      <div class="loader-text">Analyzing conversation…</div>
    </div>`);

  $.ajax({
    url: $(this).attr('action') || '/upload',
    type: 'POST',
    data: new FormData(this),
    contentType: false,
    processData: false,
    success: function(data){
      renderResults(data);
    },
    error: function(xhr){
      const msg = xhr.responseJSON?.error || 'An unexpected backend error occurred.';
      $('#resultSection').html(`
        <div style="background:rgba(251,113,133,0.08);border:1px solid rgba(251,113,133,0.25);border-radius:14px;padding:28px;color:#fb7185;">
          <b>Error:</b> ${msg}
        </div>`);
    }
  });
});

function renderResults(data){
  const leftCol = `
    <div>
      ${analysisBlock(data)}
      ${buildChartsSection(data)}
      ${speakerTabsBlock(data)}
      ${transcriptBlock(data)}
    </div>`;

  const rightCol = `
    <div class="right-col">
      ${leaderboardBlock(data)}
      ${insightsBlock(data)}
    </div>`;

  $('#resultSection').html(`
    ${statPillsSummary(data)}
    ${legendChips()}
    <div class="main-layout">
      ${leftCol}
      ${rightCol}
    </div>`);
}