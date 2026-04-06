$(document).ready(function(){
    // Chart instances storage
    let userCharts = {};
    let mainDoughnut = null;
    let interactionRadar = null;

    // Formatting Helpers
    function formatTime(seconds) {
        if (!seconds) return "00:00";
        var date = new Date(0);
        date.setSeconds(seconds);
        return date.toISOString().substring(14, 19);
    }

    const colorPalette = ['#06b6d4', '#8b5cf6', '#10b981', '#f59e0b', '#f43f5e', '#3b82f6'];
    let speakerColors = {};
    let colorIndex = 0;

    function getSpeakerColor(speaker) {
        if (!speakerColors[speaker]) {
            speakerColors[speaker] = colorPalette[colorIndex % colorPalette.length];
            colorIndex++;
        }
        return speakerColors[speaker];
    }

    // Chart Defaults for Dark Mode
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = '#1e293b';

    $('#uploadForm').submit(function(e){
        e.preventDefault();
        var formData = new FormData(this);

        $('#uploadSectionContainer').hide();
        $('#loadingSection').fadeIn();
        $('#resultSection').hide();

        $.ajax({
            url: $(this).attr('action'),
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(data){
                $('#loadingSection').hide();
                $('#resultSection').fadeIn();

                // Clear old charts
                if (mainDoughnut) mainDoughnut.destroy();
                if (interactionRadar) interactionRadar.destroy();
                for (let k in userCharts) userCharts[k].destroy();
                userCharts = {};

                let analysis = data.analysis || {};
                let metrics = data.metrics || {};
                let final_scores = data.final_scores || {ranking: [], scores:{}};
                let explanations = data.explanations || {};

                // 1. Top 4 Cards
                $('#topIntent').text(analysis.intent || "Not identified");
                $('#topSummary').text(analysis.summary ? analysis.summary.substring(0, 80) + "..." : "No summary available");
                
                let impact = analysis.decision_impact || "Low";
                $('#topImpact').text(impact);
                let impactPct = impact.toLowerCase().includes("high") ? 90 : (impact.toLowerCase().includes("medium") ? 50 : 20);
                $('#impactProgressBar').css('width', impactPct + '%');
                
                let actionsRaw = analysis.action_items || "None";
                let actionList = actionsRaw.split(/[,.]/).filter(i => i.trim().length > 3).slice(0, 2);
                let actionHTML = actionList.length > 0 ? actionList.map(a => `<div><i class="fa-solid fa-check text-muted-custom me-1"></i>${a.trim()}</div>`).join('') : "No explicit actions";
                $('#topActions').html(actionHTML);

                // 2. Transcript
                let transcriptHTML = "";
                if (data.segments && data.segments.length > 0) {
                    data.segments.forEach(seg => {
                        let initial = seg.speaker.charAt(0).toUpperCase();
                        let color = getSpeakerColor(seg.speaker);
                        let timeStr = formatTime(seg.start);
                        
                        // Fake sentiment dot logic per segment based on simple word scan (or use speaker average logic)
                        let textLower = seg.text.toLowerCase();
                        let sentColor = '#f59e0b'; // neutral
                        if (textLower.includes('good') || textLower.includes('great') || textLower.includes('agree')) sentColor = '#10b981';
                        if (textLower.includes('bad') || textLower.includes('no') || textLower.includes('issue')) sentColor = '#f43f5e';

                        transcriptHTML += `
                            <div class="chat-bubble">
                                <div class="avatar-ring" style="border-color: ${color};">${initial}</div>
                                <div>
                                    <div class="d-flex align-items-center">
                                        <span class="sentiment-spark" style="background-color: ${sentColor};"></span>
                                        <span style="color: ${color}; font-size: 0.85rem; font-weight: 600;">${seg.speaker}</span>
                                        <span class="timestamp-dim">${timeStr}</span>
                                    </div>
                                    <div class="chat-text">${seg.text}</div>
                                </div>
                            </div>
                        `;
                    });
                }
                $('#transcriptContent').html(transcriptHTML);

                // 3. Right Side: Doughnut & Leaderboard
                let shareLabels = [];
                let shareData = [];
                let bgColors = [];

                final_scores.ranking.forEach((r, idx) => {
                    let spk = r.speaker;
                    let spkMet = metrics[spk] || {};
                    let color = getSpeakerColor(spk);
                    
                    shareLabels.push(spk);
                    shareData.push(spkMet.speaking_share_percent || 0);
                    bgColors.push(color);
                });

                mainDoughnut = new Chart(document.getElementById('speakingShareChart').getContext('2d'), {
                    type: 'doughnut',
                    data: {
                        labels: shareLabels,
                        datasets: [{ data: shareData, backgroundColor: bgColors, borderWidth: 0 }]
                    },
                    options: {
                        responsive: true, maintainAspectRatio: false, cutout: '80%', plugins: { legend: { display: false } }
                    }
                });

                let leaderHTML = "";
                final_scores.ranking.forEach((r, idx) => {
                    let color = getSpeakerColor(r.speaker);
                    leaderHTML += `
                        <div>
                            <div class="d-flex justify-content-between mb-1">
                                <span class="text-main" style="font-size: 0.85rem;"><i class="fa-solid fa-trophy me-2 text-muted-custom"></i>${r.speaker}</span>
                                <span style="font-size: 0.85rem; color: ${color}; font-weight: bold;">${r.score}%</span>
                            </div>
                            <div class="progress progress-dark" style="height: 6px;">
                                <div class="progress-bar" style="width: ${r.score}%; background-color: ${color};"></div>
                            </div>
                        </div>
                    `;
                });
                $('#leaderboardContainer').html(leaderHTML);

                // 4. Speaker Accordions
                let accHTML = "";
                let radarInitData = [];

                final_scores.ranking.forEach((r, idx) => {
                    let spk = r.speaker;
                    let m = metrics[spk] || {};
                    let exp = explanations[spk] || {strengths:[], weaknesses:[]};
                    let color = getSpeakerColor(spk);
                    let initial = spk.charAt(0);

                    // Compute WPM
                    let wpm = (m.avg_duration_per_turn_sec > 0) ? Math.round((m.avg_words_per_turn / m.avg_duration_per_turn_sec) * 60) : 0;
                    
                    // Radar proxies for Sentiment
                    let s_raw = m.sentiment_score || 0; // -1 to 1
                    let p_pos = Math.max(0, s_raw) * 100;
                    let p_neg = Math.max(0, -s_raw) * 100;
                    let p_neu = 100 - (p_pos + p_neg);
                    if (p_neu < 0) p_neu = 0;

                    // Strengths HTML
                    let strHTML = exp.strengths.slice(0,3).map(s => `<div class="sw-item"><i class="fa-solid fa-circle-plus sw-pos"></i>${s}</div>`).join('');
                    let weakHTML = exp.weaknesses.slice(0,3).map(w => `<div class="sw-item"><i class="fa-solid fa-circle-minus sw-neg"></i>${w}</div>`).join('');

                    accHTML += `
                        <div class="speaker-card">
                            <button class="speaker-accordion-btn" data-bs-toggle="collapse" data-bs-target="#spk-${idx}">
                                <div class="d-flex align-items-center">
                                    <div class="avatar-ring" style="border-color:${color}">${spk.split(" ")[1] || initial}</div>
                                    <span style="font-weight: 500;">${spk}</span>
                                </div>
                                <div>
                                    <span class="text-muted-custom me-3" style="font-size: 0.8rem;">Score: ${r.score}</span>
                                    <i class="fa-solid fa-chevron-down text-muted-custom"></i>
                                </div>
                            </button>
                            <div id="spk-${idx}" class="collapse bg-base">
                                <div class="p-3 border-top border-subtle">
                                    
                                    <div class="row g-3 mb-3 text-center">
                                        <div class="col-4"><div class="data-grid-item"><div class="text-muted-custom" style="font-size:0.7rem;">EST WPM</div><div class="fs-5 text-main">${wpm}</div></div></div>
                                        <div class="col-4"><div class="data-grid-item"><div class="text-muted-custom" style="font-size:0.7rem;">VOCAB</div><div class="fs-5 text-main">${m.vocabulary_richness || 0}</div></div></div>
                                        <div class="col-4"><div class="data-grid-item"><div class="text-muted-custom" style="font-size:0.7rem;">COVERAGE</div><div class="fs-5 text-main">${m.topic_coverage_percent || 0}%</div></div></div>
                                    </div>

                                    <div class="row g-3">
                                        <div class="col-12 col-md-4 d-flex flex-column justify-content-center align-items-center">
                                            <div style="height: 120px; width: 100%;"><canvas id="radar-${idx}"></canvas></div>
                                        </div>
                                        <div class="col-12 col-md-8">
                                            <div class="row g-2">
                                                <div class="col-6">
                                                    <div class="text-uppercase text-muted-custom mb-2" style="font-size:0.7rem;">Strengths</div>
                                                    ${strHTML}
                                                </div>
                                                <div class="col-6">
                                                    <div class="text-uppercase text-muted-custom mb-2" style="font-size:0.7rem;">Weaknesses</div>
                                                    ${weakHTML}
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                </div>
                            </div>
                        </div>
                    `;

                    radarInitData.push({
                        idx: idx,
                        data: [Math.round(p_pos), Math.round(p_neu), Math.round(p_neg)],
                        color: color
                    });
                });
                $('#speakerAccordions').html(accHTML);

                // Init Radars
                radarInitData.forEach(ri => {
                    userCharts[`radar_${ri.idx}`] = new Chart(document.getElementById(`radar-${ri.idx}`).getContext('2d'), {
                        type: 'polarArea',
                        data: {
                            labels: ['Positive', 'Neutral', 'Negative'],
                            datasets: [{
                                data: ri.data,
                                backgroundColor: ['rgba(16, 185, 129, 0.4)', 'rgba(245, 158, 11, 0.4)', 'rgba(244, 63, 94, 0.4)'],
                                borderColor: ['#10b981', '#f59e0b', '#f43f5e'],
                                borderWidth: 1
                            }]
                        },
                        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { r: { display: false } } }
                    });
                });

                // 5. Interaction Dynamics Box (Footer)
                // We use global radar as requested, mapping Topic Alignment, Interaction, and Decision for all speakers
                let iLabels = shareLabels;
                let interactScores = [];
                let alignScores = [];
                let totalSenti = 0;

                final_scores.ranking.forEach(r => {
                    let m = metrics[r.speaker] || {};
                    let s = analysis.speaker_scores ? analysis.speaker_scores[r.speaker] : {};
                    let inter = s && s.interaction_score ? s.interaction_score * 10 : (m.speaking_share_percent || 0); 
                    interactScores.push(inter);
                    alignScores.push(m.agenda_alignment_percent || 0);
                    totalSenti += m.sentiment_score || 0;
                });

                interactionRadar = new Chart(document.getElementById('interactionRadarChart').getContext('2d'), {
                    type: 'radar',
                    data: {
                        labels: iLabels,
                        datasets: [
                            { label: 'Interaction Density', data: interactScores, backgroundColor: 'rgba(6, 182, 212, 0.2)', borderColor: '#06b6d4' },
                            { label: 'Topic Alignment', data: alignScores, backgroundColor: 'rgba(139, 92, 246, 0.2)', borderColor: '#8b5cf6' }
                        ]
                    },
                    options: { responsive: true, maintainAspectRatio: false, scales: { r: { angleLines: { color: '#1e293b' }, grid: { color: '#1e293b' } } } }
                });

                // Sliders
                // Conflict vs Agreement. totalSenti ranges from roughly -NumberSpeakers to +NumberSpeakers
                let avgSenti = totalSenti / Math.max(1, shareLabels.length); // -1 to 1
                let agreementPct = ((avgSenti + 1) / 2) * 100; // 0 to 100
                $('#conflictSlider').css('width', agreementPct + '%');
                $('#conflictSlider').css('background-color', agreementPct > 50 ? '#10b981' : '#f43f5e');

                // Task Completion Estimate based on Top Actions array and general metric average
                let avgScore = final_scores.ranking.reduce((sum, r) => sum + r.score, 0) / Math.max(1, shareLabels.length);
                let t_pct = Math.min(100, Math.round(avgScore * 0.8 + (actionList.length * 10)));
                $('#taskCompletionPct').text(t_pct + '%');
                $('#taskCompletionGauge').css('width', t_pct + '%');

            },
            error: function(){
                $('#loadingSection').hide();
                $('#uploadSectionContainer').show();
                alert("Processing Exception: The server encountered an error parsing the architecture.");
            }
        });
    });
});