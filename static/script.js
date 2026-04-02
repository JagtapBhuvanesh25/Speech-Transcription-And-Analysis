$(document).ready(function(){

    $('#uploadForm').submit(function(e){
        e.preventDefault();

        var formData = new FormData(this);

        $('#resultSection').html("<div class='text-center'>Processing...</div>");

        $.ajax({
            url: $(this).attr('action'),
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,

            success: function(data){

                // TRANSCRIPT
                let transcriptHTML = `
                    <div class="card p-3 mb-3">
                        <div class="section-title">📝 Transcript</div>
                        <div>${data.transcript.replace(/\n/g, "<br>")}</div>
                    </div>
                `;

                // ANALYSIS
                let analysis = data.analysis || {};

                let analysisHTML = `
                    <div class="card p-3 mb-3">
                        <div class="section-title">🧠 Analysis</div>
                        <div><b>Intent:</b> ${analysis.intent || "-"}</div>
                        <div><b>Summary:</b> ${analysis.summary || "-"}</div>
                        <div><b>Decision Impact:</b> ${analysis.decision_impact || "-"}</div>
                        <div><b>Action Items:</b> ${analysis.action_items || "-"}</div>
                    </div>
                `;

                // FINAL RANKING
                let final = data.final_scores || {};
                let ranking = final.ranking || [];

                let finalHTML = `
                    <div class="card p-3 mb-3">
                        <div class="section-title">🏆 Final Speaker Ranking</div>
                `;

                ranking.forEach(r => {
                    finalHTML += `
                        <div class="metric-box d-flex justify-content-between">
                            <div><b>#${r.rank}</b> ${r.speaker}</div>
                            <div><b>${r.score}%</b></div>
                        </div>
                    `;
                });

                finalHTML += `</div>`;

                // SPEAKER METRICS
                let scores = analysis.speaker_scores || {};
                let metricsHTML = `<div class="section-title">📊 Speaker Metrics</div>`;

                for (let speaker in data.metrics) {
                    let m = data.metrics[speaker];
                    let s = scores[speaker] || {};

                    metricsHTML += `
                        <div class="card p-3 speaker-card">
                            <h5>${speaker}</h5>

                            <div class="metric-box">Speaking Share: ${m.speaking_share_percent}%</div>
                            <div class="metric-box">Turns: ${m.num_turns}</div>
                            <div class="metric-box">Avg Words/Turn: ${m.avg_words_per_turn}</div>
                            <div class="metric-box">Avg Duration/Turn: ${m.avg_duration_per_turn_sec}s</div>
                            <div class="metric-box">Questions Asked: ${m.questions_asked}</div>
                            <div class="metric-box">Vocabulary Richness: ${m.vocabulary_richness}</div>
                            <div class="metric-box">Filler Rate: ${m.filler_rate}</div>
                            <div class="metric-box">Agenda Alignment: ${m.agenda_alignment_percent}%</div>
                            <div class="metric-box">Sentiment Score: ${m.sentiment_score}</div>
                            <div class="metric-box">Topic Coverage: ${m.topic_coverage_percent}%</div>
                            <div class="metric-box">Confidence Score: ${m.confidence_score}</div>

                            <hr>

                            <div class="metric-box"><b>Contribution Quality:</b> ${s.contribution_quality ?? "-"}</div>
                            <div class="metric-box"><b>Interaction Score:</b> ${s.interaction_score ?? "-"}</div>
                            <div class="metric-box"><b>Decision Impact:</b> ${s.decision_impact ?? "-"}</div>
                        </div>
                    `;
                }

                // EXPLANATIONS
                let explanations = data.explanations || {};

                let explainHTML = `
                    <div class="card p-3 mb-3">
                        <div class="section-title">🧾 Score Explanation</div>
                `;

                for (let speaker in explanations) {
                    let e = explanations[speaker];

                    explainHTML += `
                        <div class="mb-3">
                            <h6>${speaker}</h6>

                            <div><b>Strengths:</b></div>
                            <ul>
                                ${e.strengths.map(s => `<li>${s}</li>`).join("")}
                            </ul>

                            <div><b>Weaknesses:</b></div>
                            <ul>
                                ${e.weaknesses.map(w => `<li>${w}</li>`).join("")}
                            </ul>
                        </div>
                    `;
                }

                explainHTML += `</div>`;

                // FINAL RENDER
                $('#resultSection').html(
                    transcriptHTML + analysisHTML + finalHTML + metricsHTML + explainHTML
                );
            },

            error: function(){
                $('#resultSection').html("<div class='alert alert-danger'>Error processing file</div>");
            }
        });
    });

});