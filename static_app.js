async function api(path, opts) {
  const res = await fetch(path, opts);
  return res.json();
}

async function refreshList() {
  const data = await api('/api/list');
  const tbody = document.querySelector('#files-table tbody');
  tbody.innerHTML = '';
  data.forEach(f => {
    const tr = document.createElement('tr');
    const score = f.score === null ? '' : f.score.toFixed(3);
    tr.innerHTML = `<td><a href="#" class="select" data-base="${f.base}">${f.base}</a></td>
                    <td>${score}</td>
                    <td>${f.decision}</td>
                    <td>
                      <button class="accept" data-base="${f.base}">Accept</button>
                      <button class="reject" data-base="${f.base}">Reject</button>
                      <button class="requeue" data-base="${f.base}">Requeue</button>
                    </td>`;
    tbody.appendChild(tr);
  });
  attachRowHandlers();
}

function attachRowHandlers() {
  document.querySelectorAll('.select').forEach(a => {
    a.onclick = async (e) => {
      e.preventDefault();
      const base = a.dataset.base;
      showDetails(base);
    };
  });
  document.querySelectorAll('.accept').forEach(b => {
    b.onclick = async () => {
      const base = b.dataset.base;
      await api('/api/action', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({action:'accept', base, note:'accepted via web UI'})});
      refreshList();
    };
  });
  document.querySelectorAll('.reject').forEach(b => {
    b.onclick = async () => {
      const base = b.dataset.base;
      await api('/api/action', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({action:'reject', base, note:'rejected via web UI'})});
      refreshList();
    };
  });
  document.querySelectorAll('.requeue').forEach(b => {
    b.onclick = async () => {
      const base = b.dataset.base;
      await api('/api/action', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({action:'requeue', base, note:'requeued via web UI'})});
      refreshList();
    };
  });
}

async function showDetails(base) {
  const meta = await api(`/api/show/${base}`);
  const panel = document.getElementById('details');
  panel.innerHTML = `<h4>${base}</h4>
    <pre>${JSON.stringify({
      scores: meta.scores,
      decision: meta.decision,
      sermon_selection: meta.sermon_selection
    }, null, 2)}</pre>
    <div>
      <strong>Audio</strong><br/>
      <button id="play-working">Play Working</button>
      <button id="play-sermon">Play Sermon</button>
    </div>
    <div id="audio-player"></div>
    <div style="margin-top:8px;">
      <button id="accept-btn">Accept</button>
      <button id="reject-btn">Reject</button>
      <button id="requeue-btn">Requeue</button>
    </div>
  `;
  document.getElementById('play-working').onclick = () => playAudio(base, 'working');
  document.getElementById('play-sermon').onclick = () => playAudio(base, 'sermon');
  document.getElementById('accept-btn').onclick = async () => { await api('/api/action', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({action:'accept', base, note:'accepted via web UI'})}); refreshList(); };
  document.getElementById('reject-btn').onclick = async () => { await api('/api/action', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({action:'reject', base, note:'rejected via web UI'})}); refreshList(); };
  document.getElementById('requeue-btn').onclick = async () => { await api('/api/action', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({action:'requeue', base, note:'requeued via web UI'})}); refreshList(); };
}

function playAudio(base, kind) {
  const playerDiv = document.getElementById('audio-player');
  playerDiv.innerHTML = `<audio controls src="/api/audio/${base}/${kind}"></audio>`;
}

document.getElementById('refresh').onclick = refreshList;
window.onload = refreshList;
