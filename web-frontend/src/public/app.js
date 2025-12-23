const runBtn = document.getElementById('run-btn');
const stopBtn = document.getElementById('stop-btn');
const consoleDiv = document.getElementById('console');
const statusText = document.getElementById('status-text');
const statusContainer = document.getElementById('status-container');
const navItems = document.querySelectorAll('.nav-item');
const views = document.querySelectorAll('.view-section');

const datasetDropdowns = document.querySelectorAll('.dataset-dropdown');
const modelDropdowns = document.querySelectorAll('.model-dropdown');
const tokenizerDropdowns = document.querySelectorAll('.tokenizer-dropdown');

let activeView = 'predict';
let eventSource = null;

// --- Navigation Logic ---
navItems.forEach(item => {
    item.addEventListener('click', () => {
        // Update Nav
        navItems.forEach(i => i.classList.remove('active'));
        item.classList.add('active');

        // Update View
        const viewId = item.getAttribute('data-view');
        activeView = viewId;
        
        views.forEach(v => v.classList.remove('active'));
        document.getElementById(`view-${viewId}`).classList.add('active');
    });
});

// --- Resource Loading ---
async function fetchAndPopulate(url, dropdowns, placeholder) {
    try {
        const response = await fetch(url);
        const data = await response.json();
        const keys = Object.keys(data);
        const list = data[keys[0]]; // e.g. data.datasets or data.models

        dropdowns.forEach(dropdown => {
            dropdown.innerHTML = '';
            if (list.length === 0) {
                const opt = document.createElement('option');
                opt.value = "";
                opt.textContent = `No resources found`;
                dropdown.appendChild(opt);
            } else {
                list.forEach(path => {
                    const opt = document.createElement('option');
                    opt.value = path;
                    opt.textContent = path;
                    dropdown.appendChild(opt);
                });
            }
        });
    } catch (err) {
        console.error(`Failed to load ${url}:`, err);
    }
}

async function loadAllResources() {
    await Promise.all([
        fetchAndPopulate('/api/datasets', datasetDropdowns, 'datasets'),
        fetchAndPopulate('/api/models', modelDropdowns, 'models'),
        fetchAndPopulate('/api/tokenizers', tokenizerDropdowns, 'tokenizers')
    ]);
}

// --- SSE & Process Logic ---
function connectSSE() {
    if (eventSource) eventSource.close();
    
    eventSource = new EventSource('/api/events');
    
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'stdout' || data.type === 'stderr') {
            const span = document.createElement('span');
            span.className = data.type;
            span.textContent = data.message;
            consoleDiv.appendChild(span);
            consoleDiv.scrollTop = consoleDiv.scrollHeight;
        } else if (data.type === 'exit') {
            appendLog(`
[Process exited with code ${data.code}]
`);
            setRunning(false);
            // Refresh lists in case new files were created
            loadAllResources();
        }
    };

    eventSource.onerror = (err) => {
        console.error("SSE Error:", err);
        eventSource.close();
    };
}

function appendLog(text) {
    const span = document.createElement('span');
    span.textContent = text;
    consoleDiv.appendChild(span);
    consoleDiv.scrollTop = consoleDiv.scrollHeight;
}

function setRunning(running) {
    if (running) {
        runBtn.disabled = true;
        stopBtn.style.display = 'inline-block';
        statusText.textContent = `Status: Running (${activeView})...`;
        statusContainer.className = 'status-container status-running';
    } else {
        runBtn.disabled = false;
        stopBtn.style.display = 'none';
        statusText.textContent = 'Status: Idle';
        statusContainer.className = 'status-container status-idle';
    }
}

runBtn.addEventListener('click', async () => {
    let args = [];
    let command = activeView;

    if (activeView === 'predict') {
        args = [
            '--prompt',
            document.getElementById('predict-prompt').value,
            '--model',
            document.getElementById('predict-model-select').value,
            '--tokenizer',
            document.getElementById('predict-tokenizer-select').value,
            '--length',
            document.getElementById('predict-length').value
        ];
    } else if (activeView === 'train') {
        args = [
            '--data',
            document.getElementById('train-data-select').value,
            '--input-model',
            document.getElementById('train-model-select').value,
            '--tokenizer',
            document.getElementById('train-tokenizer-select').value,
            '--output-model',
            document.getElementById('train-data-select').value,
            '--dataset-type',
            'row-based'
        ];
        
        var row_limit = document.getElementById('row-limit').value;
        
        if (row_limit && parseInt(row_limit) > 0) {
          args = [...args, '-n', row_limit];
        }
    } else if (activeView === 'train-tokenizer') {
        args = [
            '--corpus',
            document.getElementById('tok-corpus-select').value,
            '--output',
            document.getElementById('tok-output').value,
            '--vocab-size',
            document.getElementById('tok-vocab').value,
            '--dataset-type',
            'row-based'
        ];
    } else if (activeView === 'init-model') {
        args = [
            '--output',
            document.getElementById('init-output').value,
            '--tokenizer',
            document.getElementById('init-tokenizer-select').value,
            '--dimensions',
            document.getElementById('init-dim').value,
            '--heads',
            document.getElementById('init-heads').value,
            '--layers',
            document.getElementById('init-layers').value
        ];
    }

    consoleDiv.textContent = `> Dispatching: ${command} ${args.join(' ')}

`;
    setRunning(true);
    connectSSE();

    try {
        const response = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command, args })
        });
        
        const result = await response.json();
        if (result.error) {
            appendLog(`Error: ${result.error}\n`);
            setRunning(false);
        }
    } catch (err) {
        appendLog(`Fetch Error: ${err.message}\n`);
        setRunning(false);
    }
});

stopBtn.addEventListener('click', async () => {
    await fetch('/api/stop', { method: 'POST' });
});

// Initialize
loadAllResources();
connectSSE();
