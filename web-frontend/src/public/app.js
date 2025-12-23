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

const chartContainer = document.getElementById('chart-container');
const ctx = document.getElementById('training-chart').getContext('2d');
let trainingChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Rolling Avg Loss',
            data: [],
            borderColor: '#8b5cf6',
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            borderWidth: 2,
            tension: 0.4,
            fill: true,
            pointRadius: 0,
            yAxisID: 'y'
        }, {
            label: 'Accuracy (%)',
            data: [],
            borderColor: '#22c55e',
            backgroundColor: 'transparent',
            borderWidth: 2,
            tension: 0.4,
            fill: false,
            pointRadius: 0,
            yAxisID: 'y1'
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: { color: '#f8fafc' }
            }
        },
        scales: {
            y: {
                type: 'linear',
                display: true,
                position: 'left',
                beginAtZero: true,
                suggestedMax: 25,
                grid: { color: '#334155' },
                ticks: { color: '#94a3b8' },
                title: { display: true, text: 'Loss', color: '#94a3b8' }
            },
            y1: {
                type: 'linear',
                display: true,
                position: 'right',
                beginAtZero: true,
                suggestedMax: 5,
                grid: { drawOnChartArea: false },
                ticks: { color: '#22c55e' },
                title: { display: true, text: 'Accuracy %', color: '#22c55e' }
            },
            x: {
                grid: { color: '#334155' },
                ticks: { color: '#94a3b8', maxTicksLimit: 10 }
            }
        }
    }
});

let activeView = 'predict';
let runningCommand = null;
let eventSource = null;
let stdoutBuffer = "";

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

        // Show chart if we are in training view
        if (activeView === 'train') {
            if (trainingChart.data.labels.length > 0 || runningCommand === 'train') {
                chartContainer.style.display = 'block';
            }
        } else {
            // When not in train view, we don't show the chart because it's now inside the train view DOM
            chartContainer.style.display = 'none';
        }
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

            // Parse training progress
            if (runningCommand === 'train' && data.type === 'stdout') {
                stdoutBuffer += data.message;
                
                // Process complete lines
                if (stdoutBuffer.includes('\n')) {
                    const lines = stdoutBuffer.split('\n');
                    // Keep the last partial line in the buffer
                    stdoutBuffer = lines.pop();

                    lines.forEach(line => {
                        const cleanLine = line.replace(/\u001b\[[0-9;]*m/g, '');
                        const match = cleanLine.match(/Row +(\d+) \/ \d+.*Loss: *([\d.]+).*Accuracy:\s*([\d.]+)%/i);
                        
                        if (match) {
                            const row = parseInt(match[1]);
                            const loss = parseFloat(match[2]);
                            const accuracy = parseFloat(match[3]);
                            
                            trainingChart.data.labels.push(row);
                            trainingChart.data.datasets[0].data.push(loss);
                            trainingChart.data.datasets[1].data.push(accuracy);
                            
                            // Keep the last 200 points for performance
                            if (trainingChart.data.labels.length > 200) {
                                trainingChart.data.labels.shift();
                                trainingChart.data.datasets[0].data.shift();
                                trainingChart.data.datasets[1].data.shift();
                            }
                        }
                    });
                    trainingChart.update('none');
                }
            }
        } else if (data.type === 'exit') {
            stdoutBuffer = ""; // Clear buffer on exit
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

function setRunning(running, command = null) {
    if (running) {
        runningCommand = command;
        runBtn.disabled = true;
        stopBtn.style.display = 'inline-block';
        statusText.textContent = `Status: Running (${command})...`;
        statusContainer.className = 'status-container status-running';
    } else {
        runningCommand = null;
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

    consoleDiv.textContent = `> Dispatching: ${command} ${args.join(' ')}\n\n`;
    stdoutBuffer = "";
    
    if (command === 'train') {
        chartContainer.style.display = 'block';
        trainingChart.data.labels = [];
        trainingChart.data.datasets[0].data = [];
        trainingChart.data.datasets[1].data = [];
        trainingChart.update();
    } else {
        chartContainer.style.display = 'none';
    }

    setRunning(true, command);
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
