const runBtn = document.getElementById('run-btn');
const stopBtn = document.getElementById('stop-btn');
const consoleDiv = document.getElementById('console');
const statusBar = document.getElementById('status');
const commandSelect = document.getElementById('command-select');

let eventSource = null;

// Connect to SSE
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
            appendLog(`\nProcess exited with code ${data.code}\n`);
            setRunning(false);
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
        statusBar.textContent = 'Status: Running...';
        statusBar.className = 'status-bar status-running';
    } else {
        runBtn.disabled = false;
        stopBtn.style.display = 'none';
        statusBar.textContent = 'Status: Idle';
        statusBar.className = 'status-bar status-idle';
    }
}

runBtn.addEventListener('click', async () => {
    const cmd = commandSelect.value;
    let args = [];

    if (cmd === 'predict') {
        args = [
            '--prompt', document.getElementById('predict-prompt').value,
            '--model', document.getElementById('predict-model').value,
            '--tokenizer', document.getElementById('predict-tokenizer').value
        ];
    } else if (cmd === 'train') {
        // You can add more inputs to the HTML and harvest them here
        args = ['--data', 'data.txt', '--tokenizer', 'tokenizer.bin', '--output-model', 'model.bin'];
    }

    consoleDiv.textContent = `> Dispatching: ${cmd} ${args.join(' ')}\n`;
    setRunning(true);
    connectSSE();

    try {
        const response = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: cmd, args })
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
connectSSE();

