import express from 'express';
import { spawn, ChildProcess } from 'child_process';
import path from 'path';
import cors from 'cors';
import fs from 'fs';

const app = express();
const port = 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

let currentProcess: ChildProcess | null = null;
const BINARY_PATH = path.resolve(__dirname, '../../build/llm');
const DATASETS_DIR = path.resolve(__dirname, '../../.datasets');
const MODELS_DIR = path.resolve(__dirname, '../../.models');

// Helper to broadcast logs to any connected SSE clients
let logStream: express.Response | null = null;

// Recursive helper to find files by extension
function getFiles(dir: string, extension: string): string[] {
    let results: string[] = [];
    if (!fs.existsSync(dir)) return results;
    
    const list = fs.readdirSync(dir);
    list.forEach(file => {
        const fullPath = path.join(dir, file);
        const stat = fs.statSync(fullPath);
        if (stat && stat.isDirectory()) {
            results = results.concat(getFiles(fullPath, extension));
        } else {
            if (file.endsWith(extension)) {
                // Return path relative to the project root
                results.push(path.relative(path.resolve(__dirname, '../../'), fullPath));
            }
        }
    });
    return results;
}

app.get('/api/datasets', (req, res) => {
    try {
        const datasets = getFiles(DATASETS_DIR, '.rows');
        res.json({ datasets });
    } catch (err) {
        res.status(500).json({ error: 'Failed to list datasets' });
    }
});

app.get('/api/tokenizers', (req, res) => {
    try {
        const tokenizers = getFiles(MODELS_DIR, '.tok');
        res.json({ tokenizers });
    } catch (err) {
        res.status(500).json({ error: 'Failed to list tokenizers' });
    }
});

app.get('/api/models', (req, res) => {
    try {
        const models = getFiles(MODELS_DIR, '.lm');
        res.json({ models });
    } catch (err) {
        res.status(500).json({ error: 'Failed to list models' });
    }
});

app.post('/api/run', (req, res) => {
    if (currentProcess) {
        return res.status(400).json({ error: 'A process is already running.' });
    }

    const { command, args } = req.body;
    
    // Safety check: only allow known commands
    const allowedCommands = ['train', 'train-tokenizer', 'predict', 'init-model'];
    if (!allowedCommands.includes(command)) {
        return res.status(400).json({ error: 'Invalid command.' });
    }

    const fullArgs = [command, ...args];
    console.log(`Executing: ${BINARY_PATH} ${fullArgs.join(' ')}`);

    currentProcess = spawn(BINARY_PATH, fullArgs, {
        cwd: path.resolve(__dirname, '../../'),
        env: { ...process.env, PATH: process.env.PATH }
    });

    currentProcess.stdout?.on('data', (data) => {
        const message = data.toString();
        if (logStream) {
            logStream.write(`data: ${JSON.stringify({ type: 'stdout', message })}\n\n`);
        }
    });

    currentProcess.stderr?.on('data', (data) => {
        const message = data.toString();
        if (logStream) {
            logStream.write(`data: ${JSON.stringify({ type: 'stderr', message })}\n\n`);
        }
    });

    currentProcess.on('close', (code) => {
        if (logStream) {
            logStream.write(`data: ${JSON.stringify({ type: 'exit', code })}\n\n`);
        }
        currentProcess = null;
    });

    res.json({ status: 'started' });
});

app.post('/api/stop', (req, res) => {
    if (currentProcess) {
        currentProcess.kill();
        currentProcess = null;
        return res.json({ status: 'killed' });
    }
    res.status(400).json({ error: 'No process running.' });
});

// SSE endpoint for real-time logs
app.get('/api/events', (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    logStream = res;

    req.on('close', () => {
        logStream = null;
    });
});

app.listen(port, () => {
    console.log(`LLM Web Wrapper listening at http://localhost:${port}`);
});
