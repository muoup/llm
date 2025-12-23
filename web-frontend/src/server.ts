import express from 'express';
import { spawn, ChildProcess } from 'child_process';
import path from 'path';
import cors from 'cors';

const app = express();
const port = 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

let currentProcess: ChildProcess | null = null;
const BINARY_PATH = path.resolve(__dirname, '../../build/llm');

// Helper to broadcast logs to any connected SSE clients
let logStream: express.Response | null = null;

app.post('/api/run', (req, res) => {
    if (currentProcess) {
        return res.status(400).json({ error: 'A process is already running.' });
    }

    const { command, args } = req.body;
    
    // Safety check: only allow known commands
    const allowedCommands = ['train', 'train-tokenizer', 'predict'];
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
