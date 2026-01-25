import React, { useState, useEffect, useRef } from 'react';
import EditorManager from '../../core/EditorManager';
import './Console.css';

/**
 * Log entry interface
 */
interface LogEntry {
    id: string;
    timestamp: number;
    type: 'info' | 'warn' | 'error' | 'debug';
    source: string;
    message: string;
    data?: any;
}

/**
 * Performance stats interface
 */
interface PerformanceStats {
    fps: number;
    frameTime: number;
    memory: number;
    drawCalls: number;
    triangles: number;
    splatCount: number;
}

/**
 * Console Panel - Log viewer and profiler
 *
 * Features:
 * - Log viewer with filtering (info, warn, error)
 * - Performance stats (FPS, memory, draw calls)
 * - Splat count display
 * - Network requests log
 * - Manager status indicators
 */
interface ConsoleProps {
    editorManager: EditorManager;
}

const Console: React.FC<ConsoleProps> = ({ editorManager }) => {
    // Logs
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [filter, setFilter] = useState<'all' | 'info' | 'warn' | 'error' | 'debug'>('all');
    const [searchQuery, setSearchQuery] = useState('');

    // Performance stats
    const [stats, setStats] = useState<PerformanceStats>({
        fps: 60,
        frameTime: 16.7,
        memory: 0,
        drawCalls: 0,
        triangles: 0,
        splatCount: 0
    });

    // UI state
    const [showProfiler, setShowProfiler] = useState(true);
    const [showNetwork, setShowNetwork] = useState(false);
    const [autoScroll, setAutoScroll] = useState(true);

    // Refs
    const logContainerRef = useRef<HTMLDivElement>(null);
    const performanceRef = useRef({
        frameCount: 0,
        lastTime: performance.now(),
        fpsUpdateTime: 0,
        framesSinceUpdate: 0
    });

    /**
     * Add a log entry
     */
    const addLog = (entry: Omit<LogEntry, 'id' | 'timestamp'>) => {
        const logEntry: LogEntry = {
            id: `log-${Date.now()}-${Math.random()}`,
            timestamp: Date.now(),
            ...entry
        };

        setLogs(prev => {
            const newLogs = [...prev, logEntry];
            // Keep only last 1000 logs
            if (newLogs.length > 1000) {
                return newLogs.slice(-1000);
            }
            return newLogs;
        });
    };

    /**
     * Clear all logs
     */
    const clearLogs = () => {
        setLogs([]);
    };

    /**
     * Format timestamp
     */
    const formatTime = (timestamp: number): string => {
        const date = new Date(timestamp);
        const hours = date.getHours().toString().padStart(2, '0');
        const mins = date.getMinutes().toString().padStart(2, '0');
        const secs = date.getSeconds().toString().padStart(2, '0');
        const ms = date.getMilliseconds().toString().padStart(3, '0');
        return `${hours}:${mins}:${secs}.${ms}`;
    };

    /**
     * Filter logs
     */
    const filteredLogs = logs.filter(log => {
        if (filter !== 'all' && log.type !== filter) return false;
        if (searchQuery && !log.message.toLowerCase().includes(searchQuery.toLowerCase())) return false;
        return true;
    });

    /**
     * Update performance stats
     */
    useEffect(() => {
        let animationFrameId: number;

        const updateStats = () => {
            const now = performance.now();
            const perf = performanceRef.current;

            perf.frameCount++;
            perf.framesSinceUpdate++;

            // Update FPS every 500ms
            if (now - perf.fpsUpdateTime >= 500) {
                const delta = now - perf.lastTime;
                const fps = Math.round((perf.framesSinceUpdate * 1000) / delta);

                const info = (performance as any).memory;
                const memory = info ? Math.round(info.usedJSHeapSize / 1024 / 1024) : 0;

                // Get renderer info
                const renderer = editorManager.renderer;
                const drawCalls = renderer?.info.render.calls || 0;
                const triangles = renderer?.info.render.triangles || 0;

                // Count splat objects
                let splatCount = 0;
                editorManager.scene?.traverse((obj: any) => {
                    if (obj.constructor?.name === 'SplatMesh' || obj.type?.includes('Splat')) {
                        splatCount++;
                    }
                });

                setStats({
                    fps,
                    frameTime: Math.round(1000 / fps * 10) / 10,
                    memory,
                    drawCalls,
                    triangles,
                    splatCount
                });

                perf.framesSinceUpdate = 0;
                perf.lastTime = now;
                perf.fpsUpdateTime = now;
            }

            animationFrameId = requestAnimationFrame(updateStats);
        };

        animationFrameId = requestAnimationFrame(updateStats);

        return () => cancelAnimationFrame(animationFrameId);
    }, [editorManager]);

    /**
     * Auto-scroll to bottom
     */
    useEffect(() => {
        if (autoScroll && logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [logs, autoScroll]);

    /**
     * Intercept console methods
     */
    useEffect(() => {
        const originalLog = console.log;
        const originalWarn = console.warn;
        const originalError = console.error;
        const originalDebug = console.debug;

        console.log = (...args: any[]) => {
            originalLog(...args);
            addLog({
                type: 'info',
                source: 'console',
                message: args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : String(arg)).join(' ')
            });
        };

        console.warn = (...args: any[]) => {
            originalWarn(...args);
            addLog({
                type: 'warn',
                source: 'console',
                message: args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : String(arg)).join(' ')
            });
        };

        console.error = (...args: any[]) => {
            originalError(...args);
            addLog({
                type: 'error',
                source: 'console',
                message: args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : String(arg)).join(' ')
            });
        };

        console.debug = (...args: any[]) => {
            originalDebug(...args);
            addLog({
                type: 'debug',
                source: 'console',
                message: args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : String(arg)).join(' ')
            });
        };

        return () => {
            console.log = originalLog;
            console.warn = originalWarn;
            console.error = originalError;
            console.debug = originalDebug;
        };
    }, []);

    /**
     * Render log icon
     */
    const getLogIcon = (type: LogEntry['type']): string => {
        switch (type) {
            case 'info': return 'ℹ️';
            case 'warn': return '⚠️';
            case 'error': return '❌';
            case 'debug': return '🔍';
            default: return '•';
        }
    };

    return (
        <div className="console-container">
            {/* Console header */}
            <div className="console-header">
                <div className="console-title">
                    <h3>Console</h3>
                    <div className="console-tabs">
                        <button
                            className={`tab ${!showProfiler && !showNetwork ? 'active' : ''}`}
                            onClick={() => { setShowProfiler(false); setShowNetwork(false); }}
                        >
                            Logs
                        </button>
                        <button
                            className={`tab ${showProfiler ? 'active' : ''}`}
                            onClick={() => { setShowProfiler(true); setShowNetwork(false); }}
                        >
                            Profiler
                        </button>
                        <button
                            className={`tab ${showNetwork ? 'active' : ''}`}
                            onClick={() => { setShowProfiler(false); setShowNetwork(true); }}
                        >
                            Network
                        </button>
                    </div>
                </div>

                <div className="console-actions">
                    <button
                        className="clear-button"
                        onClick={clearLogs}
                        title="Clear logs"
                    >
                        Clear
                    </button>
                </div>
            </div>

            {/* Filter bar */}
            {!showProfiler && !showNetwork && (
                <div className="console-filter">
                    <div className="filter-buttons">
                        <button
                            className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
                            onClick={() => setFilter('all')}
                        >
                            All
                        </button>
                        <button
                            className={`filter-btn ${filter === 'info' ? 'active' : ''}`}
                            onClick={() => setFilter('info')}
                        >
                            Info
                        </button>
                        <button
                            className={`filter-btn ${filter === 'warn' ? 'active' : ''}`}
                            onClick={() => setFilter('warn')}
                        >
                            Warn
                        </button>
                        <button
                            className={`filter-btn ${filter === 'error' ? 'active' : ''}`}
                            onClick={() => setFilter('error')}
                        >
                            Error
                        </button>
                        <button
                            className={`filter-btn ${filter === 'debug' ? 'active' : ''}`}
                            onClick={() => setFilter('debug')}
                        >
                            Debug
                        </button>
                    </div>

                    <input
                        type="text"
                        placeholder="Filter..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="filter-search"
                    />

                    <label className="auto-scroll-label">
                        <input
                            type="checkbox"
                            checked={autoScroll}
                            onChange={(e) => setAutoScroll(e.target.checked)}
                        />
                        Auto-scroll
                    </label>
                </div>
            )}

            {/* Profiler panel */}
            {showProfiler && (
                <div className="profiler-panel">
                    <div className="profiler-grid">
                        <div className="stat-card">
                            <div className="stat-label">FPS</div>
                            <div className={`stat-value ${stats.fps >= 50 ? 'good' : stats.fps >= 30 ? 'warning' : 'bad'}`}>
                                {stats.fps}
                            </div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-label">Frame Time</div>
                            <div className="stat-value">{stats.frameTime}ms</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-label">Memory</div>
                            <div className="stat-value">{stats.memory}MB</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-label">Draw Calls</div>
                            <div className="stat-value">{stats.drawCalls}</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-label">Triangles</div>
                            <div className="stat-value">{stats.triangles.toLocaleString()}</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-label">Splat Objects</div>
                            <div className="stat-value">{stats.splatCount}</div>
                        </div>
                    </div>

                    {/* Manager status */}
                    <div className="manager-status">
                        <h4>Manager Status</h4>
                        <div className="status-list">
                            <div className="status-item">
                                <span className="status-label">EditorManager</span>
                                <span className={`status-indicator ${editorManager.isInitialized ? 'active' : 'inactive'}`}>
                                    {editorManager.isInitialized ? 'Active' : 'Inactive'}
                                </span>
                            </div>
                            <div className="status-item">
                                <span className="status-label">Render Loop</span>
                                <span className={`status-indicator ${editorManager.renderer ? 'active' : 'inactive'}`}>
                                    {editorManager.renderer ? 'Running' : 'Stopped'}
                                </span>
                            </div>
                            <div className="status-item">
                                <span className="status-label">Play Mode</span>
                                <span className={`status-indicator ${editorManager.isPlaying ? 'active' : 'inactive'}`}>
                                    {editorManager.isPlaying ? 'Playing' : 'Editing'}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Network panel */}
            {showNetwork && (
                <div className="network-panel">
                    <div className="network-empty">
                        <p>Network requests log</p>
                        <p className="hint">Coming soon...</p>
                    </div>
                </div>
            )}

            {/* Log entries */}
            {!showProfiler && !showNetwork && (
                <div className="console-logs" ref={logContainerRef}>
                    {filteredLogs.length === 0 ? (
                        <div className="console-empty">
                            <p>No logs to display</p>
                            <p className="hint">Logs will appear here</p>
                        </div>
                    ) : (
                        filteredLogs.map(log => (
                            <div
                                key={log.id}
                                className={`log-entry log-${log.type}`}
                            >
                                <span className="log-icon">{getLogIcon(log.type)}</span>
                                <span className="log-time">{formatTime(log.timestamp)}</span>
                                <span className="log-source">[{log.source}]</span>
                                <span className="log-message">{log.message}</span>
                            </div>
                        ))
                    )}
                </div>
            )}

            {/* Console footer */}
            <div className="console-footer">
                <span className="log-count">{filteredLogs.length} entries</span>
                {filteredLogs.length !== logs.length && (
                    <span className="log-filtered">
                        ({filteredLogs.length} of {logs.length} shown)
                    </span>
                )}
            </div>
        </div>
    );
};

export default Console;
