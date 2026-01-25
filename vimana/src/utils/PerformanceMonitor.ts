/**
 * PerformanceMonitor - FPS and performance metrics tracking
 *
 * Monitors frame rate, frame time, and memory usage.
 * Provides recommendations for quality adjustments.
 */

import { DeviceTier } from './DeviceCapabilities';

export interface PerformanceMetrics {
    fps: number;
    frameTime: number;
    minFps: number;
    maxFps: number;
    avgFps: number;
    memory: number;
    memoryTrend: 'stable' | 'increasing' | 'decreasing';
}

export interface PerformanceConfig {
    /** Sample size for averaging (frames) */
    sampleSize: number;
    /** Memory check interval (ms) */
    memoryCheckInterval: number;
    /** Target FPS threshold */
    targetFps: number;
    /** Minimum acceptable FPS */
    minAcceptableFps: number;
}

export class PerformanceMonitor {
    private config: PerformanceConfig;
    private frameTimes: number[] = [];
    private fpsValues: number[] = [];
    private lastFrameTime: number = performance.now();
    private frameCount: number = 0;

    // Memory tracking
    private lastMemoryCheck: number = 0;
    private memoryHistory: number[] = [];
    private readonly maxMemoryHistory = 10;

    // FPS tracking
    private minFps: number = Infinity;
    private maxFps: number = 0;

    // Performance recommendation
    private recommendedTier: DeviceTier | null = null;

    // Callbacks
    private onIssueCallback?: () => void;

    constructor(config: Partial<PerformanceConfig> = {}) {
        this.config = {
            sampleSize: 60,
            memoryCheckInterval: 1000,
            targetFps: 60,
            minAcceptableFps: 30,
            ...config
        };
    }

    /**
     * Update monitor - call once per frame
     */
    public update(): PerformanceMetrics {
        const now = performance.now();
        const frameTime = now - this.lastFrameTime;
        this.lastFrameTime = now;

        // Calculate FPS
        const fps = frameTime > 0 ? 1000 / frameTime : 60;

        // Track frame times and FPS
        this.frameTimes.push(frameTime);
        this.fpsValues.push(fps);

        // Keep only recent samples
        if (this.frameTimes.length > this.config.sampleSize) {
            this.frameTimes.shift();
            this.fpsValues.shift();
        }

        // Track min/max FPS
        if (isFinite(fps)) {
            this.minFps = Math.min(this.minFps, fps);
            this.maxFps = Math.max(this.maxFps, fps);
        }

        // Increment frame counter
        this.frameCount++;

        // Check memory periodically
        if (now - this.lastMemoryCheck > this.config.memoryCheckInterval) {
            this.checkMemory();
            this.lastMemoryCheck = now;
        }

        // Check for performance issues
        if (this.shouldTriggerPerformanceWarning()) {
            if (this.onIssueCallback) {
                this.onIssueCallback();
            }
        }

        return this.getMetrics();
    }

    /**
     * Check current memory usage
     */
    private checkMemory(): void {
        // @ts-ignore - deviceMemory is not in all TypeScript definitions
        const memory = navigator.deviceMemory || 0;
        if (memory > 0) {
            this.memoryHistory.push(memory);
            if (this.memoryHistory.length > this.maxMemoryHistory) {
                this.memoryHistory.shift();
            }
        }
    }

    /**
     * Get memory trend
     */
    private getMemoryTrend(): 'stable' | 'increasing' | 'decreasing' {
        if (this.memoryHistory.length < 3) return 'stable';

        const recent = this.memoryHistory.slice(-3);
        const trend = recent[2] - recent[0];

        if (Math.abs(trend) < 0.1) return 'stable';
        return trend > 0 ? 'increasing' : 'decreasing';
    }

    /**
     * Check if performance warning should be triggered
     */
    private shouldTriggerPerformanceWarning(): boolean {
        if (this.fpsValues.length < this.config.sampleSize) return false;

        const avgFps = this.getAverageFps();
        return avgFps < this.config.minAcceptableFps;
    }

    /**
     * Get current performance metrics
     */
    public getMetrics(): PerformanceMetrics {
        const avgFps = this.getAverageFps();
        const avgFrameTime = this.getAverageFrameTime();

        // @ts-ignore
        const memory = navigator.deviceMemory || 0;

        return {
            fps: this.fpsValues[this.fpsValues.length - 1] || 60,
            frameTime: avgFrameTime,
            minFps: this.minFps === Infinity ? 60 : this.minFps,
            maxFps: this.maxFps,
            avgFps,
            memory,
            memoryTrend: this.getMemoryTrend()
        };
    }

    /**
     * Get average FPS over sample window
     */
    private getAverageFps(): number {
        if (this.fpsValues.length === 0) return 60;

        const sum = this.fpsValues.reduce((a, b) => a + b, 0);
        return sum / this.fpsValues.length;
    }

    /**
     * Get average frame time
     */
    private getAverageFrameTime(): number {
        if (this.frameTimes.length === 0) return 16.67;

        const sum = this.frameTimes.reduce((a, b) => a + b, 0);
        return sum / this.frameTimes.length;
    }

    /**
     * Get recommended quality tier based on performance
     */
    public getRecommendedTier(): DeviceTier {
        const avgFps = this.getAverageFps();

        if (avgFps >= 55) return 'high';
        if (avgFps >= 45) return 'medium';
        return 'low';
    }

    /**
     * Check if running smoothly
     */
    public isSmooth(): boolean {
        const avgFps = this.getAverageFps();
        return avgFps >= this.config.targetFps * 0.9;
    }

    /**
     * Check if having issues
     */
    public hasIssues(): boolean {
        const avgFps = this.getAverageFps();
        return avgFps < this.config.minAcceptableFps;
    }

    /**
     * Set callback for performance issues
     */
    public onPerformanceIssue(callback: () => void): void {
        this.onIssueCallback = callback;
    }

    /**
     * Reset statistics
     */
    public reset(): void {
        this.frameTimes = [];
        this.fpsValues = [];
        this.minFps = Infinity;
        this.maxFps = 0;
        this.frameCount = 0;
        this.memoryHistory = [];
    }

    /**
     * Get frame count since last reset
     */
    public getFrameCount(): number {
        return this.frameCount;
    }

    /**
     * Enable/disable monitoring
     */
    public setActive(active: boolean): void {
        if (active) {
            this.lastFrameTime = performance.now();
        }
    }
}
