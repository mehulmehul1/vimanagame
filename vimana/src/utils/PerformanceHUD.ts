/**
 * PerformanceHUD.ts - Performance Debug HUD
 * ==========================================
 *
 * On-screen debug display for performance metrics.
 * Shows FPS, frame time, memory usage, and subsystem timings.
 *
 * Story 4.8: Performance Validation
 */

import { PerformanceMetrics } from './PerformanceMonitor';

const logger = {
  log: (msg: string, ...args: unknown[]) => console.log(`[PerformanceHUD] ${msg}`, ...args),
};

/**
 * Frame time breakdown for subsystems
 */
export interface FrameTimeBreakdown {
  webgpu: number; // WebGPU rendering time (ms)
  fluid: number;  // Fluid compute time (ms)
  splats: number; // Visionary splats time (ms)
  post: number;   // Post-processing time (ms)
  other: number;  // Other systems time (ms)
}

/**
 * HUD configuration
 */
export interface PerformanceHUDConfig {
  enabled: boolean;
  position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  updateInterval: number; // ms between updates
  showMemory: boolean;
  showBreakdown: boolean;
  targetFPS: number;
}

/**
 * Performance HUD for on-screen debugging
 */
export class PerformanceHUD {
  private element: HTMLElement;
  private config: PerformanceHUDConfig;
  private monitor: PerformanceMetrics | null = null;
  private breakdown: FrameTimeBreakdown = { webgpu: 0, fluid: 0, splats: 0, post: 0, other: 0 };
  private lastUpdateTime = 0;
  private isVisible = false;

  constructor(config: Partial<PerformanceHUDConfig> = {}) {
    this.config = {
      enabled: false,
      position: 'top-left',
      updateInterval: 500,
      showMemory: true,
      showBreakdown: true,
      targetFPS: 60,
      ...config,
    };

    this.element = this.createHUD();
    this.setVisible(this.config.enabled);
  }

  /**
   * Create HUD element
   */
  private createHUD(): HTMLElement {
    const hud = document.createElement('div');
    hud.id = 'performance-hud';
    hud.className = 'performance-hud';
    hud.style.cssText = `
      position: fixed;
      padding: 12px;
      font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
      font-size: 12px;
      line-height: 1.4;
      z-index: 10000;
      pointer-events: none;
      display: none;
    `;
    this.updatePosition();
    document.body.appendChild(hud);
    return hud;
  }

  /**
   * Update HUD position based on config
   */
  private updatePosition(): void {
    const positions = {
      'top-left': 'top: 10px; left: 10px;',
      'top-right': 'top: 10px; right: 10px;',
      'bottom-left': 'bottom: 10px; left: 10px;',
      'bottom-right': 'bottom: 10px; right: 10px;',
    };
    const baseStyle = positions[this.config.position];
    this.element.style.cssText = `
      position: fixed;
      padding: 12px;
      font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
      font-size: 12px;
      line-height: 1.4;
      z-index: 10000;
      pointer-events: none;
      display: ${this.isVisible ? 'block' : 'none'};
      background: rgba(0, 0, 0, 0.75);
      color: #00ff00;
      border-radius: 4px;
      min-width: 180px;
      ${baseStyle}
    `;
  }

  /**
   * Set performance monitor for FPS tracking
   */
  public setMonitor(monitor: PerformanceMetrics | null): void {
    this.monitor = monitor;
  }

  /**
   * Update frame time breakdown
   */
  public setBreakdown(breakdown: Partial<FrameTimeBreakdown>): void {
    this.breakdown = { ...this.breakdown, ...breakdown };
  }

  /**
   * Update specific subsystem time
   */
  public updateSubsystemTime(system: keyof FrameTimeBreakdown, timeMs: number): void {
    this.breakdown[system] = timeMs;
  }

  /**
   * Update HUD display
   */
  public update(): void {
    const now = performance.now();
    if (now - this.lastUpdateTime < this.config.updateInterval) {
      return;
    }
    this.lastUpdateTime = now;

    if (!this.isVisible) {
      return;
    }

    const html = this.generateHTML();
    this.element.innerHTML = html;
  }

  /**
   * Generate HTML for HUD
   */
  private generateHTML(): string {
    const fps = this.monitor?.fps ?? 60;
    const frameTime = this.monitor?.frameTime ?? 16.67;
    const memory = this.monitor?.memory ?? 0;

    // Calculate color based on FPS
    const fpsPercent = fps / this.config.targetFPS;
    let fpsColor = '#00ff00'; // Green
    if (fpsPercent < 0.5) {
      fpsColor = '#ff0000'; // Red
    } else if (fpsPercent < 0.8) {
      fpsColor = '#ffff00'; // Yellow
    }

    // Calculate total breakdown time
    const totalBreakdown = this.breakdown.webgpu + this.breakdown.fluid +
                           this.breakdown.splats + this.breakdown.post + this.breakdown.other;

    let html = `
      <div style="font-weight: bold; margin-bottom: 8px; color: ${fpsColor};">
        FPS: ${fps.toFixed(1)}
      </div>
      <div>Frame: ${frameTime.toFixed(2)}ms / ${(1000 / this.config.targetFPS).toFixed(2)}ms</div>
    `;

    if (this.config.showMemory && memory > 0) {
      html += `<div>Memory: ${memory.toFixed(1)} GB</div>`;
    }

    if (this.config.showBreakdown && totalBreakdown > 0) {
      html += `<div style="margin-top: 8px; border-top: 1px solid #444; padding-top: 4px;">`;
      html += `<div>WebGPU: ${this.breakdown.webgpu.toFixed(2)}ms</div>`;
      if (this.breakdown.fluid > 0) {
        html += `<div>Fluid: ${this.breakdown.fluid.toFixed(2)}ms</div>`;
      }
      if (this.breakdown.splats > 0) {
        html += `<div>Splats: ${this.breakdown.splats.toFixed(2)}ms</div>`;
      }
      if (this.breakdown.post > 0) {
        html += `<div>Post: ${this.breakdown.post.toFixed(2)}ms</div>`;
      }
      html += `</div>`;
    }

    return html;
  }

  /**
   * Show HUD
   */
  public show(): void {
    this.setVisible(true);
  }

  /**
   * Hide HUD
   */
  public hide(): void {
    this.setVisible(false);
  }

  /**
   * Toggle HUD visibility
   */
  public toggle(): void {
    this.setVisible(!this.isVisible);
  }

  /**
   * Set HUD visibility
   */
  public setVisible(visible: boolean): void {
    this.isVisible = visible;
    this.element.style.display = visible ? 'block' : 'none';
  }

  /**
   * Check if HUD is visible
   */
  public isVisible(): boolean {
    return this.isVisible;
  }

  /**
   * Set HUD position
   */
  public setPosition(position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right'): void {
    this.config.position = position;
    this.updatePosition();
  }

  /**
   * Set target FPS for color calculation
   */
  public setTargetFPS(fps: number): void {
    this.config.targetFPS = fps;
  }

  /**
   * Dispose of HUD
   */
  public dispose(): void {
    if (this.element.parentElement) {
      this.element.parentElement.removeChild(this.element);
    }
  }

  /**
   * Create singleton HUD instance (optional pattern)
   */
  private static instance: PerformanceHUD | null = null;

  public static getInstance(config?: Partial<PerformanceHUDConfig>): PerformanceHUD {
    if (!PerformanceHUD.instance) {
      PerformanceHUD.instance = new PerformanceHUD(config);
    }
    return PerformanceHUD.instance;
  }

  /**
   * Log performance summary to console
   */
  public logSummary(): void {
    const fps = this.monitor?.fps ?? 60;
    const frameTime = this.monitor?.frameTime ?? 16.67;
    const memory = this.monitor?.memory ?? 0;

    logger.log('Performance Summary:', {
      fps: fps.toFixed(1),
      frameTime: `${frameTime.toFixed(2)}ms`,
      memory: `${memory.toFixed(1)} GB`,
      breakdown: {
        webgpu: `${this.breakdown.webgpu.toFixed(2)}ms`,
        fluid: `${this.breakdown.fluid.toFixed(2)}ms`,
        splats: `${this.breakdown.splats.toFixed(2)}ms`,
        post: `${this.breakdown.post.toFixed(2)}ms`,
        other: `${this.breakdown.other.toFixed(2)}ms`,
      },
    });
  }
}

export default PerformanceHUD;
