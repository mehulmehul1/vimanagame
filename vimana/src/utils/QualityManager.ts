/**
 * QualityManager.ts - Adaptive Quality Settings
 * =============================================
 *
 * Manages adaptive quality settings based on device capabilities
 * and real-time performance. Adjusts rendering parameters to
 * maintain target frame rate.
 *
 * Story 4.8: Performance Validation
 */

import * as THREE from 'three';
import { DeviceCapabilities, DeviceTier } from './DeviceCapabilities';
import { QUALITY_PRESETS, QualitySettings } from './QualityPresets';

const logger = {
  log: (msg: string, ...args: unknown[]) => console.log(`[QualityManager] ${msg}`, ...args),
  warn: (msg: string, ...args: unknown[]) => console.warn(`[QualityManager] ${msg}`, ...args),
};

/**
 * Quality manager configuration
 */
export interface QualityManagerConfig {
  adaptiveEnabled: boolean;
  autoDowngrade: boolean;
  autoUpgrade: boolean;
  downgradeThreshold: number; // % of target FPS to trigger downgrade (0.8 = 80%)
  upgradeThreshold: number; // % of target FPS to trigger upgrade (1.1 = 110%)
  stabilizationFrames: number; // Frames before quality change
}

/**
 * Quality level statistics
 */
export interface QualityStats {
  currentTier: DeviceTier;
  targetFPS: number;
  currentFPS: number;
  adaptiveEnabled: boolean;
  lastAdjustmentTime: number;
  adjustmentsCount: number;
}

/**
 * Manages adaptive quality settings for performance optimization
 */
export class QualityManager {
  private device: DeviceCapabilities;
  private currentTier: DeviceTier;
  private currentSettings: QualitySettings;
  private config: QualityManagerConfig;

  // Adaptive quality tracking
  private currentFPS = 60;
  private fpsHistory: number[] = [];
  private lastAdjustmentTime = 0;
  private adjustmentsCount = 0;
  private belowThresholdFrames = 0;
  private aboveThresholdFrames = 0;

  // Renderer reference for applying settings
  private renderer: THREE.WebGLRenderer | THREE.WebGPURenderer | null = null;

  constructor(config: Partial<QualityManagerConfig> = {}) {
    this.device = DeviceCapabilities.getInstance();
    this.currentTier = this.device.getTier();
    this.currentSettings = { ...QUALITY_PRESETS[this.currentTier] };

    this.config = {
      adaptiveEnabled: true,
      autoDowngrade: true,
      autoUpgrade: false, // Conservative: don't auto-upgrade by default
      downgradeThreshold: 0.8, // Downgrade if FPS < 80% of target
      upgradeThreshold: 1.1, // Upgrade if FPS > 110% of target
      stabilizationFrames: 60, // Wait 1 second (at 60 FPS) before adjusting
      ...config,
    };

    logger.log('QualityManager initialized with tier:', this.currentTier);
  }

  /**
   * Initialize quality manager with renderer
   */
  public initialize(renderer: THREE.WebGLRenderer | THREE.WebGPURenderer): void {
    this.renderer = renderer;
    this.applyCurrentSettings();
    logger.log('Quality settings applied to renderer');
  }

  /**
   * Update quality manager - call each frame
   */
  public update(fps: number): void {
    this.currentFPS = fps;

    // Track FPS history
    this.fpsHistory.push(fps);
    if (this.fpsHistory.length > this.config.stabilizationFrames) {
      this.fpsHistory.shift();
    }

    // Adaptive quality adjustment
    if (this.config.adaptiveEnabled) {
      this.checkAdaptiveQuality();
    }
  }

  /**
   * Check and adjust quality based on FPS
   */
  private checkAdaptiveQuality(): void {
    if (this.fpsHistory.length < this.config.stabilizationFrames) {
      return;
    }

    const avgFPS = this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length;
    const targetFPS = this.getTargetFPS();

    // Check for downgrade
    if (this.config.autoDowngrade && avgFPS < targetFPS * this.config.downgradeThreshold) {
      this.belowThresholdFrames++;

      if (this.belowThresholdFrames >= 30) { // Sustained poor performance
        this.downgradeQuality();
        this.belowThresholdFrames = 0;
      }
    } else {
      this.belowThresholdFrames = 0;
    }

    // Check for upgrade
    if (this.config.autoUpgrade && avgFPS > targetFPS * this.config.upgradeThreshold) {
      this.aboveThresholdFrames++;

      if (this.aboveThresholdFrames >= 120) { // Sustained good performance (2 seconds)
        this.upgradeQuality();
        this.aboveThresholdFrames = 0;
      }
    } else {
      this.aboveThresholdFrames = 0;
    }
  }

  /**
   * Downgrade quality by one tier
   */
  private downgradeQuality(): void {
    const tiers: DeviceTier[] = ['ultra', 'high', 'medium', 'low'];
    const currentIndex = tiers.indexOf(this.currentTier);

    if (currentIndex < tiers.length - 1) {
      const newTier = tiers[currentIndex + 1];
      this.setTier(newTier);
      this.adjustmentsCount++;
      this.lastAdjustmentTime = performance.now();
      logger.warn(`Downgraded quality to ${newTier} (FPS: ${this.currentFPS.toFixed(1)})`);
    }
  }

  /**
   * Upgrade quality by one tier
   */
  private upgradeQuality(): void {
    const tiers: DeviceTier[] = ['ultra', 'high', 'medium', 'low'];
    const currentIndex = tiers.indexOf(this.currentTier);

    if (currentIndex > 0) {
      const newTier = tiers[currentIndex - 1];
      this.setTier(newTier);
      this.adjustmentsCount++;
      this.lastAdjustmentTime = performance.now();
      logger.log(`Upgraded quality to ${newTier} (FPS: ${this.currentFPS.toFixed(1)})`);
    }
  }

  /**
   * Apply current quality settings to renderer
   */
  private applyCurrentSettings(): void {
    if (!this.renderer) {
      return;
    }

    // Pixel ratio cap
    const pixelRatio = Math.min(
      window.devicePixelRatio * this.currentSettings.textureMultiplier,
      this.device.isMobile() ? 1.5 : 2.0
    );
    this.renderer.setPixelRatio(pixelRatio);

    // Shadow configuration
    if ('shadowMap' in this.renderer) {
      const webglRenderer = this.renderer as THREE.WebGLRenderer;
      webglRenderer.shadowMap.enabled = this.currentSettings.enableShadows;
    }

    // Tone mapping
    this.renderer.toneMapping = this.currentSettings.shaderQuality > 0.7
      ? THREE.ACESFilmicToneMapping
      : THREE.LinearToneMapping;
    this.renderer.toneMappingExposure = 1.0;

    // Output color space
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
  }

  /**
   * Set quality tier
   */
  public setTier(tier: DeviceTier): void {
    this.currentTier = tier;
    this.currentSettings = { ...QUALITY_PRESETS[tier] };
    this.applyCurrentSettings();
  }

  /**
   * Get current quality tier
   */
  public getTier(): DeviceTier {
    return this.currentTier;
  }

  /**
   * Get current quality settings
   */
  public getSettings(): QualitySettings {
    return { ...this.currentSettings };
  }

  /**
   * Update specific setting
   */
  public updateSetting<K extends keyof QualitySettings>(key: K, value: QualitySettings[K]): void {
    this.currentSettings[key] = value;
    this.applyCurrentSettings();
  }

  /**
   * Get target FPS for current device
   */
  public getTargetFPS(): number {
    return this.device.isMobile() ? 30 : 60;
  }

  /**
   * Get particle count based on current tier
   */
  public getParticleCount(baseCount: number): number {
    return Math.floor(baseCount * this.currentSettings.particleMultiplier);
  }

  /**
   * Get recommended particle count for fluid simulation
   */
  public getRecommendedParticleCount(): number {
    switch (this.currentTier) {
      case 'ultra': return 50000;
      case 'high': return 10000;
      case 'medium': return 5000;
      case 'low': return 3000;
    }
  }

  /**
   * Enable/disable adaptive quality
   */
  public setAdaptiveEnabled(enabled: boolean): void {
    this.config.adaptiveEnabled = enabled;
  }

  /**
   * Check if adaptive quality is enabled
   */
  public isAdaptiveEnabled(): boolean {
    return this.config.adaptiveEnabled;
  }

  /**
   * Get quality statistics
   */
  public getStats(): QualityStats {
    return {
      currentTier: this.currentTier,
      targetFPS: this.getTargetFPS(),
      currentFPS: this.currentFPS,
      adaptiveEnabled: this.config.adaptiveEnabled,
      lastAdjustmentTime: this.lastAdjustmentTime,
      adjustmentsCount: this.adjustmentsCount,
    };
  }

  /**
   * Log quality stats to console
   */
  public logStats(): void {
    const stats = this.getStats();
    logger.log('Quality Stats:', {
      tier: stats.currentTier,
      fps: `${stats.currentFPS.toFixed(1)} / ${stats.targetFPS}`,
      adaptive: stats.adaptiveEnabled,
      adjustments: stats.adjustmentsCount,
    });
  }

  /**
   * Reset to device default tier
   */
  public reset(): void {
    this.setTier(this.device.getTier());
    this.fpsHistory = [];
    this.belowThresholdFrames = 0;
    this.aboveThresholdFrames = 0;
    logger.log('QualityManager reset to device tier:', this.currentTier);
  }

  /**
   * Dispose of quality manager
   */
  public dispose(): void {
    this.renderer = null;
  }
}

export default QualityManager;
