/**
 * Utils Module - Performance and resource management utilities
 */

export { DeviceCapabilities, DeviceTier, type CapabilityInfo } from './DeviceCapabilities';
export { PerformanceMonitor, type PerformanceMetrics, type PerformanceConfig } from './PerformanceMonitor';
export { QualityPresets, QUALITY_PRESETS, type QualitySettings } from './QualityPresets';
export { ResourceManager } from './ResourceManager';

// Story 4.8: Performance validation utilities
export {
    PerformanceHUD,
    type FrameTimeBreakdown,
    type PerformanceHUDConfig,
} from './PerformanceHUD';
export {
    QualityManager,
    type QualityManagerConfig,
    type QualityStats,
} from './QualityManager';
