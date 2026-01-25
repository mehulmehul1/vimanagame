/**
 * Entities Module - All game entity exports
 */

// Water and vortex
export { WaterMaterial } from './WaterMaterial';

// Jelly creatures
export { JellyManager, type JellyConfig } from './JellyManager';
export { JellyCreature, type JellyState } from './JellyCreature';

// Feedback systems
export { GentleFeedback, type ShakeConfig } from './GentleFeedback';
export { FeedbackManager } from './FeedbackManager';

// Duet mechanics
export { PatientJellyManager, DuetState, TEACHING_SEQUENCES, type DuetCallbacks, type PatientJellyConfig } from './PatientJellyManager';
export { DuetProgressTracker } from './DuetProgressTracker';

// Vortex
export { VortexActivationController } from './VortexActivationController';
export { VortexLightingManager } from './VortexLightingManager';
export { PlatformRideAnimator } from './PlatformRideAnimator';

// Shell collection
export { ShellCollectible, type ShellState, type ShellCollectibleConfig } from './ShellCollectible';
export { ShellManager, ShellCollectionState, CHAMBER_NAMES, type ChamberId } from './ShellManager';

// White flash ending
export { WhiteFlashEnding, type FlashState, type FlashConfig } from './WhiteFlashEnding';
export { WhiteFlashManager, type EndingCallbacks, type EndingConfig } from './WhiteFlashManager';
