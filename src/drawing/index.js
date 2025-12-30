/**
 * drawing/index.js - DRAWING SYSTEM MODULE EXPORTS
 * =============================================================================
 *
 * ROLE: Central export file for the drawing recognition and gameplay system.
 * Provides access to ML-based drawing recognition, particle canvas, and
 * image preprocessing utilities.
 *
 * =============================================================================
 */

export { DrawingRecognitionManager } from "./drawingRecognitionManager.js";
export { ParticleCanvas3D } from "./particleCanvas3D.js";
export { ImagePreprocessor } from "./imagePreprocessor.js";
export { DRAWING_LABELS, FULL_LABEL_SET } from "./drawingLabels.js";
export { createDrawingPuzzle } from "./drawingExample.js";
