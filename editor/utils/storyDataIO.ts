/**
 * storyDataIO.ts - Story Data Import/Export Utilities
 * =============================================================================
 *
 * Features:
 * - Load story data from JSON files
 * - Save story data to JSON files
 * - Runtime type validation for StoryData
 * - Export to formatted JSON string
 * - Import from JSON string with validation
 * - Generate backup file paths with timestamps
 *
 * CRITICAL: This is editor-only code for story file management
 */

import type {
  StoryData,
  StoryState,
  Transition,
  Act,
} from '../types/story.js';

interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

interface StoryDataValidationResult extends ValidationResult {
  data: StoryData | null;
}

/**
 * Validate StoryState structure
 */
function validateStoryState(state: unknown, stateId: string): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!state || typeof state !== 'object') {
    return { valid: false, errors: [`State "${stateId}" is not an object`], warnings };
  }

  const s = state as Partial<StoryState>;

  // Required fields
  if (typeof s.id !== 'string') {
    errors.push(`State "${stateId}": id is required and must be a string`);
  }
  if (typeof s.value !== 'number') {
    errors.push(`State "${stateId}": value is required and must be a number`);
  }
  if (typeof s.label !== 'string') {
    errors.push(`State "${stateId}": label is required and must be a string`);
  }
  if (typeof s.act !== 'number') {
    errors.push(`State "${stateId}": act is required and must be a number`);
  }
  if (typeof s.category !== 'string') {
    errors.push(`State "${stateId}": category is required and must be a string`);
  }
  if (typeof s.color !== 'string') {
    errors.push(`State "${stateId}": color is required and must be a string`);
  }

  // Optional fields with type checks
  if (s.zone !== null && typeof s.zone !== 'string') {
    errors.push(`State "${stateId}": zone must be a string or null`);
  }

  if (s.content !== undefined && typeof s.content !== 'object') {
    errors.push(`State "${stateId}": content must be an object`);
  }

  if (s.criteria !== undefined && typeof s.criteria !== 'object') {
    errors.push(`State "${stateId}": criteria must be an object`);
  }

  if (!Array.isArray(s.transitions)) {
    errors.push(`State "${stateId}": transitions must be an array`);
  } else {
    // Validate each transition
    s.transitions.forEach((trans: unknown, idx: number) => {
      const transResult = validateTransition(trans, `${stateId}.transitions[${idx}]`);
      errors.push(...transResult.errors);
      warnings.push(...transResult.warnings);
    });
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Validate Transition structure
 */
function validateTransition(transition: unknown, path: string): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!transition || typeof transition !== 'object') {
    return { valid: false, errors: [`${path}: transition is not an object`], warnings };
  }

  const t = transition as Partial<Transition>;

  if (typeof t.id !== 'string') {
    errors.push(`${path}: id is required and must be a string`);
  }
  if (typeof t.from !== 'string') {
    errors.push(`${path}: from is required and must be a string`);
  }
  if (typeof t.to !== 'string') {
    errors.push(`${path}: to is required and must be a string`);
  }
  if (typeof t.trigger !== 'string') {
    errors.push(`${path}: trigger is required and must be a string`);
  } else {
    const validTriggers = ['onComplete', 'onChoice', 'onTimeout', 'onProximity', 'onInteract', 'onState', 'custom'];
    if (!validTriggers.includes(t.trigger)) {
      warnings.push(`${path}: unknown trigger type "${t.trigger}"`);
    }
  }
  if (typeof t.label !== 'string') {
    errors.push(`${path}: label is required and must be a string`);
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Validate Act structure
 */
function validateAct(act: unknown, actId: string): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!act || typeof act !== 'object') {
    return { valid: false, errors: [`Act "${actId}" is not an object`], warnings };
  }

  const a = act as Partial<Act>;

  if (typeof a.name !== 'string') {
    errors.push(`Act "${actId}": name is required and must be a string`);
  }
  if (typeof a.color !== 'string') {
    errors.push(`Act "${actId}": color is required and must be a string`);
  }
  if (!Array.isArray(a.states)) {
    errors.push(`Act "${actId}": states is required and must be an array`);
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Runtime type guard for StoryData
 */
export function validateStoryData(data: unknown): data is StoryData {
  const result = validateStoryDataDetailed(data);
  return result.valid;
}

/**
 * Detailed validation with error messages
 */
export function validateStoryDataDetailed(data: unknown): StoryDataValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!data || typeof data !== 'object') {
    return {
      valid: false,
      errors: ['Data must be an object'],
      warnings,
      data: null,
    };
  }

  const d = data as Partial<StoryData>;

  // Validate states
  if (!d.states || typeof d.states !== 'object') {
    errors.push('states is required and must be an object');
  } else {
    for (const [stateId, state] of Object.entries(d.states)) {
      const result = validateStoryState(state, stateId);
      errors.push(...result.errors);
      warnings.push(...result.warnings);
    }
  }

  // Validate acts
  if (!d.acts || typeof d.acts !== 'object') {
    errors.push('acts is required and must be an object');
  } else {
    for (const [actId, act] of Object.entries(d.acts)) {
      const result = validateAct(act, actId);
      errors.push(...result.errors);
      warnings.push(...result.warnings);
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
    data: errors.length === 0 ? (data as StoryData) : null,
  };
}

/**
 * Load story data from a file path (for Node.js backend usage)
 */
export async function loadStoryData(path: string): Promise<StoryData> {
  try {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`Failed to load story data: ${response.statusText}`);
    }
    const json = await response.json();
    const result = validateStoryDataDetailed(json);

    if (!result.valid) {
      console.warn('Story data validation warnings:', result.warnings);
    }
    if (result.data === null) {
      throw new Error(`Invalid story data: ${result.errors.join(', ')}`);
    }

    return result.data;
  } catch (error) {
    throw new Error(`Failed to load story data from ${path}: ${error instanceof Error ? error.message : String(error)}`);
  }
}

/**
 * Save story data to a file (triggers download in browser)
 */
export async function saveStoryData(data: StoryData, filename?: string): Promise<void> {
  // Validate before saving
  const result = validateStoryDataDetailed(data);
  if (!result.valid) {
    throw new Error(`Cannot save invalid story data: ${result.errors.join(', ')}`);
  }

  // Generate JSON string
  const json = exportToJSON(data);

  // Create blob and download
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename || generateBackupFilename('storyData');
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Export story data to formatted JSON string
 */
export function exportToJSON(data: StoryData, pretty: boolean = true): string {
  return pretty ? JSON.stringify(data, null, 2) : JSON.stringify(data);
}

/**
 * Import story data from JSON string
 */
export function importFromJSON(json: string): StoryData {
  try {
    const data = JSON.parse(json);
    const result = validateStoryDataDetailed(data);

    if (!result.valid) {
      throw new Error(`Invalid story data: ${result.errors.join(', ')}`);
    }

    return result.data as StoryData;
  } catch (error) {
    throw new Error(`Failed to import story data: ${error instanceof Error ? error.message : String(error)}`);
  }
}

/**
 * Generate backup filename with timestamp
 */
export function generateBackupFilename(baseName: string = 'storyData'): string {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, '0');
  const day = String(now.getDate()).padStart(2, '0');
  const hours = String(now.getHours()).padStart(2, '0');
  const minutes = String(now.getMinutes()).padStart(2, '0');
  const seconds = String(now.getSeconds()).padStart(2, '0');

  return `${baseName}_${year}${month}${day}_${hours}${minutes}${seconds}.json`;
}

/**
 * Generate backup path (for server-side usage)
 */
export function generateBackupPath(originalPath: string): string {
  const dir = originalPath.substring(0, originalPath.lastIndexOf('/'));
  const baseName = originalPath.substring(originalPath.lastIndexOf('/') + 1).replace('.json', '');
  const timestamp = generateBackupFilename(baseName);
  return dir ? `${dir}/${timestamp}` : timestamp;
}

/**
 * Merge two story data objects (states from override take precedence)
 */
export function mergeStoryData(base: StoryData, override: Partial<StoryData>): StoryData {
  const merged: StoryData = {
    states: { ...base.states },
    acts: { ...base.acts },
  };

  // Merge states
  if (override.states) {
    Object.assign(merged.states, override.states);
  }

  // Merge acts
  if (override.acts) {
    Object.assign(merged.acts, override.acts);
  }

  return merged;
}

/**
 * Get state count by act
 */
export function getStateCountByAct(data: StoryData): Record<number, number> {
  const counts: Record<number, number> = {};

  for (const state of Object.values(data.states)) {
    counts[state.act] = (counts[state.act] || 0) + 1;
  }

  return counts;
}

/**
 * Get all transition targets for a state
 */
export function getTransitionTargets(stateId: string, data: StoryData): string[] {
  const state = data.states[stateId];
  if (!state) return [];

  return state.transitions.map(t => t.to).filter(t => data.states[t]);
}

/**
 * Find states that transition to the given state
 */
export function findIncomingStates(targetStateId: string, data: StoryData): string[] {
  const incoming: string[] = [];

  for (const [id, state] of Object.entries(data.states)) {
    if (state.transitions.some(t => t.to === targetStateId)) {
      incoming.push(id);
    }
  }

  return incoming;
}

export default {
  loadStoryData,
  saveStoryData,
  validateStoryData,
  validateStoryDataDetailed,
  exportToJSON,
  importFromJSON,
  generateBackupFilename,
  generateBackupPath,
  mergeStoryData,
  getStateCountByAct,
  getTransitionTargets,
  findIncomingStates,
};
