/**
 * generateSceneCriteria.js - Scene Criteria Code Generator
 * =============================================================================
 *
 * Reads editor/data/storyData.json and generates:
 * - Zone mapping constants (state ranges for each zone)
 * - Criteria helper functions for sceneData.js
 * - State-to-zone lookup table
 *
 * Usage: node scripts/generateSceneCriteria.js [--check] [--output=path]
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Paths
const STORY_DATA_PATH = path.join(__dirname, '../editor/data/storyData.json');
const DEFAULT_OUTPUT_PATH = path.join(__dirname, '../src/sceneCriteria.js');

// Parse CLI arguments
const args = process.argv.slice(2);
const options = {
  check: args.includes('--check'),
  output: args.find(arg => arg.startsWith('--output='))?.split('=')[1] || DEFAULT_OUTPUT_PATH,
  verbose: args.includes('--verbose') || args.includes('-v'),
};

/**
 * Read story data from JSON file
 */
function readStoryData() {
  const content = fs.readFileSync(STORY_DATA_PATH, 'utf-8');
  return JSON.parse(content);
}

/**
 * Extract all unique zones from story data
 */
function extractZones(storyData) {
  const zones = new Set();
  for (const state of Object.values(storyData.states)) {
    if (state.zone) {
      zones.add(state.zone);
    }
  }
  return Array.from(zones).sort();
}

/**
 * Group states by zone and extract value ranges
 */
function buildZoneStateRanges(storyData) {
  const zoneRanges = {};

  for (const state of Object.values(storyData.states)) {
    const { zone, value, id } = state;

    if (!zone) continue;

    if (!zoneRanges[zone]) {
      zoneRanges[zone] = {
        states: [],
        minValue: Infinity,
        maxValue: -Infinity,
      };
    }

    zoneRanges[zone].states.push({ id, value });
    zoneRanges[zone].minValue = Math.min(zoneRanges[zone].minValue, value);
    zoneRanges[zone].maxValue = Math.max(zoneRanges[zone].maxValue, value);
  }

  return zoneRanges;
}

/**
 * Build state to zone lookup table
 */
function buildStateToZoneMap(storyData) {
  const stateToZone = {};

  for (const [id, state] of Object.entries(storyData.states)) {
    if (state.zone) {
      stateToZone[id] = state.zone;
    }
  }

  return stateToZone;
}

/**
 * Generate zone constants
 */
function generateZoneConstants(zoneRanges) {
  const lines = ['// Zone state ranges for scene loading criteria', ''];
  lines.push('export const ZONE_STATE_RANGES = {');

  for (const [zone, data] of Object.entries(zoneRanges)) {
    const { minValue, maxValue } = data;
    lines.push(`  ${zone.toUpperCase()}: {`);
    lines.push(`    min: GAME_STATES.${Object.entries(zoneRanges)
      .find(([_, d]) => d.states.find(s => s.value === minValue)?.[1]?.states.find(s => s.value === minValue))
      ?.[1].states.find(s => s.value === minValue)?.id || minValue},`);
    lines.push(`    max: GAME_STATES.${Object.entries(zoneRanges)
      .find(([_, d]) => d.states.find(s => s.value === maxValue)?.[1]?.states.find(s => s.value === maxValue))
      ?.[1].states.find(s => s.value === maxValue)?.id || maxValue},`);
    lines.push(`    minValue: ${minValue},`);
    lines.push(`    maxValue: ${maxValue},`);
    lines.push('  },');
  }

  lines.push('};');
  lines.push('');

  return lines.join('\n');
}

/**
 * Generate criteria helper functions
 */
function generateCriteriaHelpers(zoneRanges) {
  const lines = [];
  lines.push('/**');
  lines.push(' * Create a criteria object for loading during a zone\'s active state range');
  lines.push(' * @param {string} zone - Zone name (e.g., "plaza", "interior")');
  lines.push(' * @param {Object} extraCriteria - Additional criteria to merge');
  lines.push(' * @returns {Object} Criteria object for sceneData.js');
  lines.push(' */');
  lines.push('export function duringZone(zone, extraCriteria = {}) {');
  lines.push('  const range = ZONE_STATE_RANGES[zone.toUpperCase()];');
  lines.push('  if (!range) {');
  lines.push('    console.warn(`Unknown zone: ${zone}`);');
  lines.push('    return {};');
  lines.push('  }');
  lines.push('  return {');
  lines.push('    currentState: {');
  lines.push('      $gte: range.min,');
  lines.push('      $lte: range.max,');
  lines.push('    },');
  lines.push('    ...extraCriteria,');
  lines.push('  };');
  lines.push('}');
  lines.push('');
  lines.push('/**');
  lines.push(' * Create a criteria object for loading before a state');
  lines.push(' * @param {number} stateValue - GAME_STATES value');
  lines.push(' * @returns {Object} Criteria object');
  lines.push(' */');
  lines.push('export function beforeState(stateValue) {');
  lines.push('  return { currentState: { $lt: stateValue } };');
  lines.push('}');
  lines.push('');
  lines.push('/**');
  lines.push(' * Create a criteria object for loading during a state range');
  lines.push(' * @param {number} minState - Minimum GAME_STATES value (inclusive)');
  lines.push(' * @param {number} maxState - Maximum GAME_STATES value (exclusive)');
  lines.push(' * @returns {Object} Criteria object');
  lines.push(' */');
  lines.push('export function duringStates(minState, maxState) {');
  lines.push('  return { currentState: { $gte: minState, $lt: maxState } };');
  lines.push('}');
  lines.push('');
  lines.push('/**');
  lines.push(' * Create a criteria object for loading from a state onward');
  lines.push(' * @param {number} stateValue - GAME_STATES value');
  lines.push(' * @returns {Object} Criteria object');
  lines.push(' */');
  lines.push('export function fromState(stateValue) {');
  lines.push('  return { currentState: { $gte: stateValue } };');
  lines.push('}');
  lines.push('');
  lines.push('/**');
  lines.push(' * Create a criteria object for loading until a state');
  lines.push(' * @param {number} stateValue - GAME_STATES value');
  lines.push(' * @returns {Object} Criteria object');
  lines.push(' */');
  lines.push('export function untilState(stateValue) {');
  lines.push('  return { currentState: { $lte: stateValue } };');
  lines.push('}');
  lines.push('');

  return lines.join('\n');
}

/**
 * Generate state to zone lookup
 */
function generateStateToZoneLookup(stateToZone) {
  const lines = ['// Lookup table for state to zone mapping', ''];
  lines.push('export const STATE_TO_ZONE = {');

  // Sort by state value
  const sortedEntries = Object.entries(stateToZone).sort((a, b) => {
    const aValue = parseInt(a[0].split('_').pop()) || 0;
    const bValue = parseInt(b[0].split('_').pop()) || 0;
    return aValue - bValue;
  });

  for (const [stateId, zone] of sortedEntries) {
    lines.push(`  ${stateId}: '${zone}',`);
  }

  lines.push('};');
  lines.push('');

  return lines.join('\n');
}

/**
 * Generate zone list constant
 */
function generateZoneList(zones) {
  const lines = ['// All zone names', ''];
  lines.push('export const ZONES = [');
  for (const zone of zones) {
    lines.push(`  '${zone}',`);
  }
  lines.push('];');
  lines.push('');

  return lines.join('\n');
}

/**
 * Generate zone description comments
 */
function generateZoneDescriptions(zoneRanges) {
  const lines = ['// Zone descriptions with active states', ''];
  lines.push('/**');
  lines.push(' * Zone Active States:');
  lines.push(' *');

  for (const [zone, data] of Object.entries(zoneRanges)) {
    const stateIds = data.states.map(s => s.id).sort((a, b) => a.localeCompare(b));
    lines.push(` * ${zone.toUpperCase()}:`);
    for (const stateId of stateIds) {
      lines.push(` *   - ${stateId}`);
    }
  }

  lines.push(' */');
  lines.push('');

  return lines.join('\n');
}

/**
 * Generate complete sceneCriteria.js file
 */
function generateSceneCriteriaFile() {
  const storyData = readStoryData();
  const timestamp = new Date().toISOString();

  const zones = extractZones(storyData);
  const zoneRanges = buildZoneStateRanges(storyData);
  const stateToZone = buildStateToZoneMap(storyData);

  const content = `/**
 * sceneCriteria.js - SCENE CRITERIA HELPERS
 * =============================================================================
 *
 * AUTO-GENERATED from editor/data/storyData.json
 * Generation timestamp: ${timestamp}
 *
 * WARNING: This file is auto-generated. Do not edit directly.
 * Modify the story data in the editor and run the generator instead.
 *
 * ROLE: Provides zone-based criteria helpers for sceneData.js
 *
 * KEY EXPORTS:
 * - ZONES: Array of all zone names
 * - ZONE_STATE_RANGES: Min/max state values for each zone
 * - STATE_TO_ZONE: Lookup table mapping state IDs to zones
 * - Helper functions: duringZone(), beforeState(), duringStates(), etc.
 *
 * USAGE:
 * import { duringZone, duringStates, fromState, beforeState } from './sceneCriteria.js';
 *
 * // In sceneData.js:
 * criteria: duringZone('plaza', { performanceProfile: 'max' })
 * criteria: duringStates(GAME_STATES.INTRO, GAME_STATES.ENTERING_OFFICE)
 * criteria: fromState(GAME_STATES.POST_DRIVE_BY)
 *
 * =============================================================================
 */

import { GAME_STATES } from './gameData.js';

${generateZoneDescriptions(zoneRanges)}
${generateZoneList(zones)}
${generateZoneConstants(zoneRanges)}
${generateStateToZoneLookup(stateToZone)}
${generateCriteriaHelpers(zoneRanges)}
/**
 * Get the active zone for a given state value
 * @param {number} stateValue - Current GAME_STATES value
 * @returns {string|null} Zone name or null if no zone matches
 */
export function getZoneForState(stateValue) {
  for (const [zone, range] of Object.entries(ZONE_STATE_RANGES)) {
    if (stateValue >= range.minValue && stateValue <= range.maxValue) {
      return zone.toLowerCase();
    }
  }
  return null;
}

/**
 * Check if a state is within a zone's range
 * @param {number} stateValue - Current GAME_STATES value
 * @param {string} zone - Zone name to check
 * @returns {boolean} True if state is in zone range
 */
export function isStateInZone(stateValue, zone) {
  const range = ZONE_STATE_RANGES[zone.toUpperCase()];
  if (!range) return false;
  return stateValue >= range.minValue && stateValue <= range.maxValue;
}

/**
 * Create criteria for exterior zones (plaza, fourWay, alley, etc.)
 * @param {Object} extraCriteria - Additional criteria to merge
 * @returns {Object} Criteria object for exterior scenes
 */
export function duringExterior(extraCriteria = {}) {
  return {
    currentState: {
      $gte: GAME_STATES.LOADING,
      $lt: GAME_STATES.ENTERING_OFFICE,
    },
    ...extraCriteria,
  };
}

/**
 * Create criteria for interior zone
 * @param {Object} extraCriteria - Additional criteria to merge
 * @returns {Object} Criteria object for interior scenes
 */
export function duringInterior(extraCriteria = {}) {
  return {
    currentState: {
      $gte: GAME_STATES.POST_DRIVE_BY,
    },
    ...extraCriteria,
  };
}

export default {
  ZONES,
  ZONE_STATE_RANGES,
  STATE_TO_ZONE,
  duringZone,
  duringStates,
  beforeState,
  fromState,
  untilState,
  getZoneForState,
  isStateInZone,
  duringExterior,
  duringInterior,
};
`;

  return { content, zones, zoneRanges, stateToZone };
}

/**
 * Validate generated criteria against story data
 */
function validateCriteria(storyData, zoneRanges) {
  const issues = [];

  // Check for states without zones that should have them
  for (const [id, state] of Object.entries(storyData.states)) {
    if (!state.zone && state.value > 0) {
      // States after INTRO should generally have zones
      if (state.category !== 'Intro & Title' && state.category !== 'Outro') {
        issues.push(`Warning: State ${id} has no zone defined`);
      }
    }
  }

  // Check for zone continuity gaps
  const sortedStates = Object.values(storyData.states)
    .filter(s => s.zone)
    .sort((a, b) => a.value - b.value);

  for (let i = 1; i < sortedStates.length; i++) {
    const prev = sortedStates[i - 1];
    const curr = sortedStates[i];
    if (curr.value - prev.value > 1) {
      // Check if there's an unzoned state in between
      const hasUnzonedBetween = Object.values(storyData.states).some(
        s => !s.zone && s.value > prev.value && s.value < curr.value
      );
      if (!hasUnzonedBetween) {
        issues.push(`Warning: Gap in zone coverage between ${prev.id} and ${curr.id}`);
      }
    }
  }

  return issues;
}

/**
 * Main execution
 */
function main() {
  console.log('🎬 Scene Criteria Code Generator');
  console.log('===============================');
  console.log(`Reading: ${STORY_DATA_PATH}`);
  console.log(`Writing: ${options.output}`);

  try {
    const storyData = readStoryData();
    const { content, zones, zoneRanges, stateToZone } = generateSceneCriteriaFile();

    // Run validation
    const issues = validateCriteria(storyData, zoneRanges);
    if (issues.length > 0 && options.verbose) {
      console.log('\n📋 Validation issues:');
      issues.forEach(issue => console.log(`  ${issue}`));
    }

    if (options.check) {
      console.log('\n✅ Validation complete (--check mode, no file written)');
      console.log(`   Found ${issues.length} potential issues`);
      return;
    }

    // Create backup of existing file
    if (fs.existsSync(options.output)) {
      const backupPath = options.output.replace('.js', '.backup.js');
      fs.copyFileSync(options.output, backupPath);
      console.log('✓ Backup created');
    }

    // Write new file
    fs.writeFileSync(options.output, content, 'utf-8');

    console.log('\n✅ Generation complete!');
    console.log(`   Zones extracted: ${zones.join(', ')}`);
    console.log(`   States mapped: ${Object.keys(stateToZone).length}`);
    console.log(`   Zone ranges: ${Object.keys(zoneRanges).length}`);

    if (issues.length > 0) {
      console.log(`\n⚠️  Found ${issues.length} validation issues (use --verbose for details)`);
    }

  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export { main };
