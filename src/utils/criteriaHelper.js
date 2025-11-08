import { Logger } from "./logger.js";

const logger = new Logger("CriteriaHelper", false);

/**
 * Criteria Helper - Unified criteria checking for game state
 *
 * Supports:
 * - Simple equality: { currentState: GAME_STATES.INTRO }
 * - Comparisons: { currentState: { $gte: GAME_STATES.INTRO, $lt: GAME_STATES.DRIVE_BY } }
 * - Arrays: { currentState: { $in: [STATE1, STATE2] } }
 * - Multiple conditions on same key
 *
 * Operators:
 * - $eq: equals (same as simple value)
 * - $ne: not equals
 * - $gt: greater than
 * - $gte: greater than or equal
 * - $lt: less than
 * - $lte: less than or equal
 * - $in: in array
 * - $nin: not in array
 * - $mod: modulo check - { $mod: [divisor, remainder] } checks if value % divisor === remainder
 */

/**
 * Check if a single value matches a criteria definition
 * @param {any} value - Value to check (e.g., state.currentState)
 * @param {any} criteria - Criteria definition (value, or object with operators)
 * @returns {boolean}
 */
export function matchesCriteria(value, criteria) {
  // Simple equality check
  if (
    typeof criteria !== "object" ||
    criteria === null ||
    Array.isArray(criteria)
  ) {
    return value === criteria;
  }

  // Operator-based checks
  for (const [operator, compareValue] of Object.entries(criteria)) {
    switch (operator) {
      case "$eq":
        if (value !== compareValue) return false;
        break;

      case "$ne":
        if (value === compareValue) return false;
        break;

      case "$gt":
        if (!(value > compareValue)) return false;
        break;

      case "$gte":
        if (!(value >= compareValue)) return false;
        break;

      case "$lt":
        if (!(value < compareValue)) return false;
        break;

      case "$lte":
        if (!(value <= compareValue)) return false;
        break;

      case "$in":
        if (!Array.isArray(compareValue) || !compareValue.includes(value))
          return false;
        break;

      case "$nin":
        if (!Array.isArray(compareValue) || compareValue.includes(value))
          return false;
        break;

      case "$mod":
        // { $mod: [divisor, remainder] } - checks if value % divisor === remainder
        if (!Array.isArray(compareValue) || compareValue.length !== 2) {
          logger.warn("$mod requires [divisor, remainder] array");
          return false;
        }
        const [divisor, remainder] = compareValue;
        if (typeof divisor !== "number" || typeof remainder !== "number") {
          logger.warn("$mod divisor and remainder must be numbers");
          return false;
        }
        if (divisor === 0) {
          logger.warn("$mod divisor cannot be zero");
          return false;
        }
        if (typeof value !== "number") return false;
        if (value % divisor !== remainder) return false;
        break;

      default:
        logger.warn(`Unknown operator "${operator}"`);
        return false;
    }
  }

  return true;
}

/**
 * Check if game state matches all criteria
 * @param {Object} gameState - Current game state
 * @param {Object} criteria - Criteria object with key-value pairs
 * @returns {boolean}
 */
export function checkCriteria(gameState, criteria) {
  if (!criteria || typeof criteria !== "object") {
    return true; // No criteria means always match
  }

  for (const [key, value] of Object.entries(criteria)) {
    const stateValue = gameState[key];

    if (!matchesCriteria(stateValue, value)) {
      return false;
    }
  }

  return true;
}

/**
 * Check if criteria could still be met in the future given current game state
 * Used to skip prefetching assets whose criteria have already passed (e.g., in debug spawn mode)
 *
 * This is a simplified check: if criteria has $lt/$lte and we're past it, or $eq and we're not at it,
 * or $in with no future states, then it can never match again.
 *
 * @param {Object} currentState - Current game state
 * @param {Object} criteria - Criteria object with key-value pairs
 * @returns {boolean} True if criteria could still be met, false if it's impossible
 */
export function couldCriteriaStillMatch(currentState, criteria) {
  if (!criteria || typeof criteria !== "object") {
    return true; // No criteria means always could match
  }

  // Only check currentState - other criteria (flags, choices) could change
  if (!criteria.currentState) {
    return true; // No currentState criteria, assume could match
  }

  const currentStateValue = currentState.currentState;
  const value = criteria.currentState;

  // Simple equality - can match if we're at that state or haven't reached it yet
  // (states only increase, so if we're past it, it can never match)
  if (typeof value === "number") {
    return currentStateValue <= value;
  }

  // Operator-based checks
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    // Check if any operator prevents future matching
    for (const [operator, compareValue] of Object.entries(value)) {
      switch (operator) {
        case "$eq":
          // Can only match if we're at that exact state
          if (currentStateValue !== compareValue) return false;
          break;

        case "$lt":
          // Can only match if currentState < compareValue
          if (currentStateValue >= compareValue) return false;
          break;

        case "$lte":
          // Can only match if currentState <= compareValue
          if (currentStateValue > compareValue) return false;
          break;

        case "$in":
          // Can match if currentState is in array or any future state is in array
          if (Array.isArray(compareValue)) {
            if (!compareValue.includes(currentStateValue)) {
              // Check if any future state (>= current) is in array
              const hasFutureState = compareValue.some(
                (s) => s >= currentStateValue
              );
              if (!hasFutureState) return false;
            }
          }
          break;

        // $gt, $gte, $ne, $nin - could always match in future (states only increase)
        // No need to check these
      }
    }
  }

  return true; // Could still match
}

export default {
  matchesCriteria,
  checkCriteria,
  couldCriteriaStillMatch,
};
