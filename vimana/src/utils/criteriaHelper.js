/**
 * criteriaHelper.js - MONGODB-STYLE CRITERIA MATCHING FOR GAME STATE
 * =============================================================================
 *
 * Provides declarative criteria matching for game state conditions.
 * Copied from shadowczar engine for vimana's use.
 *
 * =============================================================================
 */

import { Logger } from "./logger.js";

const logger = new Logger("CriteriaHelper", false);

/**
 * Check if a single value matches a criteria definition
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
 * Check if criteria could still be met in the future
 */
export function couldCriteriaStillMatch(currentState, criteria) {
  if (!criteria || typeof criteria !== "object") {
    return true;
  }

  if (!criteria.currentState) {
    return true;
  }

  const currentStateValue = currentState.currentState;
  const value = criteria.currentState;

  if (typeof value === "number") {
    return currentStateValue <= value;
  }

  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    for (const [operator, compareValue] of Object.entries(value)) {
      switch (operator) {
        case "$eq":
          if (currentStateValue !== compareValue) return false;
          break;

        case "$lt":
          if (currentStateValue >= compareValue) return false;
          break;

        case "$lte":
          if (currentStateValue > compareValue) return false;
          break;

        case "$in":
          if (Array.isArray(compareValue)) {
            if (!compareValue.includes(currentStateValue)) {
              const hasFutureState = compareValue.some(
                (s) => s >= currentStateValue
              );
              if (!hasFutureState) return false;
            }
          }
          break;
      }
    }
  }

  return true;
}

export default {
  matchesCriteria,
  checkCriteria,
  couldCriteriaStillMatch,
};
