/**
 * logger.js - CLASS-SPECIFIC DEBUG LOGGING UTILITY
 * =============================================================================
 *
 * ROLE: Provides per-class logging with enable/disable control. Allows
 * selective debug output without global console spam.
 *
 * KEY FEATURES:
 * - Per-instance debug flag
 * - Prefixed log output with class name
 * - log() only outputs when debug=true
 * - warn() and error() always output
 * - raw() outputs without prefix for copy-paste data
 *
 * USAGE:
 *   const logger = new Logger('MyManager', true);  // debug enabled
 *   logger.log('Debug message');   // [MyManager] Debug message
 *   logger.warn('Warning');        // [MyManager] Warning (always shown)
 *   logger.error('Error');         // [MyManager] Error (always shown)
 *
 * =============================================================================
 */

export class Logger {
  constructor(name = "App", debug = false) {
    this.name = name;
    this.debug = debug;
  }

  /**
   * Log message (only if debug is enabled)
   * @param {...any} args - Arguments to log
   */
  log(...args) {
    if (this.debug) {
      console.log(`[${this.name}]`, ...args);
    }
  }

  /**
   * Always log warnings (regardless of debug flag)
   * @param {...any} args - Arguments to log
   */
  warn(...args) {
    console.warn(`[${this.name}]`, ...args);
  }

  /**
   * Always log errors (regardless of debug flag)
   * @param {...any} args - Arguments to log
   */
  error(...args) {
    console.error(`[${this.name}]`, ...args);
  }

  /**
   * Log message without prefix (only if debug is enabled)
   * Useful for structured data that should be copy-paste friendly
   * @param {...any} args - Arguments to log
   */
  logRaw(...args) {
    if (this.debug) {
      console.log(...args);
    }
  }

  /**
   * Enable or disable debug logging
   * @param {boolean} enabled - Whether to enable debug logging
   */
  setDebug(enabled) {
    this.debug = enabled;
  }
}

export default Logger;
