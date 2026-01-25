/**
 * logger.js - CLASS-SPECIFIC DEBUG LOGGING UTILITY
 * =============================================================================
 *
 * Provides per-class logging with enable/disable control.
 * =============================================================================
 */

export class Logger {
  constructor(name = "App", debug = false) {
    this.name = name;
    this.debug = debug;
  }

  /**
   * Log message (only if debug is enabled)
   */
  log(...args) {
    if (this.debug) {
      console.log(`[${this.name}]`, ...args);
    }
  }

  /**
   * Always log warnings (regardless of debug flag)
   */
  warn(...args) {
    console.warn(`[${this.name}]`, ...args);
  }

  /**
   * Always log errors (regardless of debug flag)
   */
  error(...args) {
    console.error(`[${this.name}]`, ...args);
  }

  /**
   * Log message without prefix (only if debug is enabled)
   */
  logRaw(...args) {
    if (this.debug) {
      console.log(...args);
    }
  }

  /**
   * Enable or disable debug logging
   */
  setDebug(enabled) {
    this.debug = enabled;
  }
}

export default Logger;
