/**
 * Logger utility for class-specific debug logging
 *
 * Usage:
 * import { Logger } from './utils/logger.js';
 *
 * class MyManager {
 *   constructor() {
 *     this.logger = new Logger('MyManager', false); // false = debug off by default
 *     this.logger.log('This only appears if debug is enabled');
 *   }
 * }
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
