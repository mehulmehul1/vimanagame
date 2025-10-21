/**
 * FullscreenButton - Manages fullscreen toggle button
 *
 * Features:
 * - Bottom-left corner placement
 * - Toggles fullscreen mode
 * - Integrates with UIManager
 * - Handles fullscreen change events
 */

import { Logger } from "../utils/logger.js";

export class FullscreenButton {
  constructor(options = {}) {
    this.uiManager = options.uiManager || null;
    this.gameManager = options.gameManager || null;
    this.config = options.config || {};
    this.logger = new Logger("FullscreenButton", false);

    // Create the button element
    this.createButton();

    // Listen for fullscreen changes to update button state
    this.bindFullscreenEvents();
  }

  /**
   * Check if fullscreen is supported (from game state)
   * @returns {boolean}
   */
  isFullscreenSupported() {
    if (this.gameManager) {
      const state = this.gameManager.getState();
      return state.isFullscreenSupported !== false; // Default to true if not set
    }
    return true;
  }

  /**
   * Create the fullscreen button element
   */
  createButton() {
    // Create button container
    this.button = document.createElement("div");
    this.button.id = this.config.id || "fullscreen-button";
    this.button.classList.add("fullscreen-button");

    // Create button image
    this.image = document.createElement("img");
    this.image.src = this.config.image || "/images/FullScreen.svg";
    this.image.alt = "Toggle Fullscreen";

    this.button.appendChild(this.image);

    // Apply CSS variables from config (fallbacks handled in CSS)
    const position = this.config.position || {};
    const size = this.config.size || {};
    const style = this.config.style || {};
    const varMap = {
      "--fb-bottom": position.bottom,
      "--fb-right": position.right,
      "--fb-width": size.width,
      "--fb-height": size.height,
      "--fb-cursor": style.cursor,
      "--fb-opacity": style.opacity,
      "--fb-transition": style.transition,
      "--fb-pointer-events": style.pointerEvents,
      "--fb-z-index": style.zIndex,
    };
    Object.entries(varMap).forEach(([key, value]) => {
      if (value !== undefined) this.button.style.setProperty(key, value);
    });

    // Add touch feedback (visual response on mobile)
    this.button.addEventListener("touchstart", (e) => {
      e.preventDefault(); // Prevent default touch behavior
      this.button.classList.add("is-touching");
    });

    this.button.addEventListener("touchcancel", () => {
      this.button.classList.remove("is-touching");
    });

    // Add click handler for mouse input
    this.button.addEventListener("click", (e) => {
      // Prevent if this was triggered by a touch (touchend also fires click)
      if (e.pointerType === "touch") return;
      this.toggleFullscreen();
    });

    // Add touch handler for mobile devices
    this.button.addEventListener("touchend", (e) => {
      e.preventDefault(); // Prevent click event from also firing
      // Reset visual feedback
      this.button.classList.remove("is-touching");
      this.toggleFullscreen();
    });

    // Add to document
    document.body.appendChild(this.button);

    // Register with UI manager if available
    if (this.uiManager) {
      this.uiManager.registerElement(
        this.config.id || "fullscreen-button",
        this.button,
        this.config.layer || "GAME_HUD",
        {
          blocksInput:
            this.config.blocksInput !== undefined
              ? this.config.blocksInput
              : false,
          pausesGame: this.config.pausesGame || false,
        }
      );
    }

    // Hide button if fullscreen is not supported (e.g., iOS)
    if (!this.isFullscreenSupported()) {
      this.button.classList.add("is-hidden");
      this.logger.log(
        "Fullscreen API not supported on this device, button hidden"
      );
    }
  }

  /**
   * Toggle fullscreen mode
   */
  toggleFullscreen() {
    // Don't attempt if not supported
    if (!this.isFullscreenSupported()) {
      this.logger.log("Fullscreen not supported on this device");
      return;
    }

    if (!document.fullscreenElement) {
      // Enter fullscreen
      document.documentElement.requestFullscreen().catch((err) => {
        this.logger.warn("Error attempting to enable fullscreen:", err);
      });
    } else {
      // Exit fullscreen
      document.exitFullscreen();
    }
  }

  /**
   * Bind fullscreen change events
   */
  bindFullscreenEvents() {
    document.addEventListener("fullscreenchange", () => {
      this.updateButtonState();
    });

    document.addEventListener("webkitfullscreenchange", () => {
      this.updateButtonState();
    });

    document.addEventListener("mozfullscreenchange", () => {
      this.updateButtonState();
    });

    document.addEventListener("MSFullscreenChange", () => {
      this.updateButtonState();
    });

    // Listen for resize events to detect F11 fullscreen
    window.addEventListener("resize", () => {
      this.updateButtonState();
    });

    // Check initial state
    this.updateButtonState();
  }

  /**
   * Check if browser is in fullscreen mode (including F11)
   */
  isInFullscreen() {
    // Check Fullscreen API first
    if (document.fullscreenElement) {
      return true;
    }

    // Check for browser-level fullscreen (F11)
    // Compare window dimensions to screen dimensions
    const isWindowFullscreen =
      window.innerHeight === screen.height &&
      window.innerWidth === screen.width;

    return isWindowFullscreen;
  }

  /**
   * Update button appearance based on fullscreen state
   */
  updateButtonState() {
    const isFullscreen = this.isInFullscreen();

    // Update game manager state
    if (this.gameManager) {
      this.gameManager.setState({ isFullscreen });
    }

    // Hide button when in fullscreen, show when not
    if (isFullscreen) {
      this.hide();
    } else {
      this.show();
    }

    // Update tooltip
    if (isFullscreen) {
      this.button.title = "Exit Fullscreen (ESC or F11)";
    } else {
      this.button.title = "Enter Fullscreen";
    }
  }

  /**
   * Show the button
   */
  show() {
    // Don't show if fullscreen is not supported
    if (!this.isFullscreenSupported()) {
      return;
    }
    this.button.classList.remove("is-hidden");
    if (this.uiManager) {
      this.uiManager.show(this.config.id || "fullscreen-button");
    }
  }

  /**
   * Hide the button
   */
  hide() {
    this.button.classList.add("is-hidden");
    if (this.uiManager) {
      this.uiManager.hide(this.config.id || "fullscreen-button");
    }
  }

  /**
   * Set UI manager reference
   * @param {UIManager} uiManager
   */
  setUIManager(uiManager) {
    this.uiManager = uiManager;
  }

  /**
   * Clean up
   */
  destroy() {
    if (this.button && this.button.parentNode) {
      this.button.parentNode.removeChild(this.button);
    }

    if (this.uiManager) {
      this.uiManager.unregisterElement(this.config.id || "fullscreen-button");
    }
  }
}

export default FullscreenButton;
