/**
 * optionsMenu.js - IN-GAME OPTIONS AND SETTINGS MENU
 * =============================================================================
 *
 * ROLE: Manages the options/settings menu accessible during gameplay.
 * Handles volume controls, DOF toggle, captions, and performance profiles.
 *
 * KEY RESPONSIBILITIES:
 * - Open/close with Escape key
 * - Music and SFX volume sliders
 * - Depth of Field toggle
 * - Captions enable/disable
 * - Performance profile selection (mobile/laptop/desktop/max)
 * - Save/load settings to localStorage
 * - Pause game when open
 *
 * PERFORMANCE PROFILES:
 * - mobile: 2M splat budget, merged zones
 * - laptop: 5M splat budget
 * - desktop: 8M splat budget
 * - max: Full quality
 *
 * =============================================================================
 */

import { Logger } from "../utils/logger.js";

class OptionsMenu {
  constructor(options = {}) {
    this.musicManager = options.musicManager || null;
    this.sfxManager = options.sfxManager || null;
    this.gameManager = options.gameManager || null;
    this.sparkRenderer = options.sparkRenderer || null;
    this.logger = new Logger("OptionsMenu", false);
    this.uiManager = options.uiManager || null;
    this.characterController = options.characterController || null;
    this.startScreen = options.startScreen || null;
    this.isOpen = false;

    // Settings with defaults
    // Performance profile will be set early based on isMobile, but default to "laptop" for desktop
    this.settings = {
      musicVolume: 0.6,
      sfxVolume: 0.5,
      dofEnabled: true,
      captionsEnabled: true,
      performanceProfile: "laptop", // "mobile", "laptop", or "max" (laptop is default)
      ...this.loadSettings(),
    };

    // Escape key tracking
    this.escapeKeyDownTime = null;

    // Create menu elements
    this.menuElement = this.createMenuHTML();
    this.bindEvents();
    this.bindKeyboardEvents();

    // Apply initial settings (includes updateUI)
    this.applySettings();

    // Update performance mode options based on platform
    this.updatePerformanceModeOptions();

    // Register with UI manager if available (will be registered later if not available yet)
    if (this.uiManager) {
      this.registerWithUIManager(this.uiManager);
    }
  }

  /**
   * Create the HTML structure for the options menu
   */
  createMenuHTML() {
    const menu = document.createElement("div");
    menu.id = "options-menu";
    menu.className = "options-menu hidden";

    menu.innerHTML = `
      <div class="options-overlay"></div>
      <div class="options-container">
        <div class="options-header">
          <h2 class="options-title">OPTIONS</h2>
          <button class="close-button" id="close-button" aria-label="Close">×</button>
        </div>
        
        <div class="options-content">
          <!-- Music Volume -->
          <div class="option-group">
            <label class="option-label" for="music-volume">
              Music Volume
              <span class="option-value" id="music-volume-value">60%</span>
            </label>
            <input 
              type="range" 
              id="music-volume" 
              class="option-slider"
              min="0" 
              max="100" 
              value="50"
            >
          </div>

          <!-- SFX Volume -->
          <div class="option-group">
            <label class="option-label" for="sfx-volume">
              SFX & Dialog Volume
              <span class="option-value" id="sfx-volume-value">50%</span>
            </label>
            <input 
              type="range" 
              id="sfx-volume" 
              class="option-slider"
              min="0" 
              max="100" 
              value="80"
            >
          </div>

          <!-- Captions Enable Checkbox -->
          <div class="option-group">
            <label class="option-label checkbox-label" for="captions-enabled">
              Captions
              <input 
                type="checkbox" 
                id="captions-enabled" 
                class="option-checkbox"
                checked
              >
            </label>
          </div>

          <!-- Performance Profile -->
          <div class="option-group">
            <label class="option-label" for="performance-profile">Performance Mode</label>
            <select 
              id="performance-profile" 
              class="performance-select"
            >
              <option value="mobile">Mobile</option>
              <option value="laptop" selected>Laptop</option>
              <option value="desktop">Desktop</option>
              <option value="max">Max</option>
            </select>
          </div>
        </div>
      </div>

      <!-- Performance Mode Change Confirmation Modal -->
      <div class="refresh-confirm-modal hidden" id="refresh-confirm-modal">
        <div class="refresh-confirm-overlay"></div>
        <div class="refresh-confirm-container">
          <h3 class="refresh-confirm-title">Performance Mode Change</h3>
          <p class="refresh-confirm-message">This action requires a refresh to load the correct assets.</p>
          <div class="refresh-confirm-buttons">
            <button class="refresh-confirm-button refresh-confirm-cancel" id="refresh-confirm-cancel">Cancel</button>
            <button class="refresh-confirm-button refresh-confirm-ok" id="refresh-confirm-ok">Refresh</button>
          </div>
        </div>
      </div>
    `;

    document.body.appendChild(menu);
    return menu;
  }

  /**
   * Bind keyboard event listeners (ESC key)
   */
  bindKeyboardEvents() {
    // Global escape key handler (only works when game is active, not during intro)
    // Track ESC key press time to distinguish between quick press (menu) and held (exit fullscreen)
    this.keydownHandler = (e) => {
      if (
        e.key === "Escape" &&
        (!this.startScreen || !this.startScreen.isActive)
      ) {
        // Record when ESC was first pressed (ignore repeat events)
        if (!e.repeat && this.escapeKeyDownTime === null) {
          this.escapeKeyDownTime = Date.now();
        }
      }
    };

    this.keyupHandler = (e) => {
      if (
        e.key === "Escape" &&
        (!this.startScreen || !this.startScreen.isActive)
      ) {
        e.preventDefault();

        // Check if refresh confirmation modal is open - close it first
        const refreshModal = document.getElementById("refresh-confirm-modal");
        if (refreshModal && !refreshModal.classList.contains("hidden")) {
          this.cancelRefresh();
          this.escapeKeyDownTime = null;
          return;
        }

        // Only toggle menu if ESC was pressed for less than 200ms (short press)
        if (this.escapeKeyDownTime !== null) {
          const pressDuration = Date.now() - this.escapeKeyDownTime;

          if (pressDuration < 200) {
            this.toggle();
          }

          this.escapeKeyDownTime = null;
        }
      }
    };

    window.addEventListener("keydown", this.keydownHandler);
    window.addEventListener("keyup", this.keyupHandler);
  }

  /**
   * Bind event listeners
   */
  bindEvents() {
    // Music volume slider
    const musicSlider = document.getElementById("music-volume");
    const musicValue = document.getElementById("music-volume-value");

    musicSlider.addEventListener("input", (e) => {
      const value = parseInt(e.target.value);
      musicValue.textContent = `${value}%`;
      musicSlider.style.setProperty("--value", `${value}%`);
      this.settings.musicVolume = value / 100;
      this.applyMusicVolume();
    });

    musicSlider.addEventListener("change", () => {
      this.saveSettings();
    });

    // SFX volume slider
    const sfxSlider = document.getElementById("sfx-volume");
    const sfxValue = document.getElementById("sfx-volume-value");

    sfxSlider.addEventListener("input", (e) => {
      const value = parseInt(e.target.value);
      sfxValue.textContent = `${value}%`;
      sfxSlider.style.setProperty("--value", `${value}%`);
      this.settings.sfxVolume = value / 100;
      this.applySfxVolume();
    });

    sfxSlider.addEventListener("change", () => {
      this.saveSettings();
    });

    // Captions Enable checkbox
    const captionsEnabledCheckbox = document.getElementById("captions-enabled");

    captionsEnabledCheckbox.addEventListener("change", (e) => {
      this.settings.captionsEnabled = e.target.checked;
      this.applyCaptions();
      this.saveSettings();
    });

    // Performance Profile dropdown
    const performanceSelect = document.getElementById("performance-profile");
    if (performanceSelect) {
      performanceSelect.addEventListener("change", (e) => {
        const newProfile = e.target.value;
        const oldProfile = this.settings.performanceProfile;

        // Only show confirmation if profile actually changed
        if (newProfile !== oldProfile) {
          this.showRefreshConfirmation(newProfile, oldProfile);
        }
      });
    }

    // Refresh confirmation modal buttons
    const refreshConfirmModal = document.getElementById(
      "refresh-confirm-modal"
    );
    const refreshConfirmOk = document.getElementById("refresh-confirm-ok");
    const refreshConfirmCancel = document.getElementById(
      "refresh-confirm-cancel"
    );
    const refreshConfirmOverlay = refreshConfirmModal?.querySelector(
      ".refresh-confirm-overlay"
    );

    if (refreshConfirmOk) {
      refreshConfirmOk.addEventListener("click", () => {
        this.confirmRefresh();
      });
    }

    if (refreshConfirmCancel) {
      refreshConfirmCancel.addEventListener("click", () => {
        this.cancelRefresh();
      });
    }

    if (refreshConfirmOverlay) {
      refreshConfirmOverlay.addEventListener("click", () => {
        this.cancelRefresh();
      });
    }

    // Store pending profile change
    this.pendingProfileChange = null;

    // Close button
    const closeButton = document.getElementById("close-button");
    if (closeButton) {
      closeButton.addEventListener("click", (e) => {
        e.stopPropagation();
        this.close();
      });
    } else {
      this.logger.error("Close button not found!");
    }

    // Click overlay to close
    this.menuElement
      .querySelector(".options-overlay")
      .addEventListener("click", () => {
        this.close();
      });
  }

  /**
   * Open the options menu
   */
  open() {
    if (this.isOpen) return;

    this.isOpen = true;

    // Use UI manager if available, otherwise handle directly
    if (this.uiManager) {
      this.uiManager.show("options-menu");
    } else {
      this.menuElement.classList.remove("hidden");
      // Pause game if game manager exists
      if (this.gameManager && this.gameManager.pause) {
        this.gameManager.pause();
      }
    }

    // Request pointer lock release
    if (document.pointerLockElement) {
      document.exitPointerLock();
    }

    // Update performance mode options (in case platform state changed)
    this.updatePerformanceModeOptions();

    // Update UI to reflect current settings
    this.updateUI();
  }

  /**
   * Close the options menu
   */
  close() {
    if (!this.isOpen) return;

    this.isOpen = false;

    // Use UI manager if available, otherwise handle directly
    if (this.uiManager) {
      this.uiManager.hide("options-menu");
    } else {
      this.menuElement.classList.add("hidden");
      // Resume game if game manager exists
      if (this.gameManager && this.gameManager.resume) {
        this.gameManager.resume();
      }
    }
  }

  /**
   * Toggle menu open/close
   */
  toggle() {
    if (this.isOpen) {
      this.close();
    } else {
      this.open();
    }
  }

  /**
   * Update UI elements to reflect current settings
   */
  updateUI() {
    const musicSlider = document.getElementById("music-volume");
    const musicValue = document.getElementById("music-volume-value");
    const sfxSlider = document.getElementById("sfx-volume");
    const sfxValue = document.getElementById("sfx-volume-value");
    const captionsEnabledCheckbox = document.getElementById("captions-enabled");
    const performanceSelect = document.getElementById("performance-profile");

    const musicPercent = Math.round(this.settings.musicVolume * 100);
    const sfxPercent = Math.round(this.settings.sfxVolume * 100);

    musicSlider.value = musicPercent;
    musicValue.textContent = `${musicPercent}%`;
    musicSlider.style.setProperty("--value", `${musicPercent}%`);

    sfxSlider.value = sfxPercent;
    sfxValue.textContent = `${sfxPercent}%`;
    sfxSlider.style.setProperty("--value", `${sfxPercent}%`);

    captionsEnabledCheckbox.checked = this.settings.captionsEnabled;

    // Update performance profile dropdown
    if (performanceSelect) {
      performanceSelect.value = this.settings.performanceProfile;
    }
  }

  /**
   * Apply music volume setting
   */
  applyMusicVolume() {
    if (this.musicManager && this.musicManager.setVolume) {
      this.musicManager.setVolume(this.settings.musicVolume, 0.1);
    }
  }

  /**
   * Apply SFX volume setting
   */
  applySfxVolume() {
    // Apply to all sound effects through SFX manager
    if (this.sfxManager) {
      this.sfxManager.setMasterVolume(this.settings.sfxVolume);
    }
  }

  /**
   * Apply depth of field settings
   */
  applyDepthOfField() {
    // Notify character controller of DoF enabled state
    if (this.characterController && this.characterController.setDofEnabled) {
      this.characterController.setDofEnabled(this.settings.dofEnabled);
    }
  }

  /**
   * Apply captions settings
   */
  applyCaptions() {
    // Notify game manager to pass captions setting to dialog manager
    if (this.gameManager && this.gameManager.dialogManager) {
      this.gameManager.dialogManager.setCaptionsEnabled(
        this.settings.captionsEnabled
      );
    }
  }

  /**
   * Apply performance profile settings
   */
  applyPerformanceProfile() {
    // Update gameManager state with performance profile
    if (this.gameManager) {
      this.gameManager.setState({
        performanceProfile: this.settings.performanceProfile,
      });
    }
  }

  /**
   * Apply all settings
   */
  applySettings() {
    this.applyMusicVolume();
    this.applySfxVolume();
    this.applyDepthOfField();
    this.applyCaptions();
    this.applyPerformanceProfile();
    this.updateUI();
  }

  /**
   * Save settings to localStorage
   */
  saveSettings() {
    try {
      localStorage.setItem("gameSettings", JSON.stringify(this.settings));
    } catch (e) {
      this.logger.warn("Failed to save settings:", e);
    }
  }

  /**
   * Load settings from localStorage
   */
  loadSettings() {
    try {
      const saved = localStorage.getItem("gameSettings");
      return saved ? JSON.parse(saved) : {};
    } catch (e) {
      this.logger.warn("Failed to load settings:", e);
      return {};
    }
  }

  /**
   * Get current settings
   */
  getSettings() {
    return { ...this.settings };
  }

  /**
   * Set performance profile early (before options menu is fully initialized)
   * Called from main.js right after platform detection
   * @param {string} profile - "mobile", "laptop", "desktop", or "max"
   */
  setPerformanceProfile(profile) {
    if (!["mobile", "laptop", "desktop", "max"].includes(profile)) {
      this.logger.warn(
        `Invalid performance profile: ${profile}, defaulting to "max"`
      );
      profile = "max";
    }
    this.settings.performanceProfile = profile;
    // Update gameManager state if available
    if (this.gameManager) {
      this.gameManager.setState({
        performanceProfile: profile,
      });
    }
    // Update UI if menu is already created
    if (this.menuElement) {
      const performanceSelect = document.getElementById("performance-profile");
      if (performanceSelect) {
        performanceSelect.value = profile;
      }
    }
  }

  /**
   * Register with UIManager (called after UIManager is created)
   * This allows options menu to be initialized early before UIManager exists
   * @param {UIManager} uiManager - The UIManager instance
   */
  registerWithUIManager(uiManager) {
    this.uiManager = uiManager;

    // Register the menu element
    this.uiManager.registerElement(
      "options-menu",
      this.menuElement,
      "PAUSE_MENU",
      {
        blocksInput: true,
        pausesGame: true,
      }
    );

    // Listen for UI manager events to sync isOpen state
    if (this.uiManager.gameManager) {
      this.uiManager.gameManager.on("ui:shown", (id) => {
        if (id === "options-menu") {
          this.isOpen = true;
          this.updateUI();
          // Request pointer lock release
          if (document.pointerLockElement) {
            document.exitPointerLock();
          }
        }
      });

      this.uiManager.gameManager.on("ui:hidden", (id) => {
        if (id === "options-menu") {
          this.isOpen = false;
        }
      });
    }
  }

  /**
   * Show refresh confirmation modal
   * @param {string} newProfile - The new performance profile
   * @param {string} oldProfile - The current performance profile
   */
  showRefreshConfirmation(newProfile, oldProfile) {
    this.pendingProfileChange = newProfile;
    const modal = document.getElementById("refresh-confirm-modal");
    if (modal) {
      modal.classList.remove("hidden");
    }
  }

  /**
   * Confirm refresh - save settings and reload page
   */
  confirmRefresh() {
    if (this.pendingProfileChange) {
      // Save the new performance profile
      this.settings.performanceProfile = this.pendingProfileChange;
      this.saveSettings();

      // Reload the page
      window.location.reload();
    }
  }

  /**
   * Cancel refresh - revert radio button and hide modal
   */
  cancelRefresh() {
    const modal = document.getElementById("refresh-confirm-modal");
    if (modal) {
      modal.classList.add("hidden");
    }

    // Revert dropdown to current setting
    const performanceSelect = document.getElementById("performance-profile");
    if (performanceSelect) {
      performanceSelect.value = this.settings.performanceProfile;
    }

    this.pendingProfileChange = null;
  }

  /**
   * Check if running on localhost or IP address (not a named domain)
   * @returns {boolean} True if on localhost or IP address
   */
  isLocalhostOrIP() {
    const hostname = window.location.hostname;
    return (
      hostname === "localhost" ||
      hostname === "127.0.0.1" ||
      hostname === "[::1]" ||
      /^\d+\.\d+\.\d+\.\d+$/.test(hostname) || // IPv4
      /^\[[0-9a-fA-F:]+\]$/.test(hostname) // IPv6 in brackets
    );
  }

  /**
   * Update performance mode dropdown options based on platform
   *
   * Behavior:
   * - Desktop/laptop users: All options enabled (including mobile, for testing)
   * - Mobile devices on production: Only "mobile" option enabled (prevents crashes)
   * - Mobile devices on localhost/IP: All options enabled (for developer testing)
   */
  updatePerformanceModeOptions() {
    const performanceSelect = document.getElementById("performance-profile");
    if (!performanceSelect) return;

    // Check if we're on mobile
    const isMobile =
      this.gameManager?.getState?.()?.isMobile ||
      "ontouchstart" in window ||
      navigator.maxTouchPoints > 0;

    // Check if we're on localhost or IP address
    const isLocalOrIP = this.isLocalhostOrIP();

    // Only restrict on actual mobile devices when not on localhost/IP
    // Desktop/laptop users (not detected as mobile) will have all options enabled
    if (isMobile && !isLocalOrIP) {
      const options = performanceSelect.querySelectorAll("option");
      options.forEach((option) => {
        if (option.value !== "mobile") {
          option.disabled = true;
        }
      });
      // If current selection is not mobile, force it to mobile
      if (performanceSelect.value !== "mobile") {
        this.settings.performanceProfile = "mobile";
        performanceSelect.value = "mobile";
        this.saveSettings();
      }
    } else {
      // Enable all options (desktop/laptop users, or mobile on localhost/IP)
      const options = performanceSelect.querySelectorAll("option");
      options.forEach((option) => {
        option.disabled = false;
      });
    }
  }

  /**
   * Clean up
   */
  destroy() {
    // Remove keyboard event listeners
    if (this.keydownHandler) {
      window.removeEventListener("keydown", this.keydownHandler);
    }
    if (this.keyupHandler) {
      window.removeEventListener("keyup", this.keyupHandler);
    }

    if (this.menuElement && this.menuElement.parentNode) {
      this.menuElement.parentNode.removeChild(this.menuElement);
    }
  }
}

export default OptionsMenu;
