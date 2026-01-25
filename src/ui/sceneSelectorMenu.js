/**
 * sceneSelectorMenu.js - DEBUG SCENE SELECTOR MENU
 * =============================================================================
 *
 * ROLE: Provides a debug menu for jumping directly to any game state/scene.
 * Useful for testing and development without manually editing URL parameters.
 *
 * KEY RESPONSIBILITIES:
 * - Display all available game states organized by category
 * - Jump to selected state by reloading with gameState URL parameter
 * - Toggle with F5 key or button
 * - Show current state highlight
 *
 * USAGE:
 *   Press F5 to toggle the scene selector menu
 *   Click any state to jump to it
 *
 * =============================================================================
 */

import { GAME_STATES } from "../gameData.js";
import { Logger } from "../utils/logger.js";

// State categories for organization
const STATE_CATEGORIES = {
  "Intro & Title": [
    "LOADING",
    "START_SCREEN",
    "INTRO",
    "TITLE_SEQUENCE",
    "TITLE_SEQUENCE_COMPLETE",
  ],
  "Exterior - Plaza": [
    "CAT_DIALOG_CHOICE",
    "NEAR_RADIO",
  ],
  "Phone Booth Scene": [
    "PHONE_BOOTH_RINGING",
    "ANSWERED_PHONE",
    "DIALOG_CHOICE_1",
  ],
  "Drive By Scene": [
    "DRIVE_BY_PREAMBLE",
    "DRIVE_BY",
    "POST_DRIVE_BY",
  ],
  "Office Entry": [
    "DOORS_CLOSE",
    "ENTERING_OFFICE",
    "OFFICE_INTERIOR",
  ],
  "Office Phone": [
    "OFFICE_PHONE_ANSWERED",
    "PRE_VIEWMASTER",
  ],
  "ViewMaster Sequence": [
    "VIEWMASTER",
    "VIEWMASTER_COLOR",
    "VIEWMASTER_DISSOLVE",
    "VIEWMASTER_DIALOG",
    "VIEWMASTER_HELL",
    "POST_VIEWMASTER",
  ],
  "Cat Dialog 2": [
    "CAT_DIALOG_CHOICE_2",
  ],
  "Edison Scene": [
    "PRE_EDISON",
    "EDISON",
    "DIALOG_CHOICE_2",
  ],
  "Czar Struggle": [
    "CZAR_STRUGGLE",
    "SHOULDER_TAP",
    "PUNCH_OUT",
    "FALLEN",
    "LIGHTS_OUT",
  ],
  "Waking Up": [
    "WAKING_UP",
    "SHADOW_AMPLIFICATIONS",
    "CAT_SAVE",
  ],
  "Drawing Minigame": [
    "CURSOR",
    "CURSOR_FINAL",
    "POST_CURSOR",
  ],
  "Outro Sequences": [
    "OUTRO",
    "OUTRO_LECLAIRE",
    "OUTRO_CAT",
    "OUTRO_CZAR",
    "OUTRO_CREDITS",
    "OUTRO_MOVIE",
    "GAME_OVER",
  ],
};

class SceneSelectorMenu {
  constructor(options = {}) {
    this.gameManager = options.gameManager || null;
    this.logger = new Logger("SceneSelectorMenu", false);
    this.uiManager = options.uiManager || null;
    this.isOpen = false;
    this.currentState = null;

    // Create menu elements
    this.menuElement = this.createMenuHTML();
    this.bindEvents();
    this.bindKeyboardEvents();

    // Update current state display
    this.updateCurrentState();

    // Listen for state changes to update display
    if (this.gameManager) {
      this.gameManager.on("state:changed", (newState) => {
        this.currentState = newState.currentState;
        this.updateCurrentStateHighlight();
      });

      // Get initial state
      const initialState = this.gameManager.getState();
      if (initialState) {
        this.currentState = initialState.currentState;
      }
    }

    // Register with UI manager if available
    if (this.uiManager) {
      this.registerWithUIManager(this.uiManager);
    }

    this.logger.log("SceneSelectorMenu initialized");
  }

  /**
   * Create the HTML structure for the scene selector menu
   */
  createMenuHTML() {
    const menu = document.createElement("div");
    menu.id = "scene-selector-menu";
    menu.className = "scene-selector-menu hidden";

    // Build category sections
    let categoriesHTML = "";
    for (const [category, states] of Object.entries(STATE_CATEGORIES)) {
      const statesHTML = states
        .map((state) => {
          const stateValue = GAME_STATES[state];
          return `
            <button class="scene-item" data-state="${state}" data-value="${stateValue}">
              <span class="scene-name">${this.formatStateName(state)}</span>
              <span class="scene-value">${stateValue}</span>
            </button>
          `;
        })
        .join("");

      categoriesHTML += `
        <div class="scene-category">
          <h3 class="category-title">${category}</h3>
          <div class="scene-list">${statesHTML}</div>
        </div>
      `;
    }

    menu.innerHTML = `
      <div class="scene-selector-overlay"></div>
      <div class="scene-selector-container">
        <div class="scene-selector-header">
          <h2 class="scene-selector-title">SCENE SELECTOR</h2>
          <div class="current-state-info">
            <span class="current-state-label">Current:</span>
            <span class="current-state-value" id="current-state-display">Unknown</span>
          </div>
          <button class="close-button" id="scene-selector-close" aria-label="Close">×</button>
        </div>

        <div class="scene-selector-content">
          <div class="scene-selector-info">
            <p>Click any scene to jump directly to it. The page will reload.</p>
            <p class="scene-selector-tip">Press <kbd>F5</kbd> to toggle this menu</p>
          </div>
          ${categoriesHTML}
        </div>
      </div>
    `;

    document.body.appendChild(menu);
    return menu;
  }

  /**
   * Format state name for display (convert SNAKE_CASE to Title Case)
   */
  formatStateName(stateName) {
    return stateName
      .split("_")
      .map((word) => word.charAt(0) + word.slice(1).toLowerCase())
      .join(" ");
  }

  /**
   * Bind keyboard event listeners (F5 key)
   */
  bindKeyboardEvents() {
    this.keydownHandler = (e) => {
      // F5 key to toggle menu
      if (e.key === "F5") {
        e.preventDefault();
        this.toggle();
      }
    };

    window.addEventListener("keydown", this.keydownHandler);
  }

  /**
   * Bind event listeners
   */
  bindEvents() {
    // Scene item click handlers
    const sceneItems = this.menuElement.querySelectorAll(".scene-item");
    sceneItems.forEach((item) => {
      item.addEventListener("click", (e) => {
        const stateName = item.getAttribute("data-state");
        this.jumpToState(stateName);
      });
    });

    // Close button
    const closeButton = document.getElementById("scene-selector-close");
    if (closeButton) {
      closeButton.addEventListener("click", (e) => {
        e.stopPropagation();
        this.close();
      });
    }

    // Click overlay to close
    this.menuElement
      .querySelector(".scene-selector-overlay")
      ?.addEventListener("click", () => {
        this.close();
      });
  }

  /**
   * Jump to a specific game state by reloading with URL parameter
   */
  jumpToState(stateName) {
    this.logger.log(`Jumping to state: ${stateName}`);

    // Build new URL with gameState parameter
    const url = new URL(window.location.href);
    url.searchParams.set("gameState", stateName);

    // Reload page with new state
    window.location.href = url.toString();
  }

  /**
   * Update current state display
   */
  updateCurrentState() {
    const displayEl = document.getElementById("current-state-display");
    if (!displayEl) return;

    // Try to find state name from value
    let stateName = "Unknown";
    if (this.currentState !== null) {
      for (const [name, value] of Object.entries(GAME_STATES)) {
        if (value === this.currentState) {
          stateName = this.formatStateName(name);
          break;
        }
      }
    }

    displayEl.textContent = `${stateName} (${this.currentState})`;
    this.updateCurrentStateHighlight();
  }

  /**
   * Update highlight for current state in the list
   */
  updateCurrentStateHighlight() {
    // Remove all existing highlights
    const items = this.menuElement.querySelectorAll(".scene-item");
    items.forEach((item) => item.classList.remove("current-state"));

    // Add highlight to current state
    if (this.currentState !== null) {
      const currentItem = this.menuElement.querySelector(
        `.scene-item[data-value="${this.currentState}"]`
      );
      if (currentItem) {
        currentItem.classList.add("current-state");
        // Scroll into view if needed
        currentItem.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    }
  }

  /**
   * Open the scene selector menu
   */
  open() {
    if (this.isOpen) return;

    this.isOpen = true;

    if (this.uiManager) {
      this.uiManager.show("scene-selector-menu");
    } else {
      this.menuElement.classList.remove("hidden");
      if (this.gameManager && this.gameManager.pause) {
        this.gameManager.pause();
      }
    }

    // Update current state display
    this.updateCurrentState();

    // Release pointer lock
    if (document.pointerLockElement) {
      document.exitPointerLock();
    }
  }

  /**
   * Close the scene selector menu
   */
  close() {
    if (!this.isOpen) return;

    this.isOpen = false;

    if (this.uiManager) {
      this.uiManager.hide("scene-selector-menu");
    } else {
      this.menuElement.classList.add("hidden");
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
   * Register with UIManager
   */
  registerWithUIManager(uiManager) {
    this.uiManager = uiManager;

    this.uiManager.registerElement(
      "scene-selector-menu",
      this.menuElement,
      "DEBUG",
      {
        blocksInput: true,
        pausesGame: true,
      }
    );

    // Listen for UI manager events
    if (this.uiManager.gameManager) {
      this.uiManager.gameManager.on("ui:shown", (id) => {
        if (id === "scene-selector-menu") {
          this.isOpen = true;
          this.updateCurrentState();
          if (document.pointerLockElement) {
            document.exitPointerLock();
          }
        }
      });

      this.uiManager.gameManager.on("ui:hidden", (id) => {
        if (id === "scene-selector-menu") {
          this.isOpen = false;
        }
      });
    }
  }

  /**
   * Clean up
   */
  destroy() {
    if (this.keydownHandler) {
      window.removeEventListener("keydown", this.keydownHandler);
    }

    if (this.menuElement && this.menuElement.parentNode) {
      this.menuElement.parentNode.removeChild(this.menuElement);
    }
  }
}

export default SceneSelectorMenu;
