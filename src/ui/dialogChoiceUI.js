/**
 * DialogChoiceUI - Handles multiple choice dialog responses
 *
 * Features:
 * - Display choice buttons at bottom of screen
 * - Art deco black and white styling with La Porsche font
 * - Emit events when choices are made
 * - Track choice history
 */

import { GamepadMenuNavigation } from "./gamepadMenuNavigation.js";
import { Logger } from "../utils/logger.js";

class DialogChoiceUI {
  constructor(options = {}) {
    this.gameManager = options.gameManager || null;
    this.dialogManager = options.dialogManager || null;
    this.sfxManager = options.sfxManager || null;
    this.inputManager = options.inputManager || null;
    this.logger = new Logger("DialogChoiceUI", false);
    this.container = null;
    this.currentChoices = null;
    this.isVisible = false;
    this.selectedIndex = 0;
    this.choiceButtons = [];
    this.keystrokeIndex = 0; // Track which keystroke sound to play next (0-3)

    this.createUI();
    this.applyStyles();
    this.setupKeyboardListeners();

    // Setup gamepad navigation
    this.gamepadNav = new GamepadMenuNavigation({
      inputManager: this.inputManager,
      sfxManager: this.sfxManager,
      onNavigateUp: () => this.moveSelection(-1),
      onNavigateDown: () => this.moveSelection(1),
      onConfirm: () => this.confirmSelection(),
    });

    // Setup state listener if gameManager is available
    if (this.gameManager) {
      this.setupStateListener();
    }
  }

  /**
   * Set the game manager (called after initialization if not passed in constructor)
   * @param {GameManager} gameManager
   */
  setGameManager(gameManager) {
    this.gameManager = gameManager;
    this.setupStateListener();
  }

  /**
   * Set up state change listener for auto-showing dialog choices
   */
  setupStateListener() {
    if (!this.gameManager) return;

    // Track shown choices for "once" functionality
    this.shownChoices = new Set();

    // Import getChoicesForState
    import("../dialogChoiceData.js").then(({ getChoicesForState }) => {
      // Listen for state changes
      this.gameManager.on("state:changed", (newState, oldState) => {
        const matchingChoices = getChoicesForState(newState, this.shownChoices);

        // If there are matching choices for the new state
        if (matchingChoices.length > 0) {
          const choiceData = matchingChoices[0];

          this.logger.log(`Auto-showing choices "${choiceData.id}"`);

          // Track that this choice has been shown
          this.shownChoices.add(choiceData.id);

          // Show the choices
          this.showChoices(choiceData);

          // Emit event for tracking
          this.gameManager.emit(
            "dialogChoice:trigger",
            choiceData.id,
            choiceData
          );
        }
      });

      this.logger.log("State listener registered");
    });
  }

  /**
   * Create the choice UI elements
   */
  createUI() {
    // Main container
    this.container = document.createElement("div");
    this.container.id = "dialog-choices";
    this.container.className = "dialog-choices-container";
    this.container.style.display = "none";

    // Title/prompt (optional)
    this.promptElement = document.createElement("div");
    this.promptElement.className = "dialog-choices-prompt";
    this.container.appendChild(this.promptElement);

    // Choices container
    this.choicesElement = document.createElement("div");
    this.choicesElement.className = "dialog-choices-list";
    this.container.appendChild(this.choicesElement);

    document.body.appendChild(this.container);
  }

  /**
   * Apply art deco styling
   */
  applyStyles() {
    const style = document.createElement("style");
    style.textContent = `
      @keyframes slideUpFadeIn {
        from {
          transform: translateY(100%);
          opacity: 0;
        }
        to {
          transform: translateY(0);
          opacity: 1;
        }
      }

      .dialog-choices-container {
        position: fixed;
        bottom: 5%;
        left: 5%;
        width: auto;
        max-width: 500px;
        min-width: 300px;
        z-index: 2000;
        pointer-events: auto;
        background: rgba(255, 255, 255, 0.6);
        border: 4px solid black;
        box-shadow: 
          0 8px 16px rgba(0, 0, 0, 0.7),
          inset 0 0 0 2px rgba(255, 255, 255, 0.6),
          inset 0 0 0 6px black;
        padding: 4px;
        animation: slideUpFadeIn 0.4s ease-out;
      }

      .dialog-choices-prompt {
        font-family: 'PXCountryTypewriter', Arial, sans-serif;
        font-size: 32px;
        color: rgba(0, 0, 0, 0.8);
        text-align: left;
        padding: 8px 12px;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 4px;
      }

      .dialog-choices-list {
        display: flex;
        flex-direction: column;
        gap: 0;
      }

      .dialog-choice-button {
        font-family: 'PXCountryTypewriter', Arial, sans-serif;
        font-size: 24px;
        padding: 10px 16px;
        background: transparent;
        color: rgba(0, 0, 0, 0.5);
        border: none;
        cursor: pointer;
        transition: all 0.15s ease;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-align: left;
        width: 100%;
      }

      .dialog-choice-button:last-child {
        border-bottom: none;
      }

      .dialog-choice-button.selected {
        background: rgba(0, 0, 0, 0.85);
        color: white;
      }

      .dialog-choice-button:hover:not(.selected) {
        background: rgba(0, 0, 0, 0.1);
        color: rgba(0, 0, 0, 0.7);
      }

      @media (max-width: 768px) {
        .dialog-choices-container {
          left: 50%;
          right: auto;
          transform: translateX(-50%);
          max-width: 90vw;
          min-width: 280px;
          width: auto;
        }
        
        @keyframes slideUpFadeInMobile {
          from {
            transform: translateX(-50%) translateY(100%);
            opacity: 0;
          }
          to {
            transform: translateX(-50%) translateY(0);
            opacity: 1;
          }
        }
        
        .dialog-choices-container {
          animation: slideUpFadeInMobile 0.4s ease-out;
        }
        
        .dialog-choice-button {
          font-size: 22px;
          padding: 10px 14px;
        }
        
        .dialog-choices-prompt {
          font-size: 28px;
          padding: 8px 10px;
        }
      }
      
      @media (max-width: 480px) {
        .dialog-choices-container {
          max-width: 95vw;
          min-width: 260px;
          bottom: 10%;
        }
        
        .dialog-choice-button {
          font-size: 20px;
          padding: 10px 14px;
        }
        
        .dialog-choices-prompt {
          font-size: 26px;
          padding: 8px 10px;
        }
      }
    `;
    document.head.appendChild(style);
  }

  /**
   * Setup keyboard listeners for arrow keys and enter
   */
  setupKeyboardListeners() {
    this.keyboardHandler = (event) => {
      if (!this.isVisible) return;

      switch (event.key) {
        case "ArrowUp":
          event.preventDefault();
          this.moveSelection(-1);
          break;
        case "ArrowDown":
          event.preventDefault();
          this.moveSelection(1);
          break;
        case "Enter":
        case " ":
          event.preventDefault();
          this.confirmSelection();
          break;
      }
    };

    window.addEventListener("keydown", this.keyboardHandler);
    this.setupMouseListeners();
  }

  /**
   * Setup mouse listeners for scroll wheel and clicks
   */
  setupMouseListeners() {
    this.wheelHandler = (event) => {
      if (!this.isVisible) return;

      event.preventDefault();

      // deltaY > 0 means scrolling down, deltaY < 0 means scrolling up
      if (event.deltaY > 0) {
        this.moveSelection(1); // Scroll down = move selection down
      } else if (event.deltaY < 0) {
        this.moveSelection(-1); // Scroll up = move selection up
      }
    };

    // Handle clicks when pointer is locked (won't hit button elements)
    this.clickHandler = (event) => {
      if (!this.isVisible) return;

      // Check if pointer is locked
      const isPointerLocked = document.pointerLockElement !== null;

      if (isPointerLocked) {
        // When pointer locked, clicking confirms the currently selected choice
        event.preventDefault();
        this.confirmSelection();
      }
      // If pointer is not locked, the individual button click handlers will fire
    };

    window.addEventListener("wheel", this.wheelHandler, { passive: false });
    window.addEventListener("click", this.clickHandler);
  }

  /**
   * Move selection up or down
   * @param {number} direction - -1 for up, 1 for down
   */
  moveSelection(direction) {
    if (!this.currentChoices || this.choiceButtons.length === 0) return;

    // Remove selected class from current button
    this.choiceButtons[this.selectedIndex].classList.remove("selected");

    // Update selected index
    this.selectedIndex += direction;

    // Wrap around
    if (this.selectedIndex < 0) {
      this.selectedIndex = this.choiceButtons.length - 1;
    } else if (this.selectedIndex >= this.choiceButtons.length) {
      this.selectedIndex = 0;
    }

    // Add selected class to new button
    this.choiceButtons[this.selectedIndex].classList.add("selected");

    // Play typewriter keystroke sound
    if (this.sfxManager) {
      const soundId = `typewriter-keystroke-0${this.keystrokeIndex}`;
      this.sfxManager.play(soundId);
      // Cycle through 0-3
      this.keystrokeIndex = (this.keystrokeIndex + 1) % 4;
    }

    this.logger.log(`Selected option ${this.selectedIndex}`);
  }

  /**
   * Confirm the current selection
   */
  confirmSelection() {
    if (!this.currentChoices || this.choiceButtons.length === 0) return;

    // Play typewriter return sound
    if (this.sfxManager) {
      this.sfxManager.play("typewriter-return");
    }

    const selectedChoice = this.currentChoices.choices[this.selectedIndex];
    this.selectChoice(selectedChoice, this.currentChoices);
  }

  /**
   * Show choices to the player
   * @param {Object} choiceData - Dialog choice data
   * @param {string} choiceData.prompt - Optional prompt text
   * @param {Array} choiceData.choices - Array of choice objects
   */
  showChoices(choiceData) {
    if (!choiceData || !choiceData.choices || choiceData.choices.length === 0) {
      this.logger.warn("No choices provided");
      return;
    }

    this.currentChoices = choiceData;
    this.isVisible = true;
    this.selectedIndex = 0;
    this.choiceButtons = [];
    this.keystrokeIndex = 0; // Reset keystroke cycle when showing new choices

    // Set prompt if provided
    if (choiceData.prompt) {
      this.promptElement.textContent = choiceData.prompt;
      this.promptElement.style.display = "block";
    } else {
      this.promptElement.style.display = "none";
    }

    // Clear existing choices
    this.choicesElement.innerHTML = "";

    // Create buttons for each choice
    choiceData.choices.forEach((choice, index) => {
      const button = document.createElement("button");
      button.className = "dialog-choice-button";
      if (index === 0) {
        button.classList.add("selected");
      }
      button.textContent = choice.text;
      button.dataset.choiceResponseType = choice.responseType;
      button.dataset.choiceIndex = index;

      button.addEventListener("click", () => {
        // Play typewriter return sound on click
        if (this.sfxManager) {
          this.sfxManager.play("typewriter-return");
        }
        this.selectedIndex = index;
        this.selectChoice(choice, choiceData);
      });

      button.addEventListener("mouseenter", () => {
        // Update selection when hovering
        this.choiceButtons[this.selectedIndex].classList.remove("selected");
        this.selectedIndex = index;
        button.classList.add("selected");

        // Play typewriter keystroke sound on hover
        if (this.sfxManager) {
          const soundId = `typewriter-keystroke-0${this.keystrokeIndex}`;
          this.sfxManager.play(soundId);
          // Cycle through 0-3
          this.keystrokeIndex = (this.keystrokeIndex + 1) % 4;
        }
      });

      this.choicesElement.appendChild(button);
      this.choiceButtons.push(button);
    });

    // Show container
    this.container.style.display = "block";

    // Fade out touch joysticks on mobile
    if (this.inputManager) {
      this.inputManager.fadeOutTouchControls();
    }

    this.logger.log("Showing choices", choiceData);
  }

  /**
   * Handle choice selection
   * @param {Object} choice - Selected choice object
   * @param {Object} choiceData - Full choice data
   */
  selectChoice(choice, choiceData) {
    this.logger.log("Choice selected", choice);

    // Hide choices
    this.hide();

    // Build state update combining choice responseType and onSelect callback
    const stateUpdate = {};

    // Add choice responseType to state update
    if (choiceData.stateKey) {
      stateUpdate[choiceData.stateKey] = choice.responseType;
    }

    // Get additional state updates from onSelect callback
    if (choice.onSelect && typeof choice.onSelect === "function") {
      try {
        const additionalUpdates = choice.onSelect(this.gameManager, choice);
        if (additionalUpdates && typeof additionalUpdates === "object") {
          Object.assign(stateUpdate, additionalUpdates);
        }
      } catch (error) {
        this.logger.error("Error in onSelect callback", error);
      }
    }

    // Apply all state updates at once (prevents triggering autoplay multiple times)
    if (Object.keys(stateUpdate).length > 0 && this.gameManager) {
      this.logger.log("Applying state updates:", stateUpdate);
      this.gameManager.setState(stateUpdate);
    }

    // Play response dialog if specified
    if (choice.responseDialog && this.dialogManager) {
      this.logger.log("Playing response dialog", choice.responseDialog);
      this.dialogManager.playDialog(choice.responseDialog);
    }

    // Call global onChoiceSelected callback if provided
    if (
      choiceData.onChoiceSelected &&
      typeof choiceData.onChoiceSelected === "function"
    ) {
      try {
        choiceData.onChoiceSelected(this.gameManager, choice);
      } catch (error) {
        this.logger.error("Error in onChoiceSelected callback", error);
      }
    }
  }

  /**
   * Update gamepad navigation (call in main update loop)
   * @param {number} dt - Delta time in seconds
   */
  update(dt) {
    if (this.isVisible && this.gamepadNav) {
      this.gamepadNav.update(dt);
    }
  }

  /**
   * Hide the choice UI
   */
  hide() {
    this.container.style.display = "none";
    this.isVisible = false;
    this.currentChoices = null;
    this.selectedIndex = 0;
    this.choiceButtons = [];
    this.choicesElement.innerHTML = "";

    // Reset gamepad navigation state
    if (this.gamepadNav) {
      this.gamepadNav.reset();
    }

    // Fade in touch joysticks on mobile (use requestAnimationFrame to ensure DOM update completes)
    if (this.inputManager) {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          this.inputManager.fadeInTouchControls();
        });
      });
    }
  }

  /**
   * Check if choices are currently visible
   * @returns {boolean}
   */
  isShowingChoices() {
    return this.isVisible;
  }

  /**
   * Destroy the UI
   */
  destroy() {
    if (this.keyboardHandler) {
      window.removeEventListener("keydown", this.keyboardHandler);
      this.keyboardHandler = null;
    }

    if (this.wheelHandler) {
      window.removeEventListener("wheel", this.wheelHandler);
      this.wheelHandler = null;
    }

    if (this.clickHandler) {
      window.removeEventListener("click", this.clickHandler);
      this.clickHandler = null;
    }

    if (this.container && this.container.parentNode) {
      this.container.parentNode.removeChild(this.container);
    }
  }
}

export default DialogChoiceUI;
