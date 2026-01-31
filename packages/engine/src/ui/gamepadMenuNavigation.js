/**
 * gamepadMenuNavigation.js - GAMEPAD MENU NAVIGATION UTILITY
 * =============================================================================
 *
 * ROLE: Shared utility for gamepad-based menu navigation. Handles D-pad/stick
 * up/down and A button confirm for consistent menu behavior.
 *
 * KEY RESPONSIBILITIES:
 * - Poll gamepad for navigation input
 * - Edge detection for button presses
 * - Cooldown to prevent rapid navigation
 * - Play navigation sounds via SFXManager
 *
 * USAGE:
 * Used by StartScreen, DialogChoiceUI, and other menu systems.
 *
 * =============================================================================
 */

export class GamepadMenuNavigation {
  constructor(options = {}) {
    this.inputManager = options.inputManager || null;
    this.sfxManager = options.sfxManager || null;

    // Callbacks
    this.onNavigateUp = options.onNavigateUp || null;
    this.onNavigateDown = options.onNavigateDown || null;
    this.onConfirm = options.onConfirm || null;

    // State tracking for edge detection
    this.lastNavigationUp = false;
    this.lastNavigationDown = false;
    this.lastConfirmButton = false;

    // Cooldown to prevent rapid navigation
    this.navigationCooldown = 0;
    this.navigationDelay = options.navigationDelay || 0.2; // 200ms between inputs

    // Sound effect index for keystroke cycling
    this.keystrokeIndex = 0;
  }

  /**
   * Update gamepad navigation state (call in update loop)
   * @param {number} dt - Delta time in seconds
   */
  update(dt) {
    if (!this.inputManager) return;

    const gamepad = this.inputManager.getGamepad();
    if (!gamepad) return;

    // Update cooldown timer
    if (this.navigationCooldown > 0) {
      this.navigationCooldown -= dt;
    }

    // D-pad up/down or left stick Y axis for navigation
    const dpadUp = gamepad.buttons[12]?.pressed || false;
    const dpadDown = gamepad.buttons[13]?.pressed || false;
    const leftStickY = gamepad.axes[1] || 0;
    const buttonA = gamepad.buttons[0]?.pressed || false;

    // Detect navigation up (D-pad up or left stick up)
    const navigationUp = dpadUp || leftStickY < -0.5;
    // Detect navigation down (D-pad down or left stick down)
    const navigationDown = dpadDown || leftStickY > 0.5;

    // Navigate up (detect press, not hold)
    if (
      navigationUp &&
      !this.lastNavigationUp &&
      this.navigationCooldown <= 0
    ) {
      if (this.onNavigateUp) {
        this.onNavigateUp();
      }
      this.navigationCooldown = this.navigationDelay;
      this.playKeystrokeSound();
    }

    // Navigate down (detect press, not hold)
    if (
      navigationDown &&
      !this.lastNavigationDown &&
      this.navigationCooldown <= 0
    ) {
      if (this.onNavigateDown) {
        this.onNavigateDown();
      }
      this.navigationCooldown = this.navigationDelay;
      this.playKeystrokeSound();
    }

    // Confirm selection (A button - detect press, not hold)
    if (buttonA && !this.lastConfirmButton) {
      if (this.onConfirm) {
        this.onConfirm();
      }
      this.playReturnSound();
    }

    // Store previous state for edge detection
    this.lastNavigationUp = navigationUp;
    this.lastNavigationDown = navigationDown;
    this.lastConfirmButton = buttonA;
  }

  /**
   * Play typewriter keystroke sound
   */
  playKeystrokeSound() {
    if (this.sfxManager) {
      const soundId = `typewriter-keystroke-0${this.keystrokeIndex}`;
      this.sfxManager.play(soundId);
      // Cycle through 0-3
      this.keystrokeIndex = (this.keystrokeIndex + 1) % 4;
    }
  }

  /**
   * Play typewriter return sound
   */
  playReturnSound() {
    if (this.sfxManager) {
      this.sfxManager.play("typewriter-return");
    }
  }

  /**
   * Reset navigation state (call when menu is hidden/destroyed)
   */
  reset() {
    this.lastNavigationUp = false;
    this.lastNavigationDown = false;
    this.lastConfirmButton = false;
    this.navigationCooldown = 0;
    this.keystrokeIndex = 0;
  }

  /**
   * Set the input manager
   * @param {InputManager} inputManager
   */
  setInputManager(inputManager) {
    this.inputManager = inputManager;
  }

  /**
   * Set the SFX manager
   * @param {SfxManager} sfxManager
   */
  setSfxManager(sfxManager) {
    this.sfxManager = sfxManager;
  }
}
