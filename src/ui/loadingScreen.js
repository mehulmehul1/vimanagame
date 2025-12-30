/**
 * loadingScreen.js - ASSET LOADING PROGRESS OVERLAY
 * =============================================================================
 *
 * ROLE: DOM overlay showing loading progress before the game starts.
 * Tracks multiple loading tasks and displays aggregate progress.
 *
 * KEY RESPONSIBILITIES:
 * - Create DOM overlay with progress bar and SVG animation
 * - Track multiple named loading tasks
 * - Calculate and display aggregate progress
 * - Fade out and trigger START_SCREEN when complete
 *
 * TASK TRACKING:
 * - registerTask(name): Add a new loading task
 * - updateTask(name, progress): Update task progress (0-1)
 * - completeTask(name): Mark task as 100% complete
 *
 * =============================================================================
 */

import { Logger } from "../utils/logger.js";
import { GAME_STATES } from "../gameData.js";

export class LoadingScreen {
  constructor() {
    this.container = null;
    this.progressRect = null;
    this.logger = new Logger("LoadingScreen", false);
    this.progressText = null;
    this.loadingTasks = new Map(); // task name -> { progress: number } (0-1)
    this.isVisible = true;
    this.isComplete = false;
    this.gameManager = null; // Will be set after GameManager is created

    this.createUI();
  }

  /**
   * Set the game manager reference (called after GameManager is instantiated)
   * @param {GameManager} gameManager - Game manager instance
   */
  setGameManager(gameManager) {
    this.gameManager = gameManager;
  }

  async createUI() {
    // Create container
    this.container = document.createElement("div");
    this.container.id = "loading-screen";
    this.container.className = "loading-screen";

    // Create content wrapper
    const content = document.createElement("div");
    content.className = "loading-content";

    // Create loading title
    const title = document.createElement("div");
    title.className = "loading-title";
    title.textContent = "LOADING";
    content.appendChild(title);

    // Load and insert the SVG
    try {
      const response = await fetch("images/Loading.svg");
      const svgText = await response.text();
      const svgContainer = document.createElement("div");
      svgContainer.className = "loading-svg-container";
      svgContainer.innerHTML = svgText;
      content.appendChild(svgContainer);

      // Get reference to the progress rectangle
      this.progressRect = svgContainer.querySelector("#progressRect");
    } catch (error) {
      this.logger.error("Failed to load Loading.svg:", error);
    }

    // Create progress text
    this.progressText = document.createElement("div");
    this.progressText.className = "loading-progress-text";
    this.progressText.textContent = "0%";
    content.appendChild(this.progressText);

    // Assemble UI
    this.container.appendChild(content);

    // Add to document
    document.body.appendChild(this.container);
  }

  /**
   * Register a loading task (one asset)
   * @param {string} taskName - Unique name for the task
   */
  registerTask(taskName) {
    this.loadingTasks.set(taskName, { progress: 0 });
    this.updateProgress();
  }

  /**
   * Update progress for a specific task
   * @param {string} taskName - Name of the task
   * @param {number} progress - Progress value (0-1, where 1 = complete)
   */
  updateTask(taskName, progress) {
    const task = this.loadingTasks.get(taskName);
    if (task) {
      task.progress = Math.max(0, Math.min(1, progress));
      this.updateProgress();
    }
  }

  /**
   * Mark a task as complete
   * @param {string} taskName - Name of the task
   */
  completeTask(taskName) {
    const task = this.loadingTasks.get(taskName);
    if (task) {
      task.progress = 1;
      this.updateProgress();
    }
  }

  /**
   * Calculate and update overall progress
   */
  updateProgress() {
    if (this.loadingTasks.size === 0) {
      return;
    }

    let totalProgress = 0;
    let allTasksComplete = true;

    for (const task of this.loadingTasks.values()) {
      totalProgress += task.progress;
      if (task.progress < 1) {
        allTasksComplete = false;
      }
    }

    const averageProgress = totalProgress / this.loadingTasks.size;
    const progressPercent = averageProgress * 100;

    // Update UI - SVG progress rect width goes from 0 to 850
    if (this.progressRect) {
      const width = progressPercent * 8.5; // 850 / 100
      this.progressRect.setAttribute("width", width.toString());
    }
    if (this.progressText) {
      this.progressText.textContent = `${Math.round(progressPercent)}%`;
    }

    // Check if complete: ALL tasks must be individually at 100% progress
    if (allTasksComplete && !this.isComplete) {
      this.isComplete = true;
      this.handleLoadingComplete();
    }
  }

  /**
   * Set references to managers for deferred asset loading
   * @param {Object} managers - Object containing renderer and asset managers
   */
  setManagers(managers) {
    this.renderer = managers.renderer;
    this.musicManager = managers.musicManager;
    this.sfxManager = managers.sfxManager;
    this.dialogManager = managers.dialogManager;
    this.cameraAnimationManager = managers.cameraAnimationManager;
    this.videoManager = managers.videoManager;
  }

  /**
   * Handle loading completion - hide screen, fade in renderer, load deferred assets
   */
  handleLoadingComplete() {
    if (!this.isLoadingComplete()) return;

    const fadeDuration = 0.5;

    // Hide loading screen
    this.hide(fadeDuration);

    // Fade in renderer
    if (this.renderer && this.renderer.domElement) {
      this.renderer.domElement.style.transition = `opacity ${fadeDuration}s ease-in`;
      setTimeout(() => {
        this.renderer.domElement.style.opacity = "1";
      }, 100);
    }

    // Load deferred assets after loading screen hides
    setTimeout(() => {
      this.logger.log("Loading deferred assets...");
      if (this.musicManager) this.musicManager.loadDeferredTracks();
      if (this.sfxManager) this.sfxManager.loadDeferredSounds();
      if (this.dialogManager) this.dialogManager.loadDeferredDialogs();
      if (this.cameraAnimationManager)
        this.cameraAnimationManager.loadDeferredAnimations();
      if (this.gameManager) this.gameManager.loadDeferredSceneObjects();
      if (this.videoManager) this.videoManager.loadDeferredVideos();
    }, 600); // Start loading after fade completes
  }

  /**
   * Hide the loading screen with a fade-out animation
   * @param {number} duration - Fade duration in seconds (default 0.5)
   */
  hide(duration = 0.5) {
    if (!this.isVisible || !this.container) return;

    this.isVisible = false;
    this.container.style.transition = `opacity ${duration}s ease-out`;
    this.container.style.opacity = "0";

    // Transition game state from LOADING to START_SCREEN (only if currently LOADING)
    if (this.gameManager) {
      // Only transition to START_SCREEN if we're currently in LOADING state
      // This preserves debug spawn states
      if (this.gameManager.state.currentState === GAME_STATES.LOADING) {
        this.gameManager.setState({ currentState: GAME_STATES.START_SCREEN });
      }
    }

    // Remove from DOM after fade completes
    setTimeout(() => {
      if (this.container && this.container.parentNode) {
        this.container.parentNode.removeChild(this.container);
      }
    }, duration * 1000);
  }

  /**
   * Show the loading screen
   */
  show() {
    if (this.container) {
      this.container.style.opacity = "1";
      this.isVisible = true;
    }
  }

  /**
   * Check if loading is complete
   */
  isLoadingComplete() {
    return this.isComplete;
  }

  /**
   * Get current progress (0-100)
   */
  getProgress() {
    if (this.loadingTasks.size === 0) {
      return 0;
    }

    let totalProgress = 0;
    for (const task of this.loadingTasks.values()) {
      totalProgress += task.progress;
    }

    return (totalProgress / this.loadingTasks.size) * 100;
  }
}
