import { Logger } from "../utils/logger.js";

/**
 * LoadingScreen - Minimal loading screen that tracks asset loading progress
 * Shows before main.js initialization and hides when all assets are loaded
 */

export class LoadingScreen {
  constructor() {
    this.container = null;
    this.progressRect = null;
    this.logger = new Logger("LoadingScreen", false);
    this.progressText = null;
    this.loadingTasks = new Map(); // task name -> { loaded: number, total: number }
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
   * Register a loading task
   * @param {string} taskName - Unique name for the task
   * @param {number} total - Total number of items to load (default 1)
   */
  registerTask(taskName, total = 1) {
    this.loadingTasks.set(taskName, { loaded: 0, total });
    this.updateProgress();
  }

  /**
   * Update progress for a specific task
   * @param {string} taskName - Name of the task
   * @param {number} loaded - Number of items loaded
   * @param {number} total - Total items (optional, updates total if provided)
   */
  updateTask(taskName, loaded, total = null) {
    const task = this.loadingTasks.get(taskName);
    if (task) {
      task.loaded = loaded;
      if (total !== null) {
        task.total = total;
      }
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
      task.loaded = task.total;
      this.updateProgress();
    }
  }

  /**
   * Calculate and update overall progress
   */
  updateProgress() {
    let totalLoaded = 0;
    let totalItems = 0;

    for (const task of this.loadingTasks.values()) {
      totalLoaded += task.loaded;
      totalItems += task.total;
    }

    const progress = totalItems > 0 ? (totalLoaded / totalItems) * 100 : 0;

    // Update UI - SVG progress rect width goes from 0 to 850
    if (this.progressRect) {
      const width = (progress / 100) * 850;
      this.progressRect.setAttribute("width", width.toString());
    }
    if (this.progressText) {
      this.progressText.textContent = `${Math.round(progress)}%`;
    }

    // Check if complete
    if (progress >= 100 && !this.isComplete) {
      this.isComplete = true;
    }
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
      // Import GAME_STATES dynamically to avoid circular dependencies
      import("../gameData.js").then(({ GAME_STATES }) => {
        // Only transition to START_SCREEN if we're currently in LOADING state
        // This preserves debug spawn states
        if (this.gameManager.state.currentState === GAME_STATES.LOADING) {
          this.gameManager.setState({ currentState: GAME_STATES.START_SCREEN });
        }
      });
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
    let totalLoaded = 0;
    let totalItems = 0;

    for (const task of this.loadingTasks.values()) {
      totalLoaded += task.loaded;
      totalItems += task.total;
    }

    return totalItems > 0 ? (totalLoaded / totalItems) * 100 : 0;
  }
}
