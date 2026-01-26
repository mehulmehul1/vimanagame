# UIManager - First Principles Guide

## Overview

The **UIManager** is the central orchestration system for all user interface elements in the game. It manages the lifecycle of UI screens (loading, start, options, dialogs), handles transitions between states, coordinates with game state changes, and ensures responsive UI updates. Rather than having each screen manage itself independently, the UIManager provides a single source of truth for what's visible and when.

Think of the UIManager as the **"stage director"** of the UI - just as a theater director decides which actors appear on stage and when, the UIManager decides which UI screens are shown, how they transition, and how they respond to game events.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create smooth, intentional UI flow that never breaks immersion. Every transition should feel purposeful, every screen should appear at the right moment, and players should never wait or wonder what's happening.

**Why Centralized UI Management?**
- **Consistent Flow**: Same transition logic across all screens
- **State Synchronization**: UI always matches game state
- **Performance**: Only render what's visible
- **Debugging**: Single place to see all UI state
- **Accessibility**: Global keyboard/navigation handling

**Player Experience Flow**:
```
Launch Game
    ↓
Loading Screen → "The game is preparing" → Anticipation
    ↓ (fade)
Start Screen → "Welcome, ready to play?" → Agency
    ↓ (selection)
Game World → UI hidden → Immersion
    ↓ (pause/event)
Dialog/Menu → "Make a choice" → Engagement
    ↓ (selection)
Return to Game → Seamless resume → Flow maintained
```

### Design Principles

**1. Single Screen at a Time**
Only one primary screen should be active (loading, start, options, etc.). This prevents:
- Visual clutter
- Input conflicts (which screen gets the click?)
- Player confusion

**2. Smooth Transitions**
Never hard-cut between screens. Always use:
- Fade out old screen
- Fade in new screen
- Or slide/wipe for thematic transitions

**3. Responsive to Game State**
UI should react immediately to game events:
- Pause triggers pause menu
- Dialog triggers dialog UI
- Loading triggers loading screen

**4. Performance-Aware**
Hide/disable what's not visible:
- Invisible screens don't update
- Off-screen DOM elements don't render
- Particle effects pause when UI visible

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the UIManager, you should know:
- **DOM manipulation** - Creating, showing, hiding HTML elements
- **CSS transitions** - Smooth visual changes
- **Event-driven architecture** - Subscribing to and emitting events
- **State management** - Tracking what's active/visible
- **z-index layering** - Controlling element stacking order

### Core Architecture

```
UI MANAGER ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│                      UI MANAGER                         │
│  - Screen registry (all known screens)                  │
│  - Active screen tracking                               │
│  - Transition management                                │
│  - Event coordination                                   │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   SCREENS     │  │  TRANSITIONS  │  │   EVENTS     │
│  - Loading    │  │  - Fade       │  │  - gameState │
│  - Start      │  │  - Slide      │  │  - dialog    │
│  - Options    │  │  - Wipe       │  │  - pause     │
│  - Dialog     │  │  - Zoom       │  │  - loading   │
└───────────────┘  └───────────────┘  └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   RENDERER    │
                    │  - DOM update │
                    │  - CSS apply  │
                    └───────────────┘
```

### UIManager Class

```javascript
class UIManager {
  constructor(options = {}) {
    this.gameManager = options.gameManager;
    this.container = options.container || document.body;
    this.logger = options.logger || console;

    // Screen registry
    this.screens = new Map();  // name -> screen instance
    this.screenStack = [];     // History for navigation back

    // Current state
    this.currentScreen = null;
    this.transitioning = false;
    this.transitionDuration = 300;  // ms

    // UI state
    this.state = {
      isPaused: false,
      isInGame: false,
      isLoading: false,
      hasDialog: false
    };

    // Transition types
    this.TransitionType = {
      NONE: 'none',
      FADE: 'fade',
      SLIDE_LEFT: 'slideLeft',
      SLIDE_RIGHT: 'slideRight',
      SLIDE_UP: 'slideUp',
      SLIDE_DOWN: 'slideDown',
      ZOOM_IN: 'zoomIn',
      ZOOM_OUT: 'zoomOut'
    };

    // Setup
    this.setupContainer();
    this.setupEventListeners();
  }

  /**
   * Set up main UI container
   */
  setupContainer() {
    // Create main UI container if it doesn't exist
    if (!this.uiContainer) {
      this.uiContainer = document.createElement('div');
      this.uiContainer.id = 'ui-container';
      this.uiContainer.className = 'ui-container';
      this.container.appendChild(this.uiContainer);
    }

    // Create screen container
    if (!this.screenContainer) {
      this.screenContainer = document.createElement('div');
      this.screenContainer.id = 'screen-container';
      this.screenContainer.className = 'screen-container';
      this.uiContainer.appendChild(this.screenContainer);
    }

    // Create overlay container (for dialogs, toasts, etc.)
    if (!this.overlayContainer) {
      this.overlayContainer = document.createElement('div');
      this.overlayContainer.id = 'overlay-container';
      this.overlayContainer.className = 'overlay-container';
      this.uiContainer.appendChild(this.overlayContainer);
    }

    // Create HUD container (in-game UI)
    if (!this.hudContainer) {
      this.hudContainer = document.createElement('div');
      this.hudContainer.id = 'hud-container';
      this.hudContainer.className = 'hud-container';
      this.uiContainer.appendChild(this.hudContainer);
    }
  }

  /**
   * Register a screen
   */
  registerScreen(name, screenInstance) {
    if (this.screens.has(name)) {
      this.logger.warn(`Screen "${name}" already registered, overwriting`);
    }

    this.screens.set(name, {
      instance: screenInstance,
      element: screenInstance.element,
      isOverlay: screenInstance.isOverlay || false
    });

    // Add screen element to appropriate container
    if (screenInstance.isOverlay) {
      this.overlayContainer.appendChild(screenInstance.element);
    } else {
      this.screenContainer.appendChild(screenInstance.element);
    }

    // Initialize screen
    if (screenInstance.initialize) {
      screenInstance.initialize();
    }

    this.logger.log(`Registered screen: ${name}`);
  }

  /**
   * Unregister a screen
   */
  unregisterScreen(name) {
    const screen = this.screens.get(name);
    if (!screen) {
      this.logger.warn(`Screen "${name}" not found`);
      return;
    }

    // Remove element
    screen.element.remove();

    // Cleanup screen
    if (screen.instance.destroy) {
      screen.instance.destroy();
    }

    this.screens.delete(name);
    this.logger.log(`Unregistered screen: ${name}`);
  }

  /**
   * Show a screen (with optional transition)
   */
  async showScreen(screenName, options = {}) {
    const {
      transition = this.TransitionType.FADE,
      data = null,
      replace = false,      // Don't add to stack if true
      hideCurrent = true   // Hide current screen first
    } = options;

    // Check if screen exists
    const screen = this.screens.get(screenName);
    if (!screen) {
      this.logger.error(`Screen "${screenName}" not found`);
      return false;
    }

    // Don't show if already showing (unless forced)
    if (this.currentScreen === screenName && !options.force) {
      this.logger.log(`Screen "${screenName}" already showing`);
      return true;
    }

    // Wait for any ongoing transition
    if (this.transitioning) {
      await this.waitForTransition();
    }

    this.transitioning = true;

    // Hide current screen first
    if (hideCurrent && this.currentScreen) {
      await this.hideScreen(this.currentScreen, { transition: 'none' });
    }

    // Add to screen stack
    if (!replace && this.currentScreen && !screen.isOverlay) {
      this.screenStack.push(this.currentScreen);
    }

    // Prepare the new screen
    const screenElement = screen.element;
    screenElement.style.display = '';

    // Call screen's onShow
    if (screen.instance.onShow) {
      screen.instance.onShow(data);
    }

    // Apply transition
    await this.applyTransition(screenElement, 'in', transition);

    // Update state
    if (!screen.isOverlay) {
      this.currentScreen = screenName;
    }

    // Update UI state flags
    this.updateUIState();

    this.transitioning = false;
    this.logger.log(`Showing screen: ${screenName}`);

    return true;
  }

  /**
   * Hide a screen (with optional transition)
   */
  async hideScreen(screenName, options = {}) {
    const {
      transition = this.TransitionType.FADE,
      remove = false  // Remove from DOM when hidden
    } = options;

    const screen = this.screens.get(screenName);
    if (!screen) {
      this.logger.warn(`Cannot hide screen "${screenName}": not found`);
      return false;
    }

    // Apply exit transition
    await this.applyTransition(screen.element, 'out', transition);

    // Hide element
    screen.element.style.display = 'none';

    // Call screen's onHide
    if (screen.instance.onHide) {
      screen.instance.onHide();
    }

    // Update current screen if this was it
    if (this.currentScreen === screenName) {
      // Go back in stack if available
      if (this.screenStack.length > 0) {
        const previousScreen = this.screenStack.pop();
        await this.showScreen(previousScreen, {
          transition: 'none',
          hideCurrent: false
        });
      } else {
        this.currentScreen = null;
      }
    }

    this.updateUIState();
    this.logger.log(`Hid screen: ${screenName}`);

    return true;
  }

  /**
   * Apply transition to an element
   */
  async applyTransition(element, direction, type) {
    if (type === this.TransitionType.NONE) {
      return Promise.resolve();
    }

    return new Promise(resolve => {
      // Reset element state
      element.style.transition = '';
      element.style.opacity = '';

      // Force reflow
      void element.offsetWidth;

      // Set initial state based on direction
      if (direction === 'in') {
        this.setTransitionInitialState(element, type);
      } else {
        // For exit, element should already be visible
        element.style.opacity = '1';
        element.style.transform = '';
      }

      // Force reflow
      void element.offsetWidth;

      // Apply transition
      element.style.transition = `all ${this.transitionDuration}ms ease`;

      if (direction === 'in') {
        // Animate to final state
        element.style.opacity = '1';
        element.style.transform = 'translate(0, 0) scale(1)';
      } else {
        // Animate to exit state
        this.setTransitionExitState(element, type);
      }

      // Resolve after transition
      setTimeout(() => {
        element.style.transition = '';
        resolve();
      }, this.transitionDuration);
    });
  }

  /**
   * Set initial state for transition (for entry)
   */
  setTransitionInitialState(element, type) {
    switch (type) {
      case this.TransitionType.FADE:
        element.style.opacity = '0';
        break;

      case this.TransitionType.SLIDE_LEFT:
        element.style.opacity = '0';
        element.style.transform = 'translateX(100%)';
        break;

      case this.TransitionType.SLIDE_RIGHT:
        element.style.opacity = '0';
        element.style.transform = 'translateX(-100%)';
        break;

      case this.TransitionType.SLIDE_UP:
        element.style.opacity = '0';
        element.style.transform = 'translateY(100%)';
        break;

      case this.TransitionType.SLIDE_DOWN:
        element.style.opacity = '0';
        element.style.transform = 'translateY(-100%)';
        break;

      case this.TransitionType.ZOOM_IN:
        element.style.opacity = '0';
        element.style.transform = 'scale(0.8)';
        break;

      case this.TransitionType.ZOOM_OUT:
        element.style.opacity = '0';
        element.style.transform = 'scale(1.2)';
        break;
    }
  }

  /**
   * Set exit state for transition
   */
  setTransitionExitState(element, type) {
    switch (type) {
      case this.TransitionType.FADE:
        element.style.opacity = '0';
        break;

      case this.TransitionType.SLIDE_LEFT:
        element.style.opacity = '0';
        element.style.transform = 'translateX(-100%)';
        break;

      case this.TransitionType.SLIDE_RIGHT:
        element.style.opacity = '0';
        element.style.transform = 'translateX(100%)';
        break;

      case this.TransitionType.SLIDE_UP:
        element.style.opacity = '0';
        element.style.transform = 'translateY(-100%)';
        break;

      case this.TransitionType.SLIDE_DOWN:
        element.style.opacity = '0';
        element.style.transform = 'translateY(100%)';
        break;

      case this.TransitionType.ZOOM_IN:
        element.style.opacity = '0';
        element.style.transform = 'scale(1.2)';
        break;

      case this.TransitionType.ZOOM_OUT:
        element.style.opacity = '0';
        element.style.transform = 'scale(0.8)';
        break;
    }
  }

  /**
   * Go back to previous screen
   */
  async goBack() {
    if (this.screenStack.length === 0) {
      this.logger.warn('No previous screen to go back to');
      return false;
    }

    const previousScreen = this.screenStack.pop();
    await this.showScreen(previousScreen, {
      transition: this.TransitionType.SLIDE_RIGHT,
      hideCurrent: true
    });

    return true;
  }

  /**
   * Show loading screen
   */
  async showLoading(message = 'Loading...') {
    this.state.isLoading = true;
    return this.showScreen('loading', { data: { message } });
  }

  /**
   * Hide loading screen
   */
  async hideLoading() {
    this.state.isLoading = false;
    return this.hideScreen('loading');
  }

  /**
   * Show dialog (overlay)
   */
  async showDialog(dialogConfig) {
    const dialogScreen = this.screens.get('dialog');
    if (!dialogScreen) {
      this.logger.error('Dialog screen not registered');
      return;
    }

    this.state.hasDialog = true;

    // Configure dialog
    if (dialogScreen.instance.configure) {
      dialogScreen.instance.configure(dialogConfig);
    }

    return this.showScreen('dialog', {
      transition: this.TransitionType.ZOOM_IN
    });
  }

  /**
   * Hide dialog
   */
  async hideDialog() {
    this.state.hasDialog = false;
    return this.hideScreen('dialog');
  }

  /**
   * Toggle pause menu
   */
  async togglePause() {
    if (this.state.isPaused) {
      return this.resume();
    } else {
      return this.pause();
    }
  }

  /**
   * Pause game and show pause menu
   */
  async pause() {
    if (this.state.isPaused) return;

    this.state.isPaused = true;
    this.gameManager.emit('game:paused');

    return this.showScreen('pause', {
      transition: this.TransitionType.SLIDE_UP
    });
  }

  /**
   * Resume game and hide pause menu
   */
  async resume() {
    if (!this.state.isPaused) return;

    this.state.isPaused = false;
    this.gameManager.emit('game:resumed');

    return this.hideScreen('pause', {
      transition: this.TransitionType.SLIDE_DOWN
    });
  }

  /**
   * Update UI state flags
   */
  updateUIState() {
    // Update container classes based on state
    this.uiContainer.classList.toggle('is-paused', this.state.isPaused);
    this.uiContainer.classList.toggle('is-loading', this.state.isLoading);
    this.uiContainer.classList.toggle('has-dialog', this.state.hasDialog);
    this.uiContainer.classList.toggle('in-game', this.state.isInGame);

    // Emit state change event
    this.gameManager.emit('ui:stateChanged', { ...this.state });
  }

  /**
   * Set in-game state
   */
  setInGame(inGame) {
    this.state.isInGame = inGame;

    if (inGame) {
      // Hide all screens, show HUD
      if (this.currentScreen) {
        this.hideScreen(this.currentScreen, { transition: 'none' });
      }
      this.hudContainer.style.display = '';
    } else {
      // Hide HUD
      this.hudContainer.style.display = 'none';
    }

    this.updateUIState();
  }

  /**
   * Wait for current transition to complete
   */
  async waitForTransition() {
    while (this.transitioning) {
      await new Promise(resolve => setTimeout(resolve, 50));
    }
  }

  /**
   * Set up event listeners
   */
  setupEventListeners() {
    // Game state events
    this.gameManager.on('game:paused', () => {
      this.updateUIState();
    });

    this.gameManager.on('game:resumed', () => {
      this.updateUIState();
    });

    this.gameManager.on('game:loadingStarted', () => {
      this.showLoading();
    });

    this.gameManager.on('game:loadingComplete', () => {
      this.hideLoading();
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => this.onKeyDown(e));
  }

  /**
   * Handle keyboard input
   */
  onKeyDown(event) {
    // Escape key behavior
    if (event.key === 'Escape') {
      if (this.state.hasDialog) {
        this.hideDialog();
      } else if (this.state.isPaused) {
        this.resume();
      } else if (this.state.isInGame) {
        this.pause();
      }
    }

    // Pause key (P)
    if (event.key === 'p' && this.state.isInGame && !this.state.hasDialog) {
      this.togglePause();
    }
  }

  /**
   * Update loop (for animated UI elements)
   */
  update(dt) {
    // Update current screen
    if (this.currentScreen) {
      const screen = this.screens.get(this.currentScreen);
      if (screen && screen.instance.update) {
        screen.instance.update(dt);
      }
    }

    // Update visible overlays
    for (const [name, screen] of this.screens) {
      if (screen.isOverlay && screen.instance.update) {
        screen.instance.update(dt);
      }
    }
  }

  /**
   * Get current screen
   */
  getCurrentScreen() {
    return this.currentScreen;
  }

  /**
   * Check if a screen is registered
   */
  hasScreen(name) {
    return this.screens.has(name);
  }

  /**
   * Get screen instance
   */
  getScreen(name) {
    const screen = this.screens.get(name);
    return screen ? screen.instance : null;
  }

  /**
   * Clean up
   */
  destroy() {
    // Destroy all screens
    for (const [name, screen] of this.screens) {
      if (screen.instance.destroy) {
        screen.instance.destroy();
      }
    }

    // Remove event listeners
    document.removeEventListener('keydown', this.onKeyDown);

    // Clear containers
    this.uiContainer.remove();
  }
}

export default UIManager;
```

### Base Screen Class

```javascript
/**
 * Base class for all UI screens
 */
class UIScreen {
  constructor(options = {}) {
    this.name = options.name || 'screen';
    this.uiManager = options.uiManager;
    this.gameManager = options.gameManager;
    this.isOverlay = options.isOverlay || false;

    // Create screen element
    this.element = this.createElement();
    this.element.className = `screen screen-${this.name}`;
    this.element.style.display = 'none';
  }

  /**
   * Create screen DOM element (override in subclasses)
   */
  createElement() {
    const div = document.createElement('div');
    div.innerHTML = `
      <div class="screen-content">
        <h2>${this.name}</h2>
        <p>Override createElement() in your screen class</p>
      </div>
    `;
    return div;
  }

  /**
   * Initialize screen (called when registered)
   */
  initialize() {
    // Override in subclass
  }

  /**
   * Called when screen is shown
   */
  onShow(data) {
    // Override in subclass
    // data: any data passed to showScreen()
  }

  /**
   * Called when screen is hidden
   */
  onHide() {
    // Override in subclass
  }

  /**
   * Update loop (called every frame while visible)
   */
  update(dt) {
    // Override in subclass for animated screens
  }

  /**
   * Clean up
   */
  destroy() {
    // Override in subclass
    this.element.remove();
  }
}

export { UIScreen };
```

---

## 📝 How To Build A Screen Like This

### Step 1: Extend the Base Screen Class

```javascript
class MyCustomScreen extends UIScreen {
  constructor(options = {}) {
    super({
      ...options,
      name: 'myCustom',
      isOverlay: false  // or true for dialogs/popups
    });

    // Screen-specific state
    this.selectedOption = 0;
    this.options = ['Option 1', 'Option 2', 'Option 3'];
  }

  createElement() {
    const div = document.createElement('div');
    div.className = 'screen-my-custom';
    div.innerHTML = `
      <div class="screen-content">
        <h1>My Custom Screen</h1>
        <div class="options-list">
          ${this.options.map((opt, i) => `
            <button class="option-btn" data-index="${i}">${opt}</button>
          `).join('')}
        </div>
        <button class="back-btn">Back</button>
      </div>
    `;

    // Store references to elements
    this.optionButtons = div.querySelectorAll('.option-btn');
    this.backButton = div.querySelector('.back-btn');

    return div;
  }

  initialize() {
    // Setup button listeners
    this.optionButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const index = parseInt(btn.dataset.index);
        this.onOptionSelected(index);
      });
    });

    this.backButton.addEventListener('click', () => {
      this.uiManager.goBack();
    });
  }

  onShow(data) {
    // Called when screen appears
    console.log('MyCustomScreen shown with data:', data);

    // Update selected option highlight
    this.updateSelection();
  }

  onHide() {
    // Called when screen disappears
    console.log('MyCustomScreen hidden');
  }

  onOptionSelected(index) {
    this.selectedOption = index;
    this.updateSelection();

    // Do something with selection
    console.log('Selected:', this.options[index]);
  }

  updateSelection() {
    this.optionButtons.forEach((btn, i) => {
      btn.classList.toggle('selected', i === this.selectedOption);
    });
  }

  destroy() {
    // Clean up listeners
    this.optionButtons.forEach(btn => {
      btn.replaceWith(btn.cloneNode(true));
    });
    this.backButton.replaceWith(this.backButton.cloneNode(true));

    super.destroy();
  }
}
```

### Step 2: Register the Screen

```javascript
// In your game initialization
const uiManager = new UIManager({
  gameManager,
  container: document.body
});

// Create and register screen
const myScreen = new MyCustomScreen({
  uiManager,
  gameManager
});

uiManager.registerScreen('myCustom', myScreen);
```

### Step 3: Show the Screen

```javascript
// Show with default fade transition
uiManager.showScreen('myCustom');

// Show with specific transition and data
uiManager.showScreen('myCustom', {
  transition: UIManager.TransitionType.SLIDE_LEFT,
  data: { someData: 'value' }
});
```

---

## 🔧 Variations For Your Game

### Modal Dialog System

```javascript
class DialogScreen extends UIScreen {
  constructor(options) {
    super({ ...options, name: 'dialog', isOverlay: true });
    this.callback = null;
  }

  createElement() {
    const div = document.createElement('div');
    div.className = 'dialog-screen';
    div.innerHTML = `
      <div class="dialog-container">
        <div class="dialog-content">
          <h3 class="dialog-title"></h3>
          <p class="dialog-message"></p>
          <div class="dialog-choices"></div>
        </div>
      </div>
    `;
    return div;
  }

  configure(config) {
    const { title, message, choices, callback } = config;

    this.element.querySelector('.dialog-title').textContent = title;
    this.element.querySelector('.dialog-message').textContent = message;

    const choicesContainer = this.element.querySelector('.dialog-choices');
    choicesContainer.innerHTML = '';

    choices.forEach(choice => {
      const btn = document.createElement('button');
      btn.className = 'dialog-choice-btn';
      btn.textContent = choice.label;
      btn.addEventListener('click', () => {
        this.uiManager.hideDialog();
        if (callback) callback(choice.value);
      });
      choicesContainer.appendChild(btn);
    });
  }
}
```

### Toast Notification System

```javascript
class ToastManager {
  constructor(uiManager) {
    this.uiManager = uiManager;
    this.container = uiManager.overlayContainer;
    this.toasts = [];
  }

  show(message, duration = 3000) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;

    this.container.appendChild(toast);
    this.toasts.push(toast);

    // Animate in
    requestAnimationFrame(() => {
      toast.classList.add('show');
    });

    // Auto dismiss
    setTimeout(() => this.dismiss(toast), duration);

    return toast;
  }

  dismiss(toast) {
    toast.classList.remove('show');
    toast.addEventListener('transitionend', () => {
      toast.remove();
      this.toasts = this.toasts.filter(t => t !== toast);
    });
  }
}
```

### Progress Bar for Loading

```javascript
class LoadingScreen extends UIScreen {
  constructor(options) {
    super({ ...options, name: 'loading' });
    this.progress = 0;
  }

  createElement() {
    const div = document.createElement('div');
    div.className = 'loading-screen';
    div.innerHTML = `
      <div class="loading-content">
        <h2 class="loading-title">Loading...</h2>
        <div class="progress-bar-container">
          <div class="progress-bar-fill"></div>
        </div>
        <p class="loading-status"></p>
      </div>
    `;
    return div;
  }

  setProgress(value) {
    this.progress = Math.max(0, Math.min(100, value));
    const fill = this.element.querySelector('.progress-bar-fill');
    fill.style.width = `${this.progress}%`;
  }

  setStatus(message) {
    const status = this.element.querySelector('.loading-status');
    status.textContent = message;
  }
}
```

---

## Common Mistakes Beginners Make

### 1. No Transition Between Screens

```javascript
// ❌ WRONG: Hard cut between screens
showScreen(newScreen) {
  oldScreen.style.display = 'none';
  newScreen.style.display = 'block';
}
// Jarring, breaks immersion

// ✅ CORRECT: Smooth transition
async showScreen(newScreen) {
  await this.fadeOut(oldScreen);
  await this.fadeIn(newScreen);
}
// Maintains flow and immersion
```

### 2. Multiple Screens Visible at Once

```javascript
// ❌ WRONG: No cleanup
showScreen(name) {
  screens[name].style.display = 'block';
}
// Old screens still visible underneath

// ✅ CORRECT: Hide previous screen
async showScreen(name) {
  if (currentScreen) {
    await hideScreen(currentScreen);
  }
  screens[name].style.display = 'block';
  currentScreen = name;
}
// Only one screen visible at a time
```

### 3. Not Handling Back Navigation

```javascript
// ❌ WRONG: No back button support
optionsButton.onclick = () => {
  showScreen('options');
}
// User gets trapped

// ✅ CORRECT: Maintain screen stack
optionsButton.onclick = () => {
  screenStack.push(currentScreen);
  showScreen('options');
}

backButton.onclick = () => {
  const previous = screenStack.pop();
  showScreen(previous);
}
// User can always go back
```

### 4. Forgetting Mobile Responsive Design

```javascript
// ❌ WRONG: Fixed pixel sizes
.screen-content {
  width: 800px;
  height: 600px;
}
// Breaks on phones

// ✅ CORRECT: Responsive units
.screen-content {
  width: 90vw;
  max-width: 800px;
  height: 80vh;
  max-height: 600px;
}
// Works on all devices
```

---

## Performance Considerations

```
UI MANAGER PERFORMANCE:

Screen Transitions:
├── CSS transitions: GPU-accelerated (fast)
├── DOM manipulation: Minimal (only showing/hiding)
└── Impact: Negligible

Update Loop:
├── Only updates visible screens
├── Pauses hidden screen updates
└── Impact: Minimal

Memory:
├── DOM elements persist once created
├── No repeated create/destroy cycles
└── Impact: Low overhead

Optimization Strategies:
├── Use CSS transforms (GPU-accelerated)
├── Avoid layout thrashing (batch DOM reads/writes)
├── Use requestAnimationFrame for animations
├── Lazy-load screen content when needed
└── Virtualize long lists (recycle DOM elements)
```

---

## CSS Structure for UI

```css
/* Main container */
.ui-container {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;  /* Let clicks pass through to game */
  z-index: 100;
}

.screen-container,
.overlay-container,
.hud-container {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: auto;
}

/* Screens */
.screen {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.8);
}

.screen-content {
  background: #1a1a1a;
  border-radius: 8px;
  padding: 2rem;
  max-width: 800px;
  width: 90%;
  max-height: 80vh;
  overflow-y: auto;
}

/* Transitions */
.screen {
  transition: opacity 300ms ease,
              transform 300ms ease;
}

/* Dialog overlay */
.dialog-screen {
  background: rgba(0, 0, 0, 0.5);
  z-index: 200;
}

.dialog-container {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
}

.dialog-content {
  background: #2a2a2a;
  border-radius: 8px;
  padding: 1.5rem;
  min-width: 300px;
  max-width: 500px;
}

/* HUD */
.hud-container {
  pointer-events: none;
}

.hud-element {
  pointer-events: auto;
}

/* Loading */
.progress-bar-container {
  width: 100%;
  height: 8px;
  background: #333;
  border-radius: 4px;
  overflow: hidden;
}

.progress-bar-fill {
  height: 100%;
  background: #00ff88;
  transition: width 300ms ease;
}

/* Toast */
.toast {
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%) translateY(100px);
  background: #333;
  color: white;
  padding: 12px 24px;
  border-radius: 4px;
  opacity: 0;
  transition: transform 300ms ease, opacity 300ms ease;
}

.toast.show {
  transform: translateX(-50%) translateY(0);
  opacity: 1;
}
```

---

## Related Systems

- [Loading Screen](./loading-screen.md) - Loading progress display
- [Start Screen](./start-screen.md) - Main menu and game entry
- [Options Menu](./options-menu.md) - Settings and configuration
- [Dialog Choice UI](./dialog-choice-ui.md) - Interactive dialog system
- [Touch Joystick](./touch-joystick.md) - Mobile controls
- [GameManager](../02-core/game-manager.md) - Game state coordination

---

## Source File Reference

**Primary Files**:
- `../src/ui/UIManager.js` - Main UI orchestration (estimated)

**Key Classes**:
- `UIManager` - Central screen management
- `UIScreen` - Base class for all screens

**Dependencies**:
- DOM API (element creation, manipulation)
- CSS Transitions (smooth animations)
- GameManager (event coordination)

---

## References

- [MDN: Web APIs](https://developer.mozilla.org/en-US/docs/Web/API) - Browser APIs
- [MDN: CSS Transitions](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Transitions) - Smooth animations
- [Web Performance](https://web.dev/fast/) - Performance optimization

*Documentation last updated: January 12, 2026*
