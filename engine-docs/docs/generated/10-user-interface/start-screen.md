# Start Screen - First Principles Guide

## Overview

The **Start Screen** (also called Main Menu) is the gateway to your game. It's the first thing players see after loading completes, and where they decide how to experience the game—new game, continue, options, or quit. More than just a menu, the start screen sets expectations, establishes the game's tone, and provides a comfortable space for players to prepare themselves.

Think of the start screen as the **"front porch"** of your game—like a home's entrance, it welcomes visitors, gives them a sense of what's inside, and lets them decide when they're ready to step through the door.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create a welcoming atmosphere that builds anticipation while providing a comfortable decision space. Players should feel invited and intrigued, not overwhelmed or confused.

**Why a Dedicated Start Screen?**
- **Pacing**: Let players breathe before starting
- **Expectation Setting**: Shows game style and mood
- **Preparation**: Allow settings adjustment before play
- **Commitment**: Explicit "start" action creates intention
- **Save Access**: Continue from previous sessions

**Player Journey**:
```
Loading Completes
    ↓
Start Screen Appears → "Here's my game" → First impression formed
    ↓
Background/Atmosphere → "This feels like [genre]" → Mood established
    ↓
Menu Options → "What can I do?" → Agency provided
    ↓
Selection → "I'm ready" → Intentional start
    ↓
Game Begins → Player committed and prepared
```

### Design Principles

**1. Clear Visual Hierarchy**
The most important action should be most prominent:
- "New Game" or "Continue" should stand out
- Secondary options (Settings, Credits) less prominent
- Exit option available but not emphasized

**2. Immediate Playability**
If the player has a save, "Continue" should be:
- The first option
- Auto-selected
- Startable with one key press (Enter)

**3. Atmospheric Background**
The menu should feel like part of the game:
- Animated game scene (not static image)
- Ambient audio (music, environmental sounds)
- Subtle motion (breathing life into the scene)

**4. Responsive Feedback**
Every interaction should feel good:
- Hover states on all options
- Selection sounds
- Smooth transitions between menu and game

### Menu Structure

A typical start screen hierarchy:

```
START SCREEN
├── PRIMARY ACTIONS (prominent)
│   ├── New Game
│   ├── Continue (if saves exist)
│   └── Load Game (if multiple saves)
│
├── SECONDARY ACTIONS (less prominent)
│   ├── Options/Settings
│   ├── Credits
│   └── Extras/Bonus Content
│
├── SYSTEM (minimal)
│   └── Quit/Exit
│
└── VERSION INFO (subtle)
    └── Version number, build info
```

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the start screen, you should know:
- **DOM manipulation** - Creating interactive HTML elements
- **Event handling** - Click, keyboard, and gamepad input
- **CSS animations** - Hover effects and transitions
- **Save data** - Reading existing game saves
- **State management** - Tracking menu selection

### Core Architecture

```
START SCREEN ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│                     START SCREEN                        │
│  - Menu navigation                                      │
│  - Save detection                                       │
│  - Background rendering                                 │
│  - Audio management                                     │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   MENU       │  │   BACKGROUND  │  │   SAVES      │
│  - Navigation│  │  - 3D Scene   │  │  - Detect    │
│  - Selection │  │  - Animation  │  │  - Display   │
│  - Confirm   │  │  - Particles  │  │  - Load      │
└───────────────┘  └───────────────┘  └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   INPUT      │
                    │  - Mouse     │
                    │  - Keyboard  │
                    │  - Gamepad   │
                    └───────────────┘
```

### StartScreen Class

```javascript
class StartScreen extends UIScreen {
  constructor(options = {}) {
    super({
      ...options,
      name: 'start',
      isOverlay: false
    });

    // Menu state
    this.menuItems = [];
    this.selectedIndex = 0;
    this.saveGames = [];

    // Configuration
    this.config = {
      showNewGame: true,
      showContinue: true,
      showLoadGame: true,
      showOptions: true,
      showCredits: true,
      showQuit: true,
      animatedBackground: true,
      ambientAudio: true
    };

    Object.assign(this.config, options.config || {});

    // Input handling
    this.inputHandler = new StartScreenInputHandler(this);

    // Background scene
    this.backgroundScene = null;
  }

  /**
   * Create start screen DOM
   */
  createElement() {
    const div = document.createElement('div');
    div.className = 'start-screen';
    div.innerHTML = `
      <div class="start-background">
        <canvas id="start-background-canvas"></canvas>
        <div class="vignette"></div>
      </div>

      <div class="start-content">
        <!-- Game Logo/Title -->
        <div class="game-title">
          <h1 class="title-main">SHADOW ENGINE</h1>
          <p class="title-subtitle">A WebGPU Experience</p>
        </div>

        <!-- Main Menu -->
        <nav class="main-menu">
          <button class="menu-item" data-action="new-game">
            <span class="item-icon">▶</span>
            <span class="item-label">New Game</span>
          </button>
          <button class="menu-item" data-action="continue" style="display: none;">
            <span class="item-icon">↻</span>
            <span class="item-label">Continue</span>
            <span class="item-info">Auto-save</span>
          </button>
          <button class="menu-item" data-action="load-game">
            <span class="item-icon">📁</span>
            <span class="item-label">Load Game</span>
          </button>
          <button class="menu-item" data-action="options">
            <span class="item-icon">⚙</span>
            <span class="item-label">Options</span>
          </button>
          <button class="menu-item" data-action="credits">
            <span class="item-icon">©</span>
            <span class="item-label">Credits</span>
          </button>
          <button class="menu-item menu-item-secondary" data-action="quit">
            <span class="item-icon">✕</span>
            <span class="item-label">Quit</span>
          </button>
        </nav>

        <!-- Version Info -->
        <div class="version-info">
          <span class="version-number">v1.0.0</span>
          <span class="build-info">| Build 2024.01.12</span>
        </div>

        <!-- Footer hints -->
        <div class="menu-hints">
          <span class="hint">↑↓ Navigate</span>
          <span class="hint">Enter Select</span>
          <span class="hint">ESC Back</span>
        </div>
      </div>
    `;

    // Store element references
    this.menuContainer = div.querySelector('.main-menu');
    this.menuButtons = Array.from(div.querySelectorAll('.menu-item'));
    this.titleElement = div.querySelector('.game-title');
    this.versionInfo = div.querySelector('.version-info');

    return div;
  }

  /**
   * Initialize start screen
   */
  initialize() {
    // Setup menu items
    this.setupMenuItems();

    // Check for save games
    this.detectSaveGames();

    // Setup input handlers
    this.inputHandler.initialize();

    // Setup background
    if (this.config.animatedBackground) {
      this.setupBackground();
    }

    // Start ambient audio if configured
    if (this.config.ambientAudio) {
      this.startAmbientAudio();
    }

    // Auto-select first enabled option
    this.selectFirstEnabled();
  }

  /**
   * Setup menu items
   */
  setupMenuItems() {
    this.menuItems = this.menuButtons
      .map(btn => ({
        element: btn,
        action: btn.dataset.action,
        enabled: true,
        visible: btn.style.display !== 'none'
      }))
      .filter(item => item.visible);

    // Add click handlers
    this.menuButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        this.handleAction(btn.dataset.action);
      });

      // Hover sound
      btn.addEventListener('mouseenter', () => {
        this.playHoverSound();
      });
    });

    // Add keyboard navigation
    document.addEventListener('keydown', (e) => this.onKeyDown(e));
  }

  /**
   * Detect existing save games
   */
  detectSaveGames() {
    // Check for save games in localStorage or indexedDB
    this.saveGames = this.gameManager.saveManager.getSaveList();

    // Show/hide continue button based on saves
    const continueBtn = this.menuButtons.find(btn => btn.dataset.action === 'continue');
    const loadGameBtn = this.menuButtons.find(btn => btn.dataset.action === 'load-game');

    if (this.saveGames.length > 0) {
      // Show continue with latest save info
      if (continueBtn) {
        continueBtn.style.display = '';
        const latestSave = this.saveGames[0];
        const infoSpan = continueBtn.querySelector('.item-info');
        if (infoSpan && latestSave.metadata) {
          const date = new Date(latestSave.metadata.timestamp);
          infoSpan.textContent = this.formatSaveInfo(latestSave.metadata);
        }
      }

      // Enable load game
      if (loadGameBtn) {
        loadGameBtn.classList.remove('disabled');
      }
    } else {
      // Hide continue, disable load game
      if (continueBtn) {
        continueBtn.style.display = 'none';
      }
      if (loadGameBtn) {
        loadGameBtn.classList.add('disabled');
      }
    }

    // Rebuild menu items list
    this.menuItems = this.menuButtons
      .map(btn => ({
        element: btn,
        action: btn.dataset.action,
        enabled: !btn.classList.contains('disabled'),
        visible: btn.style.display !== 'none'
      }))
      .filter(item => item.visible);
  }

  /**
   * Format save info for display
   */
  formatSaveInfo(metadata) {
    const date = new Date(metadata.timestamp);
    const timeStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    if (metadata.chapter) {
      return `Chapter ${metadata.chapter} - ${timeStr}`;
    } else if (metadata.level) {
      return `${metadata.level} - ${timeStr}`;
    }
    return timeStr;
  }

  /**
   * Setup animated background
   */
  setupBackground() {
    const canvas = this.element.querySelector('#start-background-canvas');
    if (!canvas) return;

    // Create a simple 3D background scene
    this.backgroundScene = new StartBackgroundScene({
      canvas,
      gameManager: this.gameManager
    });

    this.backgroundScene.initialize();
  }

  /**
   * Start ambient audio
   */
  startAmbientAudio() {
    // Start menu music or ambient sounds
    this.gameManager.audioManager.playMusic('menu_theme', {
      volume: 0.5,
      loop: true,
      fade: 1.0
    });
  }

  /**
   * Select menu item by index
   */
  selectItem(index) {
    if (index < 0 || index >= this.menuItems.length) return;

    // Deselect current
    if (this.menuItems[this.selectedIndex]) {
      this.menuItems[this.selectedIndex].element.classList.remove('selected');
    }

    // Select new
    this.selectedIndex = index;
    this.menuItems[index].element.classList.add('selected');

    // Play selection sound
    this.playSelectSound();
  }

  /**
   * Select first enabled item
   */
  selectFirstEnabled() {
    const firstEnabled = this.menuItems.findIndex(item => item.enabled);
    if (firstEnabled >= 0) {
      this.selectItem(firstEnabled);
    }
  }

  /**
   * Handle keyboard input
   */
  onKeyDown(event) {
    // Only handle if this screen is visible
    if (!this.element.style.display || this.element.style.display === 'none') {
      return;
    }

    switch (event.key) {
      case 'ArrowUp':
      case 'w':
      case 'W':
        event.preventDefault();
        this.selectPrevious();
        break;

      case 'ArrowDown':
      case 's':
      case 'S':
        event.preventDefault();
        this.selectNext();
        break;

      case 'Enter':
      case ' ':
        event.preventDefault();
        this.confirmSelection();
        break;

      case 'Escape':
        event.preventDefault();
        // On start screen, ESC could show quit confirmation
        this.showQuitConfirmation();
        break;
    }
  }

  /**
   * Select previous menu item
   */
  selectPrevious() {
    let newIndex = this.selectedIndex - 1;
    if (newIndex < 0) {
      newIndex = this.menuItems.length - 1;  // Wrap to bottom
    }

    // Find next enabled item
    while (!this.menuItems[newIndex].enabled && newIndex !== this.selectedIndex) {
      newIndex--;
      if (newIndex < 0) {
        newIndex = this.menuItems.length - 1;
      }
    }

    this.selectItem(newIndex);
  }

  /**
   * Select next menu item
   */
  selectNext() {
    let newIndex = this.selectedIndex + 1;
    if (newIndex >= this.menuItems.length) {
      newIndex = 0;  // Wrap to top
    }

    // Find next enabled item
    while (!this.menuItems[newIndex].enabled && newIndex !== this.selectedIndex) {
      newIndex++;
      if (newIndex >= this.menuItems.length) {
        newIndex = 0;
      }
    }

    this.selectItem(newIndex);
  }

  /**
   * Confirm current selection
   */
  confirmSelection() {
    const selectedItem = this.menuItems[this.selectedIndex];
    if (selectedItem && selectedItem.enabled) {
      this.handleAction(selectedItem.action);
    }
  }

  /**
   * Handle menu action
   */
  async handleAction(action) {
    // Play confirm sound
    this.playConfirmSound();

    switch (action) {
      case 'new-game':
        await this.startNewGame();
        break;

      case 'continue':
        await this.continueGame();
        break;

      case 'load-game':
        this.showLoadGameMenu();
        break;

      case 'options':
        await this.showOptions();
        break;

      case 'credits':
        await this.showCredits();
        break;

      case 'quit':
        await this.quitGame();
        break;
    }
  }

  /**
   * Start a new game
   */
  async startNewGame() {
    // Show confirmation if saves exist
    if (this.saveGames.length > 0) {
      const confirmed = await this.showConfirmation(
        'Start New Game?',
        'This will overwrite your auto-save. Continue?'
      );
      if (!confirmed) return;
    }

    // Fade out menu audio
    this.gameManager.audioManager.fadeMusic(0, 1.0);

    // Hide start screen
    await this.uiManager.hideScreen('start', {
      transition: UIManager.TransitionType.FADE
    });

    // Start new game
    this.gameManager.emit('game:newGame');
  }

  /**
   * Continue from latest save
   */
  async continueGame() {
    if (this.saveGames.length === 0) return;

    const latestSave = this.saveGames[0];

    // Fade out menu audio
    this.gameManager.audioManager.fadeMusic(0, 1.0);

    // Hide start screen
    await this.uiManager.hideScreen('start', {
      transition: UIManager.TransitionType.FADE
    });

    // Load save
    this.gameManager.emit('game:loadSave', { saveId: latestSave.id });
  }

  /**
   * Show load game submenu
   */
  showLoadGameMenu() {
    // This would show a save selection screen
    this.uiManager.showScreen('loadGame', {
      transition: UIManager.TransitionType.SLIDE_LEFT,
      data: { saves: this.saveGames }
    });
  }

  /**
   * Show options menu
   */
  async showOptions() {
    this.uiManager.showScreen('options', {
      transition: UIManager.TransitionType.SLIDE_LEFT
    });
  }

  /**
   * Show credits screen
   */
  async showCredits() {
    this.uiManager.showScreen('credits', {
      transition: UIManager.TransitionType.SLIDE_LEFT
    });
  }

  /**
   * Show quit confirmation
   */
  async showQuitConfirmation() {
    const confirmed = await this.showConfirmation(
      'Quit Game?',
      'Are you sure you want to exit?'
    );

    if (confirmed) {
      this.quitGame();
    }
  }

  /**
   * Quit the game
   */
  async quitGame() {
    // Fade out everything
    this.gameManager.audioManager.fadeMusic(0, 1.0);

    // Show fade overlay
    const overlay = document.createElement('div');
    overlay.className = 'quit-overlay';
    overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: black;
      opacity: 0;
      transition: opacity 1s ease;
      z-index: 9999;
    `;
    document.body.appendChild(overlay);

    // Fade in
    requestAnimationFrame(() => {
      overlay.style.opacity = '1';
    });

    // Wait for fade, then close
    setTimeout(() => {
      this.gameManager.quit();
    }, 1000);
  }

  /**
   * Show a confirmation dialog
   */
  showConfirmation(title, message) {
    return new Promise(resolve => {
      this.uiManager.showDialog({
        title,
        message,
        choices: [
          { label: 'Yes', value: true },
          { label: 'No', value: false }
        ],
        callback: (result) => resolve(result)
      });
    });
  }

  /**
   * Play hover sound
   */
  playHoverSound() {
    this.gameManager.audioManager.playSfx('menu_hover', { volume: 0.3 });
  }

  /**
   * Play selection sound
   */
  playSelectSound() {
    this.gameManager.audioManager.playSfx('menu_select', { volume: 0.4 });
  }

  /**
   * Play confirm sound
   */
  playConfirmSound() {
    this.gameManager.audioManager.playSfx('menu_confirm', { volume: 0.5 });
  }

  /**
   * Called when screen is shown
   */
  onShow(data) {
    // Refresh save detection
    this.detectSaveGames();

    // Select first enabled item
    this.selectFirstEnabled();

    // Start background animation
    if (this.backgroundScene) {
      this.backgroundScene.start();
    }

    // Resume ambient audio
    if (this.config.ambientAudio) {
      this.gameManager.audioManager.resumeMusic();
    }
  }

  /**
   * Called when screen is hidden
   */
  onHide() {
    // Stop background animation
    if (this.backgroundScene) {
      this.backgroundScene.stop();
    }

    // Pause ambient audio
    if (this.config.ambientAudio) {
      this.gameManager.audioManager.pauseMusic();
    }
  }

  /**
   * Update loop
   */
  update(dt) {
    // Update background scene
    if (this.backgroundScene) {
      this.backgroundScene.update(dt);
    }
  }

  /**
   * Clean up
   */
  destroy() {
    // Cleanup background
    if (this.backgroundScene) {
      this.backgroundScene.destroy();
    }

    // Remove event listeners
    document.removeEventListener('keydown', this.onKeyDown);

    // Cleanup input handler
    this.inputHandler.destroy();

    super.destroy();
  }
}

/**
 * Input handler for start screen (supports gamepad)
 */
class StartScreenInputHandler {
  constructor(startScreen) {
    this.startScreen = startScreen;
    this.gamepadIndex = null;
    this.lastButtonState = {};
  }

  initialize() {
    // Listen for gamepad connection
    window.addEventListener('gamepadconnected', (e) => {
      this.gamepadIndex = e.gamepad.index;
      this.startScreen.logger.log('Gamepad connected:', e.gamepad.id);
    });

    window.addEventListener('gamepaddisconnected', (e) => {
      if (this.gamepadIndex === e.gamepad.index) {
        this.gamepadIndex = null;
      }
    });
  }

  update() {
    if (this.gamepadIndex === null) return;

    const gamepad = navigator.getGamepads()[this.gamepadIndex];
    if (!gamepad) return;

    // D-pad up / Left stick up
    if (this.isButtonPressed(gamepad, 12) || gamepad.axes[1] < -0.5) {
      this.startScreen.selectPrevious();
    }

    // D-pad down / Left stick down
    if (this.isButtonPressed(gamepad, 13) || gamepad.axes[1] > 0.5) {
      this.startScreen.selectNext();
    }

    // A button / Cross button
    if (this.isButtonPressed(gamepad, 0)) {
      this.startScreen.confirmSelection();
    }

    // B button / Circle button
    if (this.isButtonPressed(gamepad, 1)) {
      this.startScreen.showQuitConfirmation();
    }

    // Update button state
    gamepad.buttons.forEach((button, i) => {
      this.lastButtonState[i] = button.pressed;
    });
  }

  isButtonPressed(gamepad, buttonIndex) {
    const button = gamepad.buttons[buttonIndex];
    const wasPressed = this.lastButtonState[buttonIndex];
    const isPressed = button.pressed;

    // Return true on button down (not held)
    return isPressed && !wasPressed;
  }

  destroy() {
    // Cleanup event listeners
    window.removeEventListener('gamepadconnected', this.onGamepadConnected);
    window.removeEventListener('gamepaddisconnected', this.onGamepadDisconnected);
  }
}

export default StartScreen;
```

### Start Background Scene

```javascript
/**
 * Animated background for start screen
 */
class StartBackgroundScene {
  constructor(options = {}) {
    this.canvas = options.canvas;
    this.gameManager = options.gameManager;

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.isRunning = false;
  }

  initialize() {
    // Create basic Three.js scene
    const THREE = this.gameManager.THREE;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0a1a);
    this.scene.fog = new THREE.Fog(0x0a0a1a, 1, 10);

    // Camera
    this.camera = new THREE.PerspectiveCamera(
      60,
      this.canvas.clientWidth / this.canvas.clientHeight,
      0.1,
      100
    );
    this.camera.position.set(0, 2, 5);
    this.camera.lookAt(0, 0, 0);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true
    });
    this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Add content
    this.addParticles();
    this.addGround();
    this.addLights();
  }

  addParticles() {
    const THREE = this.gameManager.THREE;

    const geometry = new THREE.BufferGeometry();
    const count = 500;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 20;
      positions[i * 3 + 1] = Math.random() * 5;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 20;

      // Cyan/green colors
      colors[i * 3] = 0;
      colors[i * 3 + 1] = 0.5 + Math.random() * 0.5;
      colors[i * 3 + 2] = 0.5 + Math.random() * 0.5;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending
    });

    this.particles = new THREE.Points(geometry, material);
    this.scene.add(this.particles);
  }

  addGround() {
    const THREE = this.gameManager.THREE;

    const geometry = new THREE.PlaneGeometry(50, 50, 50, 50);

    // Add some height variation
    const positions = geometry.attributes.position.array;
    for (let i = 0; i < positions.length; i += 3) {
      positions[i + 2] = Math.sin(positions[i] * 0.5) * Math.cos(positions[i + 1] * 0.5) * 0.3;
    }
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      color: 0x1a1a2e,
      roughness: 0.9,
      metalness: 0.1,
      wireframe: true
    });

    this.ground = new THREE.Mesh(geometry, material);
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.position.y = -0.5;
    this.scene.add(this.ground);
  }

  addLights() {
    const THREE = this.gameManager.THREE;

    const ambient = new THREE.AmbientLight(0x404060, 0.5);
    this.scene.add(ambient);

    const directional = new THREE.DirectionalLight(0x00ff88, 0.5);
    directional.position.set(5, 10, 5);
    this.scene.add(directional);

    const point = new THREE.PointLight(0x00aaff, 1, 10);
    point.position.set(0, 2, 0);
    this.scene.add(point);
    this.pointLight = point;
  }

  start() {
    this.isRunning = true;
    this.startTime = Date.now();
    this.animate();
  }

  stop() {
    this.isRunning = false;
  }

  update(dt) {
    if (!this.isRunning) return;

    const time = (Date.now() - this.startTime) / 1000;

    // Rotate particles
    if (this.particles) {
      this.particles.rotation.y = time * 0.05;
    }

    // Move point light in circle
    if (this.pointLight) {
      this.pointLight.position.x = Math.sin(time * 0.5) * 3;
      this.pointLight.position.z = Math.cos(time * 0.5) * 3;
    }

    // Subtle camera sway
    this.camera.position.x = Math.sin(time * 0.2) * 0.2;
    this.camera.position.y = 2 + Math.sin(time * 0.3) * 0.1;
    this.camera.lookAt(0, 0, 0);
  }

  animate() {
    if (!this.isRunning) return;

    this.update(1 / 60);
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(() => this.animate());
  }

  destroy() {
    this.stop();
    if (this.renderer) {
      this.renderer.dispose();
    }
  }
}

export default StartBackgroundScene;
```

---

## 📝 How To Build A Start Screen Like This

### Step 1: Create Basic Menu Structure

```javascript
class SimpleStartScreen extends UIScreen {
  createElement() {
    const div = document.createElement('div');
    div.className = 'start-screen';
    div.innerHTML = `
      <div class="menu-container">
        <h1>My Game</h1>
        <nav class="menu">
          <button data-action="start">Start Game</button>
          <button data-action="options">Options</button>
          <button data-action="quit">Quit</button>
        </nav>
      </div>
    `;

    // Add click handlers
    div.querySelectorAll('button').forEach(btn => {
      btn.addEventListener('click', () => {
        this.handleAction(btn.dataset.action);
      });
    });

    return div;
  }

  handleAction(action) {
    switch (action) {
      case 'start':
        this.gameManager.start();
        break;
      case 'options':
        this.uiManager.showScreen('options');
        break;
      case 'quit':
        this.gameManager.quit();
        break;
    }
  }
}
```

### Step 2: Add Keyboard Navigation

```javascript
initialize() {
  this.selectedIndex = 0;
  this.menuItems = Array.from(this.element.querySelectorAll('button'));

  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') {
      this.selectedIndex = (this.selectedIndex + 1) % this.menuItems.length;
      this.updateSelection();
    } else if (e.key === 'ArrowUp') {
      this.selectedIndex = (this.selectedIndex - 1 + this.menuItems.length) % this.menuItems.length;
      this.updateSelection();
    } else if (e.key === 'Enter') {
      this.menuItems[this.selectedIndex].click();
    }
  });
}

updateSelection() {
  this.menuItems.forEach((btn, i) => {
    btn.classList.toggle('selected', i === this.selectedIndex);
  });
}
```

### Step 3: Style with CSS

```css
.start-screen {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
}

.menu-container {
  text-align: center;
  color: white;
}

.menu-container h1 {
  font-size: 4rem;
  margin-bottom: 2rem;
  text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
}

.menu {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.menu button {
  padding: 1rem 2rem;
  font-size: 1.2rem;
  background: rgba(255, 255, 255, 0.1);
  border: 2px solid rgba(0, 255, 136, 0.3);
  color: white;
  cursor: pointer;
  transition: all 0.3s ease;
}

.menu button:hover,
.menu button.selected {
  background: rgba(0, 255, 136, 0.2);
  border-color: rgba(0, 255, 136, 0.8);
  transform: scale(1.05);
}
```

---

## 🔧 Variations For Your Game

### Story-Heavy Menu

```javascript
// Show chapter selection or story recap
class StoryStartScreen extends StartScreen {
  detectSaveGames() {
    super.detectSaveGames();

    // Show "Previously On" section if has saves
    if (this.saveGames.length > 0) {
      this.showPreviouslyOn();
    }
  }

  showPreviouslyOn() {
    const recap = document.createElement('div');
    recap.className = 'previously-on';
    recap.innerHTML = `
      <h3>Previously On...</h3>
      <p>You discovered the truth about the shadows...</p>
    `;
    this.menuContainer.prepend(recap);
  }
}
```

### Quick-Continue Menu

```javascript
// For games with frequent saves, show continue prominently
class QuickContinueScreen extends StartScreen {
  createElement() {
    const div = super.createElement();

    // Add large continue button at top
    const continueBtn = document.createElement('button');
    continueBtn.className = 'quick-continue';
    continueBtn.innerHTML = `
      <span class="continue-icon">▶</span>
      <span class="continue-text">Continue</span>
    `;
    continueBtn.addEventListener('click', () => this.continueGame());

    const content = div.querySelector('.start-content');
    content.insertBefore(continueBtn, content.firstChild);

    return div;
  }
}
```

### Character-Select Menu

```javascript
// For games with multiple characters/protagonists
class CharacterSelectScreen extends StartScreen {
  createElement() {
    const div = super.createElement();

    const charSelect = document.createElement('div');
    charSelect.className = 'character-select';
    charSelect.innerHTML = `
      <div class="character-list">
        ${this.characters.map((char, i) => `
          <div class="character-card" data-index="${i}">
            <img src="${char.portrait}" alt="${char.name}">
            <h3>${char.name}</h3>
            <p>${char.description}</p>
          </div>
        `).join('')}
      </div>
    `;
    div.appendChild(charSelect);

    return div;
  }
}
```

---

## Common Mistakes Beginners Make

### 1. No "Continue" for Existing Saves

```javascript
// ❌ WRONG: Always shows "New Game" first
showMenu() {
  // Always new game at top
  showButton('New Game');
  showButton('Continue');
  showButton('Load Game');
}
// Returning players frustrated

// ✅ CORRECT: Prioritize continue
showMenu() {
  if (hasSaves) {
    showButton('Continue', { primary: true });
    showButton('New Game');
  } else {
    showButton('New Game', { primary: true });
  }
  showButton('Load Game');
}
// Returning players can jump right back in
```

### 2. No Keyboard Navigation

```javascript
// ❌ WRONG: Mouse only
<button onclick="startGame()">Start Game</button>
// Inaccessible for keyboard users

// ✅ CORRECT: Keyboard + mouse
<button onclick="startGame()" tabindex="0">Start Game</button>
// Plus arrow key navigation implementation
```

### 3. Jarring Transitions to Game

```javascript
// ❌ WRONG: Instant cut
startGame() {
  hideMenu();
  startGameplay();
}
// Whiplash-inducing

// ✅ CORRECT: Smooth transition
async startGame() {
  await fadeOutMenu();
  await fadeInGame();
  startGameplay();
}
// Smooth, professional feel
```

### 4. Static Boring Background

```javascript
// ❌ WRONG: Static image
background: url('menu.png');
// Looks like a cheap mobile game

// ✅ CORRECT: Animated scene
// Use actual game scene with slow camera movement
// Or animated particles, subtle effects
// Feels alive and premium
```

---

## CSS for Start Screen

```css
.start-screen {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

.start-background {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

#start-background-canvas {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.vignette {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: radial-gradient(ellipse at center,
    transparent 0%,
    transparent 50%,
    rgba(0, 0, 0, 0.5) 100%);
  pointer-events: none;
}

.start-content {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  padding: 2rem;
  color: white;
}

.game-title {
  text-align: center;
  margin-bottom: 3rem;
  animation: title-float 3s ease-in-out infinite;
}

.title-main {
  font-size: clamp(2rem, 8vw, 6rem);
  font-weight: bold;
  text-shadow:
    0 0 20px rgba(0, 255, 136, 0.5),
    0 0 40px rgba(0, 255, 136, 0.3);
  margin-bottom: 0.5rem;
}

.title-subtitle {
  font-size: 1.2rem;
  color: rgba(255, 255, 255, 0.7);
  letter-spacing: 0.2em;
  text-transform: uppercase;
}

@keyframes title-float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-10px); }
}

.main-menu {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  min-width: 250px;
}

.menu-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem 1.5rem;
  background: rgba(255, 255, 255, 0.05);
  border: 2px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  color: white;
  font-size: 1.1rem;
  cursor: pointer;
  transition: all 0.3s ease;
  text-align: left;
}

.menu-item:hover:not(.disabled),
.menu-item.selected {
  background: rgba(0, 255, 136, 0.1);
  border-color: rgba(0, 255, 136, 0.5);
  transform: translateX(10px);
}

.menu-item.selected {
  box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
}

.menu-item.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.item-icon {
  font-size: 1.2rem;
  width: 24px;
  text-align: center;
}

.item-label {
  flex: 1;
}

.item-info {
  font-size: 0.8rem;
  color: rgba(255, 255, 255, 0.5);
}

.menu-item-secondary {
  margin-top: 1rem;
  opacity: 0.7;
}

.version-info {
  margin-top: 3rem;
  font-size: 0.8rem;
  color: rgba(255, 255, 255, 0.4);
}

.menu-hints {
  margin-top: 2rem;
  display: flex;
  gap: 2rem;
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.5);
}

.quit-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: black;
  z-index: 9999;
}
```

---

## Related Systems

- [UIManager](./ui-manager.md) - Screen orchestration
- [Options Menu](./options-menu.md) - Settings configuration
- [Loading Screen](./loading-screen.md) - Loading display
- [GameManager](../02-core/game-manager.md) - Game state management

---

## Source File Reference

**Primary Files**:
- `../src/ui/StartScreen.js` - Start screen component (estimated)

**Key Classes**:
- `StartScreen` - Main menu implementation
- `StartBackgroundScene` - Animated 3D background
- `StartScreenInputHandler` - Gamepad support

**Dependencies**:
- UIManager (screen management)
- GameManager (events, saves)
- Three.js (background rendering)

---

## References

- [Gamepad API](https://developer.mozilla.org/en-US/docs/Web/API/Gamepad_API) - Controller support
- [CSS Gradients](https://developer.mozilla.org/en-US/docs/Web/CSS/gradient) - Background styling
- [Web Animations API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Animations_API) - Smooth animations

*Documentation last updated: January 12, 2026*
