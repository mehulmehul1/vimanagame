# Loading Screen - First Principles Guide

## Overview

The **Loading Screen** is the player's first impression of the game and the bridge between launching and playing. It displays while assets load, shaders compile, and the game world initializes. A good loading screen isn't just a "please wait" message—it's an opportunity to set the mood, provide context, and maintain player engagement during what would otherwise be a boring wait.

Think of the loading screen as the **"curtain raiser"**—like the opening act before a theater performance, it builds anticipation and provides the first taste of the game's atmosphere before the main experience begins.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Transform waiting into anticipation. Instead of "this is taking forever," the player should feel "something exciting is coming." The loading screen is where expectations are set and the game's identity is established.

**Why a Dedicated Loading Screen?**
- **First Impression**: Sets visual tone and mood
- **Expectation Setting**: Hints at what kind of game this is
- **Distraction**: Makes waiting feel shorter
- **Feedback**: Shows progress so player knows game isn't frozen
- **Lore Opportunity**: Can tell story or provide hints during load

**Player Psychology**:
```
Launch Game → Blank Screen → "Is it broken?"
     ↓
Loading Appears → "Okay, it's working" → Relief
     ↓
Art/Story Shows → "Oh, this looks cool" → Interest
     ↓
Progress Bar Moves → "Almost there..." → Anticipation
     ↓
Game Starts → Seamless transition → Immersion
```

### Design Principles

**1. Never Show a Blank Screen**
Even a simple logo is better than nothing. The player should immediately know the game is running.

**2. Show Meaningful Progress**
Fake progress bars that jump randomly break trust. Real progress based on actual loading stages builds confidence.

**3. Provide Visual Interest**
Static text is boring. Use:
- Animated logo or art
- Subtle particle effects
- Fading tips or lore text
- Ambient sound or music preview

**4. Match Game's Tone**
The loading screen should feel like part of the game:
- Horror game: Unsettling imagery, slow build
- Action game: Dynamic, energetic visuals
- Puzzle game: Clean, minimalist design

### Loading Stages

The game typically loads in stages:

```
STAGE 1: Core Assets (0-30%)
├── Engine initialization
├── Essential textures
└── Base shaders

STAGE 2: World Data (30-70%)
├── Level geometry
├── Object models
└── Environment assets

STAGE 3: Finalization (70-100%)
├── Audio loading
├── Post-processing setup
└── World spawn
```

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the loading screen, you should know:
- **Asset loading** - How files are fetched and processed
- **Promises** - Handling asynchronous operations
- **Progress tracking** - Calculating loading percentages
- **DOM updates** - Updating the loading display
- **Event emission** - Signaling when loading completes

### Core Architecture

```
LOADING SCREEN ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│                   LOADING SCREEN                        │
│  - Display progress bar                                 │
│  - Show loading messages                                │
│  - Display tips/lore                                    │
│  - Handle loading completion                            │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ PROGRESS      │  │   TIPS       │  │   VISUALS    │
│  - Bar fill   │  │  - Rotate    │  │  - Logo      │
│  - Percentage │  │  - Random    │  │  - Animation │
│  - Stage name │  │  - Timed     │  │  - Particles │
└───────────────┘  └───────────────┘  └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  LOADING     │
                    │  MANAGER     │
                    │  - Track     │
                    │  - Report    │
                    └───────────────┘
```

### LoadingScreen Class

```javascript
class LoadingScreen extends UIScreen {
  constructor(options = {}) {
    super({
      ...options,
      name: 'loading',
      isOverlay: false
    });

    // Loading state
    this.progress = 0;
    this.currentStage = '';
    this.loadingStages = [];
    this.tips = [];
    this.currentTipIndex = 0;
    this.tipInterval = null;

    // Configuration
    this.config = {
      showPercentage: true,
      showStageName: true,
      showTips: true,
      tipRotationTime: 8000,  // 8 seconds per tip
      minimumDisplayTime: 2000,  // Min 2 seconds
      smoothProgress: true,  // Smooth bar animation
      particlesEnabled: true
    };

    // Merge user config
    Object.assign(this.config, options.config || {});

    // Animation state
    this.displayedProgress = 0;  // For smooth animation
  }

  /**
   * Create loading screen DOM
   */
  createElement() {
    const div = document.createElement('div');
    div.className = 'loading-screen';
    div.innerHTML = `
      <div class="loading-background">
        <div class="loading-particles"></div>
      </div>

      <div class="loading-content">
        <!-- Logo / Title -->
        <div class="loading-logo">
          <h1 class="game-title">SHADOW ENGINE</h1>
          <div class="logo-subtitle">Initializing...</div>
        </div>

        <!-- Progress Bar -->
        <div class="progress-container">
          <div class="progress-bar-bg">
            <div class="progress-bar-fill"></div>
          </div>
          <div class="progress-text">
            <span class="progress-percentage">0%</span>
            <span class="progress-stage">Initializing...</span>
          </div>
        </div>

        <!-- Tips / Lore -->
        <div class="loading-tips">
          <div class="tip-icon"></div>
          <p class="tip-text"></p>
        </div>

        <!-- Loading indicators -->
        <div class="loading-indicators">
          <div class="indicator-dot"></div>
          <div class="indicator-dot"></div>
          <div class="indicator-dot"></div>
        </div>
      </div>
    `;

    // Store element references
    this.logoElement = div.querySelector('.loading-logo');
    this.progressBarFill = div.querySelector('.progress-bar-fill');
    this.progressPercentage = div.querySelector('.progress-percentage');
    this.progressStage = div.querySelector('.progress-stage');
    this.tipText = div.querySelector('.tip-text');
    this.indicatorDots = div.querySelectorAll('.indicator-dot');

    return div;
  }

  /**
   * Initialize loading screen
   */
  initialize() {
    // Setup tips
    if (this.config.showTips) {
      this.setupTips();
    }

    // Create particles if enabled
    if (this.config.particlesEnabled) {
      this.createParticles();
    }

    // Start indicator animation
    this.startIndicatorAnimation();
  }

  /**
   * Setup tips rotation
   */
  setupTips() {
    // Default tips
    this.tips = [
      "Tip: You can pause the game at any time by pressing ESC or P.",
      "Tip: Draw symbols carefully for better recognition.",
      "Tip: Explore thoroughly to find hidden secrets.",
      "Tip: Listen to audio cues for important information.",
      "Tip: Some interactions require specific timing.",
      "Lore: The shadows have been whispering for centuries...",
      "Lore: Those who listen too closely may never return.",
      "Lore: The amplifier holds the key to understanding."
    ];

    // Show first tip
    this.showRandomTip();

    // Rotate tips
    this.tipInterval = setInterval(() => {
      this.showRandomTip();
    }, this.config.tipRotationTime);
  }

  /**
   * Show a random tip
   */
  showRandomTip() {
    if (this.tips.length === 0) return;

    // Get a tip different from current
    let newTipIndex;
    do {
      newTipIndex = Math.floor(Math.random() * this.tips.length);
    } while (newTipIndex === this.currentTipIndex && this.tips.length > 1);

    this.currentTipIndex = newTipIndex;

    // Fade out, change text, fade in
    const tipElement = this.tipText;
    tipElement.style.opacity = '0';

    setTimeout(() => {
      tipElement.textContent = this.tips[newTipIndex];
      tipElement.style.opacity = '1';
    }, 300);
  }

  /**
   * Update loading progress
   */
  setProgress(value, stageName = '') {
    // Clamp value
    this.progress = Math.max(0, Math.min(100, value));

    // Update stage name
    if (stageName) {
      this.currentStage = stageName;
    }

    // Update display (smoothed if configured)
    if (this.config.smoothProgress) {
      // Actual update happens in animate loop
    } else {
      this.displayedProgress = this.progress;
      this.updateDisplay();
    }
  }

  /**
   * Update the visual display
   */
  updateDisplay() {
    // Update progress bar
    this.progressBarFill.style.width = `${this.displayedProgress}%`;

    // Update percentage text
    if (this.config.showPercentage) {
      this.progressPercentage.textContent = `${Math.round(this.displayedProgress)}%`;
    }

    // Update stage name
    if (this.config.showStageName && this.currentStage) {
      this.progressStage.textContent = this.currentStage;
    }

    // Update color based on progress
    const hue = (this.displayedProgress / 100) * 120;  // 0=red, 120=green
    this.progressBarFill.style.backgroundColor = `hsl(${hue}, 70%, 50%)`;
  }

  /**
   * Animate progress bar (for smooth transitions)
   */
  animateProgress(dt) {
    if (!this.config.smoothProgress) return;

    const target = this.progress;
    const current = this.displayedProgress;
    const diff = target - current;

    // Smooth approach
    if (Math.abs(diff) > 0.1) {
      this.displayedProgress += diff * 10 * dt;
    } else {
      this.displayedProgress = target;
    }

    this.updateDisplay();
  }

  /**
   * Set loading stages
   */
  setStages(stages) {
    // stages: [{ name: 'Loading assets', weight: 30 }, ...]
    this.loadingStages = stages;
  }

  /**
   * Create particle effects
   */
  createParticles() {
    const particleContainer = this.element.querySelector('.loading-particles');
    const particleCount = 50;

    for (let i = 0; i < particleCount; i++) {
      const particle = document.createElement('div');
      particle.className = 'loading-particle';
      particle.style.cssText = `
        left: ${Math.random() * 100}%;
        top: ${Math.random() * 100}%;
        width: ${2 + Math.random() * 4}px;
        height: ${2 + Math.random() * 4}px;
        animation-delay: ${Math.random() * 5}s;
        animation-duration: ${5 + Math.random() * 10}s;
      `;
      particleContainer.appendChild(particle);
    }
  }

  /**
   * Start loading indicator animation
   */
  startIndicatorAnimation() {
    let dotIndex = 0;

    setInterval(() => {
      // Reset all
      this.indicatorDots.forEach(dot => {
        dot.style.opacity = '0.3';
        dot.style.transform = 'scale(1)';
      });

      // Highlight current
      this.indicatorDots[dotIndex].style.opacity = '1';
      this.indicatorDots[dotIndex].style.transform = 'scale(1.3)';

      dotIndex = (dotIndex + 1) % this.indicatorDots.length;
    }, 500);
  }

  /**
   * Called when screen is shown
   */
  onShow(data) {
    // Reset progress
    this.progress = 0;
    this.displayedProgress = 0;
    this.currentStage = 'Initializing...';

    // Reset tip rotation
    if (this.tipInterval) {
      clearInterval(this.tipInterval);
    }
    if (this.config.showTips) {
      this.setupTips();
    }

    // Start animation loop
    this.startAnimationLoop();
  }

  /**
   * Called when screen is hidden
   */
  onHide() {
    // Stop tip rotation
    if (this.tipInterval) {
      clearInterval(this.tipInterval);
      this.tipInterval = null;
    }

    // Stop animation loop
    this.stopAnimationLoop();
  }

  /**
   * Start update loop
   */
  startAnimationLoop() {
    const animate = (time) => {
      if (!this.isShown) return;

      const dt = 1 / 60;  // Assume 60fps
      this.animateProgress(dt);

      this.animationFrame = requestAnimationFrame(animate);
    };

    this.isShown = true;
    this.animationFrame = requestAnimationFrame(animate);
  }

  /**
   * Stop update loop
   */
  stopAnimationLoop() {
    this.isShown = false;
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  /**
   * Update loop (for any continuous animations)
   */
  update(dt) {
    // Particles are handled by CSS animations
    // Progress smoothing is handled in animateProgress
  }

  /**
   * Trigger loading completion
   */
  complete() {
    // Ensure progress reaches 100
    this.setProgress(100, 'Complete!');

    // Wait a moment before hiding
    return new Promise(resolve => {
      setTimeout(() => {
        this.uiManager.hideLoading();
        resolve();
      }, this.config.minimumDisplayTime);
    });
  }

  /**
   * Clean up
   */
  destroy() {
    this.onHide();
    super.destroy();
  }
}

export default LoadingScreen;
```

### LoadingManager Class

```javascript
/**
 * Manages the loading process and coordinates with LoadingScreen
 */
class LoadingManager {
  constructor(options = {}) {
    this.gameManager = options.gameManager;
    this.uiManager = options.uiManager;
    this.assetLoader = options.assetLoader;
    this.logger = options.logger || console;

    // Loading state
    this.isLoading = false;
    this.currentProgress = 0;
    this.loadingStages = [];
    this.currentStageIndex = 0;

    // Progress tracking
    this.progressCallbacks = [];
  }

  /**
   * Start loading process
   */
  async load(loadConfig) {
    if (this.isLoading) {
      this.logger.warn('Already loading');
      return;
    }

    this.isLoading = true;

    // Show loading screen
    this.uiManager.showLoading();

    // Emit loading started event
    this.gameManager.emit('loading:started');

    // Define loading stages
    this.loadingStages = loadConfig.stages || [
      { name: 'Initializing', load: () => this.initialize() },
      { name: 'Loading Assets', load: () => this.loadAssets(loadConfig.assets) },
      { name: 'Preparing World', load: () => this.prepareWorld(loadConfig.world) },
      { name: 'Finalizing', load: () => this.finalize() }
    ];

    // Execute loading stages
    try {
      for (let i = 0; i < this.loadingStages.length; i++) {
        this.currentStageIndex = i;
        const stage = this.loadingStages[i];

        // Update stage name
        this.updateProgress(
          this.currentProgress,
          stage.name
        );

        // Execute stage loading
        await stage.load();

        // Update progress
        this.currentProgress = ((i + 1) / this.loadingStages.length) * 100;
        this.updateProgress(this.currentProgress, stage.name);
      }

      // Loading complete
      this.onLoadingComplete();

    } catch (error) {
      this.onLoadingError(error);
    }
  }

  /**
   * Update progress display
   */
  updateProgress(percent, stage) {
    const loadingScreen = this.uiManager.getScreen('loading');
    if (loadingScreen) {
      loadingScreen.setProgress(percent, stage);
    }

    // Notify callbacks
    this.progressCallbacks.forEach(callback => {
      callback(percent, stage);
    });

    // Emit event
    this.gameManager.emit('loading:progress', { percent, stage });
  }

  /**
   * Initialize stage
   */
  async initialize() {
    // Initialize engine systems
    this.logger.log('Initializing engine...');

    // Simulate initialization time
    await this.delay(500);
  }

  /**
   * Load assets stage
   */
  async loadAssets(assetList) {
    this.logger.log('Loading assets...');

    if (!this.assetLoader) {
      this.logger.warn('No asset loader provided');
      return;
    }

    // Load assets with progress tracking
    let loaded = 0;
    const total = assetList ? assetList.length : 0;

    for (const asset of assetList || []) {
      await this.assetLoader.load(asset);
      loaded++;

      // Update progress within this stage
      const stageProgress = (loaded / total) * 30;  // 30% for assets
      const totalProgress = 10 + stageProgress;  // After init (10%)
      this.updateProgress(totalProgress, `Loading ${asset.name}`);
    }
  }

  /**
   * Prepare world stage
   */
  async prepareWorld(worldConfig) {
    this.logger.log('Preparing world...');

    // Load world data, spawn entities, etc.
    await this.delay(1000);
  }

  /**
   * Finalize stage
   */
  async finalize() {
    this.logger.log('Finalizing...');

    // Final setup, start music, etc.
    await this.delay(500);
  }

  /**
   * Handle loading completion
   */
  onLoadingComplete() {
    this.logger.log('Loading complete!');

    this.isLoading = false;

    // Get loading screen and trigger completion
    const loadingScreen = this.uiManager.getScreen('loading');
    if (loadingScreen && loadingScreen.complete) {
      loadingScreen.complete();
    } else {
      this.uiManager.hideLoading();
    }

    // Emit event
    this.gameManager.emit('loading:complete');
  }

  /**
   * Handle loading error
   */
  onLoadingError(error) {
    this.logger.error('Loading error:', error);

    this.isLoading = false;

    // Show error message
    const loadingScreen = this.uiManager.getScreen('loading');
    if (loadingScreen) {
      loadingScreen.setProgress(0, 'Error: ' + error.message);
    }

    // Emit error event
    this.gameManager.emit('loading:error', { error });
  }

  /**
   * Add progress callback
   */
  onProgress(callback) {
    this.progressCallbacks.push(callback);
  }

  /**
   * Utility: Delay helper
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get current progress
   */
  getProgress() {
    return this.currentProgress;
  }

  /**
   * Check if currently loading
   */
  get isLoading() {
    return this.isLoading;
  }
}

export default LoadingManager;
```

---

## 📝 How To Build A Loading Screen Like This

### Step 1: Create the Loading Screen Class

```javascript
class MyLoadingScreen extends UIScreen {
  constructor(options) {
    super({ ...options, name: 'loading' });

    this.progress = 0;
    this.createElement();
  }

  createElement() {
    const div = document.createElement('div');
    div.className = 'loading-screen';
    div.innerHTML = `
      <div class="loading-content">
        <h1>Loading...</h1>
        <div class="progress-bar">
          <div class="progress-fill"></div>
        </div>
        <span class="progress-text">0%</span>
      </div>
    `;
    return div;
  }

  setProgress(value) {
    this.progress = Math.min(100, Math.max(0, value));
    const fill = this.element.querySelector('.progress-fill');
    const text = this.element.querySelector('.progress-text');

    fill.style.width = `${this.progress}%`;
    text.textContent = `${Math.round(this.progress)}%`;
  }
}
```

### Step 2: Register with UIManager

```javascript
const loadingScreen = new MyLoadingScreen({ uiManager, gameManager });
uiManager.registerScreen('loading', loadingScreen);
```

### Step 3: Use During Game Loading

```javascript
// Show loading screen
uiManager.showLoading();

// Update progress as you load
async function loadGame() {
  const loadingScreen = uiManager.getScreen('loading');

  loadingScreen.setProgress(0);
  await loadEngineAssets();
  loadingScreen.setProgress(30);

  await loadWorldAssets();
  loadingScreen.setProgress(70);

  await spawnPlayer();
  loadingScreen.setProgress(100);

  // Hide loading screen
  await uiManager.hideLoading();

  // Start game
  startGameplay();
}
```

---

## 🔧 Variations For Your Game

### Story-Driven Loading

```javascript
class StoryLoadingScreen extends LoadingScreen {
  setupTips() {
    this.storySegments = [
      "Chapter 1: The Awakening",
      "You open your eyes in a place you don't recognize...",
      "The shadows seem alive, watching your every move...",
      "Something terrible happened here, long ago...",
      "You must uncover the truth before it consumes you..."
    ];

    this.showNextStorySegment();
  }

  showNextStorySegment() {
    // Show segments sequentially
    this.tipText.textContent = this.storySegments[this.currentSegmentIndex];
    this.currentSegmentIndex = (this.currentSegmentIndex + 1) % this.storySegments.length;
  }
}
```

### Interactive Loading (Minigame)

```javascript
class InteractiveLoadingScreen extends LoadingScreen {
  createElement() {
    const div = super.createElement();
    div.innerHTML += `
      <div class="minigame-container">
        <p class="minigame-prompt">Tap the circles while loading!</p>
        <div class="minigame-area"></div>
      </div>
    `;
    return div;
  }

  startMinigame() {
    // Spawn clickable circles during loading
    this.minigameInterval = setInterval(() => {
      this.spawnCircle();
    }, 1000);
  }
}
```

### Atmospheric Loading

```javascript
class AtmosphericLoadingScreen extends LoadingScreen {
  createElement() {
    const div = super.createElement();
    div.innerHTML = `
      <div class="atmospheric-bg">
        <div class="fog-layer"></div>
        <div class="floating-particles"></div>
      </div>
      <div class="loading-text">
        <p>Entering the shadows...</p>
      </div>
    `;
    return div;
  }

  createParticles() {
    // Use canvas-based particle system
    const canvas = document.createElement('canvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    // ... particle animation
  }
}
```

---

## Common Mistakes Beginners Make

### 1. No Progress Feedback

```javascript
// ❌ WRONG: Silent loading
async function loadGame() {
  await loadAssets();
  await loadWorld();
  startGame();
}
// Player thinks game froze

// ✅ CORRECT: Show progress
async function loadGame() {
  loadingScreen.setProgress(0, 'Loading assets...');
  await loadAssets();
  loadingScreen.setProgress(50, 'Loading world...');
  await loadWorld();
  loadingScreen.setProgress(100, 'Ready!');
}
// Player knows what's happening
```

### 2. Instant Disappearance

```javascript
// ❌ WRONG: Immediate hide
loadingComplete() {
  hideLoadingScreen();
  startGame();
}
// Jarring transition

// ✅ CORRECT: Smooth transition
async loadingComplete() {
  loadingScreen.setProgress(100, 'Complete!');
  await delay(1000);  // Let player see completion
  await fadeOutLoadingScreen();
  startGame();
}
// Satisfying conclusion
```

### 3. Fake Progress

```javascript
// ❌ WRONG: Fake progress doesn't match reality
setInterval(() => {
  progress += Math.random() * 10;
  updateBar(progress);
}, 500);
// Player knows it's fake, breaks trust

// ✅ CORRECT: Real progress based on actual loading
for (const asset of assets) {
  await loadAsset(asset);
  progress = (loaded / total) * 100;
  updateBar(progress);
}
// Accurate feedback builds confidence
```

### 4. Too Short Display

```javascript
// ❌ WRONG: Flashes by too fast
async function loadGame() {
  showLoading();
  await loadAssets();  // Only takes 0.5 seconds!
  hideLoading();
}
// Loading screen flickers, annoying

// ✅ CORRECT: Minimum display time
async function loadGame() {
  const startTime = Date.now();
  showLoading();
  await loadAssets();
  const elapsed = Date.now() - startTime;
  if (elapsed < 2000) {
    await delay(2000 - elapsed);
  }
  hideLoading();
}
// Player can read and appreciate the screen
```

---

## Performance Considerations

```
LOADING SCREEN PERFORMANCE:

Visual Effects:
├── Particles: CSS animations (GPU-accelerated)
├── Progress bar: Simple width change (negligible)
├── Tip rotation: Text fade (minimal)
└── Impact: Negligible

Memory:
├── DOM elements: Static, created once
├── No assets loaded during screen display
└── Impact: Minimal overhead

Optimization:
├── Use CSS animations (not JS)
├── Avoid creating elements during load
├── Don't load heavy assets for the screen itself
└── Keep it simple and lightweight
```

---

## CSS for Loading Screen

```css
.loading-screen {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.loading-background {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

.loading-particles {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

.loading-particle {
  position: absolute;
  background: rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  animation: float-up linear infinite;
}

@keyframes float-up {
  0% {
    transform: translateY(100vh) scale(0);
    opacity: 0;
  }
  10% {
    opacity: 1;
  }
  90% {
    opacity: 1;
  }
  100% {
    transform: translateY(-10vh) scale(1);
    opacity: 0;
  }
}

.loading-content {
  position: relative;
  text-align: center;
  color: white;
  z-index: 1;
}

.loading-logo h1 {
  font-size: 3rem;
  margin-bottom: 0.5rem;
  text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
}

.logo-subtitle {
  font-size: 1.2rem;
  color: rgba(255, 255, 255, 0.7);
  margin-bottom: 3rem;
}

.progress-container {
  max-width: 500px;
  margin: 0 auto 2rem;
}

.progress-bar-bg {
  width: 100%;
  height: 8px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  overflow: hidden;
}

.progress-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #00ff88, #00ccff);
  border-radius: 4px;
  width: 0%;
  transition: width 0.3s ease, background-color 0.3s ease;
  box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
}

.progress-text {
  display: flex;
  justify-content: space-between;
  margin-top: 0.5rem;
  font-size: 0.9rem;
  color: rgba(255, 255, 255, 0.8);
}

.loading-tips {
  max-width: 400px;
  margin: 0 auto;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 8px;
  min-height: 80px;
  display: flex;
  align-items: center;
  gap: 1rem;
}

.tip-icon {
  width: 40px;
  height: 40px;
  background: rgba(0, 255, 136, 0.2);
  border-radius: 50%;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
}

.tip-text {
  text-align: left;
  font-style: italic;
  color: rgba(255, 255, 255, 0.8);
  transition: opacity 0.3s ease;
}

.loading-indicators {
  display: flex;
  justify-content: center;
  gap: 0.5rem;
  margin-top: 2rem;
}

.indicator-dot {
  width: 12px;
  height: 12px;
  background: rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  transition: opacity 0.3s ease, transform 0.3s ease;
}
```

---

## Related Systems

- [UIManager](./ui-manager.md) - Screen orchestration
- [Start Screen](./start-screen.md) - Main menu
- [AssetLoader](../03-content/asset-loader.md) - Asset loading
- [GameManager](../02-core/game-manager.md) - Game state

---

## Source File Reference

**Primary Files**:
- `../src/ui/LoadingScreen.js` - Loading screen component (estimated)
- `../src/ui/LoadingManager.js` - Loading process coordination (estimated)

**Key Classes**:
- `LoadingScreen` - UI display for loading
- `LoadingManager` - Loading process coordination

**Dependencies**:
- UIManager (screen management)
- AssetLoader (asset loading)
- GameManager (events)

---

## References

- [MDN: Progress Element](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/progress) - Native progress
- [CSS Animations](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Animations) - Smooth effects
- [Web Performance Loading](https://web.dev/fast/#load-fast) - Loading optimization

*Documentation last updated: January 12, 2026*
