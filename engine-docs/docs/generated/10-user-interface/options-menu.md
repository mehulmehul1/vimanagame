# Options Menu - First Principles Guide

## Overview

The **Options Menu** (also called Settings) is where players customize their experience. It's critical for accessibility (different players have different needs), performance (allow players to adjust for their hardware), and personalization (let players play how they want). A good options menu is comprehensive without being overwhelming, organized logically, and provides immediate feedback.

Think of the options menu as the **"control panel"**—like the settings on a car's dashboard or a physical device, it gives players control over how the game behaves, looks, and sounds.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Give players agency and comfort. When players can adjust the game to their preferences, they feel more in control and can focus on enjoying the experience rather than fighting with uncomfortable settings.

**Why a Comprehensive Options Menu?**
- **Accessibility**: Not everyone sees, hears, or plays the same way
- **Performance**: Let players balance quality vs. frame rate
- **Comfort**: Subtitles, colorblind modes, motion sensitivity
- **Inclusivity**: More players can enjoy your game
- **Longevity**: Players can tweak as hardware/needs change

**Player Psychology**:
```
Open Options → "I can customize this?" → Empowerment
     ↓
Adjust Setting → See immediate effect → Control
     ↓
Game Feels Better → "This is my experience now" → Ownership
     ↓
Play Longer → Comfortable customization = retention
```

### Design Principles

**1. Categorize Logically**
Don't dump everything in one list. Use categories:
- **Graphics/Display**: Resolution, quality, fullscreen
- **Audio**: Volume, subtitles
- **Controls**: Keybindings, sensitivity
- **Accessibility**: Colorblind, motion controls
- **Gameplay**: Difficulty, auto-save

**2. Immediate Feedback**
Settings should apply right away when possible:
- Sliders show current value
- Graphics changes preview immediately
- Audio sliders test with sound
- No "restart required" for basic settings

**3. Save Immediately**
Auto-save settings as they change. Don't make players manually save—if they change something, it should stick.

**4. Show Impact**
Help players understand what settings do:
- Show approximate FPS impact for graphics
- Show "High/Medium/Low" labels for quality
- Show examples for accessibility options

### Performance Profiles

Performance profiles (presets) are essential for players who don't want to tweak individual settings. They provide:

```
PRESET PROFILES:

┌─────────────────────────────────────────────────────────┐
│  ULTRA         HIGH         MEDIUM        LOW           │
│  ─────────     ─────────     ─────────     ─────────     │
│  Max Quality   Beautiful    Balanced     Performance    │
│  60+ FPS       60 FPS        60 FPS       60+ FPS       │
│  RTX On        High Shadows  Med Shadows  Low Shadows   │
│  Full Res      1080p         900p         720p          │
│  All Effects   Most Effects  Some Effects Minimal       │
│                                                         │
│  For:          For:          For:          For:          │
│  High-end PCs  Good PCs      Mid-range     Low-end       │
│  RTX cards     Modern GPUs   Older GPUs    Integrated    │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the options menu, you should know:
- **localStorage** - Saving player preferences
- **DOM manipulation** - Creating interactive form elements
- **Event handling** - Responding to setting changes
- **Audio contexts** - Controlling volume
- **Canvas/display** - Changing resolution and fullscreen

### Core Architecture

```
OPTIONS MENU ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│                    OPTIONS MENU                         │
│  - Tab navigation (Graphics/Audio/Controls/Access)      │
│  - Setting value storage                                │
│  - Preset/profile management                            │
│  - Save/load configuration                              │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   TABS        │  │   PRESETS    │  │   SETTINGS   │
│  - Graphics   │  │  - Ultra     │  │  - Sliders   │
│  - Audio      │  │  - High      │  │  - Toggles   │
│  - Controls   │  │  - Medium    │  │  - Dropdowns │
│  - Access     │  │  - Low       │  │  - Binds     │
└───────────────┘  └───────────────┘  └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   STORAGE    │
                    │  - localStorage
                    │  - Apply to engine
                    └───────────────┘
```

### OptionsMenu Class

```javascript
class OptionsMenu extends UIScreen {
  constructor(options = {}) {
    super({
      ...options,
      name: 'options',
      isOverlay: false
    });

    // Settings storage
    this.settings = this.getDefaultSettings();
    this.currentTab = 'graphics';
    this.pendingChanges = new Set();

    // Configuration
    this.config = {
      showPerformancePresets: true,
      showAdvancedOptions: false,
      requireRestartForGraphics: false
    };

    Object.assign(this.config, options.config || {});

    // Performance presets
    this.presets = {
      ultra: this.getUltraPreset(),
      high: this.getHighPreset(),
      medium: this.getMediumPreset(),
      low: this.getLowPreset(),
      custom: null  // Represents manually configured settings
    };
  }

  /**
   * Get default settings
   */
  getDefaultSettings() {
    return {
      // Graphics
      graphics: {
        preset: 'high',
        resolution: 'native',
        fullscreen: true,
        vsync: true,
        antialiasing: 'fxaa',
        textureQuality: 'high',
        shadowQuality: 'high',
        reflectionQuality: 'medium',
        particleQuality: 'high',
        postProcessing: true,
        bloom: true,
        motionBlur: false,
        ambientOcclusion: true,
        renderScale: 1.0
      },

      // Audio
      audio: {
        masterVolume: 0.8,
        musicVolume: 0.7,
        sfxVolume: 0.8,
        voiceVolume: 1.0,
        subtitles: true,
        subtitleSize: 'medium',
        subtitleBackground: true
      },

      // Controls
      controls: {
        mouseSensitivity: 1.0,
        gamepadSensitivity: 1.0,
        invertY: false,
        vibration: true,
        aimAssist: false,
        keybindings: this.getDefaultKeybindings()
      },

      // Accessibility
      accessibility: {
        colorblindMode: 'none',
        highContrast: false,
        reduceMotion: false,
        cameraShake: true,
        holdToSkip: false,
        skipHoldTime: 1.0,
        tutorialPopups: true,
        gameplayDifficulty: 'normal'
      }
    };
  }

  /**
   * Get default keybindings
   */
  getDefaultKeybindings() {
    return {
      forward: ['KeyW', 'ArrowUp'],
      backward: ['KeyS', 'ArrowDown'],
      left: ['KeyA', 'ArrowLeft'],
      right: ['KeyD', 'ArrowRight'],
      jump: ['Space'],
      interact: ['KeyE'],
      run: ['ShiftLeft', 'ShiftRight'],
      crouch: ['KeyC'],
      pause: ['Escape'],
      quickSave: ['F5'],
      quickLoad: ['F9']
    };
  }

  /**
   * Create options menu DOM
   */
  createElement() {
    const div = document.createElement('div');
    div.className = 'options-menu';
    div.innerHTML = `
      <div class="options-background"></div>

      <div class="options-content">
        <!-- Header -->
        <div class="options-header">
          <h1 class="options-title">Options</h1>
          <button class="back-button" data-action="back">
            <span class="back-icon">←</span>
            <span class="back-label">Back</span>
          </button>
        </div>

        <!-- Preset Selection (top bar) -->
        <div class="preset-bar">
          <span class="preset-label">Performance Profile:</span>
          <div class="preset-buttons">
            <button class="preset-btn" data-preset="ultra">Ultra</button>
            <button class="preset-btn" data-preset="high">High</button>
            <button class="preset-btn" data-preset="medium">Medium</button>
            <button class="preset-btn" data-preset="low">Low</button>
          </div>
        </div>

        <!-- Main Options Area -->
        <div class="options-main">
          <!-- Tab Navigation -->
          <nav class="options-tabs">
            <button class="tab-btn active" data-tab="graphics">
              <span class="tab-icon">🎨</span>
              <span class="tab-label">Graphics</span>
            </button>
            <button class="tab-btn" data-tab="audio">
              <span class="tab-icon">🔊</span>
              <span class="tab-label">Audio</span>
            </button>
            <button class="tab-btn" data-tab="controls">
              <span class="tab-icon">🎮</span>
              <span class="tab-label">Controls</span>
            </button>
            <button class="tab-btn" data-tab="accessibility">
              <span class="tab-icon">♿</span>
              <span class="tab-label">Accessibility</span>
            </button>
          </nav>

          <!-- Tab Content -->
          <div class="options-tabs-content">
            <div class="tab-content active" data-tab="graphics">
              ${this.createGraphicsTab()}
            </div>
            <div class="tab-content" data-tab="audio">
              ${this.createAudioTab()}
            </div>
            <div class="tab-content" data-tab="controls">
              ${this.createControlsTab()}
            </div>
            <div class="tab-content" data-tab="accessibility">
              ${this.createAccessibilityTab()}
            </div>
          </div>
        </div>

        <!-- Footer Info -->
        <div class="options-footer">
          <span class="apply-status">Settings auto-save</span>
          <span class="resolution-info">Current: ${this.getCurrentResolution()}</span>
        </div>
      </div>
    `;

    // Store references
    this.backButton = div.querySelector('.back-button');
    this.tabButtons = Array.from(div.querySelectorAll('.tab-btn'));
    this.tabContents = Array.from(div.querySelectorAll('.tab-content'));
    this.presetButtons = Array.from(div.querySelectorAll('.preset-btn'));
    this.resolutionInfo = div.querySelector('.resolution-info');
    this.applyStatus = div.querySelector('.apply-status');

    return div;
  }

  /**
   * Create graphics tab content
   */
  createGraphicsTab() {
    return `
      <div class="options-section">
        <h2 class="section-title">Display</h2>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Resolution</label>
            <span class="setting-desc">Game render resolution</span>
          </div>
          <select class="setting-select" data-setting="resolution">
            <option value="native">Native (Recommended)</option>
            <option value="1920x1080">1920×1080</option>
            <option value="1600x900">1600×900</option>
            <option value="1280x720">1280×720</option>
            <option value="854x480">854×480</option>
          </select>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Fullscreen</label>
            <span class="setting-desc">Display in fullscreen mode</span>
          </div>
          <label class="toggle-switch">
            <input type="checkbox" data-setting="fullscreen" checked>
            <span class="toggle-slider"></span>
          </label>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">V-Sync</label>
            <span class="setting-desc">Synchronize framerate with display</span>
          </div>
          <label class="toggle-switch">
            <input type="checkbox" data-setting="vsync" checked>
            <span class="toggle-slider"></span>
          </label>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Render Scale</label>
            <span class="setting-desc">Resolution multiplier (lower = better performance)</span>
          </div>
          <div class="slider-control">
            <input type="range" class="setting-slider" data-setting="renderScale"
                   min="0.5" max="1.5" step="0.1" value="1.0">
            <span class="slider-value">100%</span>
          </div>
        </div>
      </div>

      <div class="options-section">
        <h2 class="section-title">Quality</h2>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Anti-Aliasing</label>
            <span class="setting-desc">Smooth jagged edges</span>
          </div>
          <select class="setting-select" data-setting="antialiasing">
            <option value="none">Off</option>
            <option value="fxaa">FXAA (Fast)</option>
            <option value="smaa">SMAA (Quality)</option>
            <option value="taa">TAA (Temporal)</option>
          </select>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Texture Quality</label>
            <span class="setting-desc">Higher = more VRAM usage</span>
          </div>
          <select class="setting-select" data-setting="textureQuality">
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
            <option value="ultra">Ultra</option>
          </select>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Shadow Quality</label>
            <span class="setting-desc">Higher = more realistic lighting</span>
          </div>
          <select class="setting-select" data-setting="shadowQuality">
            <option value="off">Off</option>
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
            <option value="ultra">Ultra</option>
          </select>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Post Processing</label>
            <span class="setting-desc">Bloom, color grading, etc.</span>
          </div>
          <label class="toggle-switch">
            <input type="checkbox" data-setting="postProcessing" checked>
            <span class="toggle-slider"></span>
          </label>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Motion Blur</label>
            <span class="setting-desc">Blur during camera movement</span>
          </div>
          <label class="toggle-switch">
            <input type="checkbox" data-setting="motionBlur">
            <span class="toggle-slider"></span>
          </label>
        </div>
      </div>
    `;
  }

  /**
   * Create audio tab content
   */
  createAudioTab() {
    return `
      <div class="options-section">
        <h2 class="section-title">Volume</h2>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Master Volume</label>
            <span class="setting-desc">Overall volume level</span>
          </div>
          <div class="slider-control">
            <input type="range" class="setting-slider volume-slider" data-setting="masterVolume"
                   min="0" max="1" step="0.05" value="0.8">
            <span class="slider-value">80%</span>
            <button class="test-sound-btn" data-sound="master">🔊 Test</button>
          </div>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Music Volume</label>
            <span class="setting-desc">Background music volume</span>
          </div>
          <div class="slider-control">
            <input type="range" class="setting-slider volume-slider" data-setting="musicVolume"
                   min="0" max="1" step="0.05" value="0.7">
            <span class="slider-value">70%</span>
            <button class="test-sound-btn" data-sound="music">🎵 Test</button>
          </div>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">SFX Volume</label>
            <span class="setting-desc">Sound effects volume</span>
          </div>
          <div class="slider-control">
            <input type="range" class="setting-slider volume-slider" data-setting="sfxVolume"
                   min="0" max="1" step="0.05" value="0.8">
            <span class="slider-value">80%</span>
            <button class="test-sound-btn" data-sound="sfx">💥 Test</button>
          </div>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Voice Volume</label>
            <span class="setting-desc">Dialog/voiceover volume</span>
          </div>
          <div class="slider-control">
            <input type="range" class="setting-slider volume-slider" data-setting="voiceVolume"
                   min="0" max="1" step="0.05" value="1.0">
            <span class="slider-value">100%</span>
            <button class="test-sound-btn" data-sound="voice">🗣️ Test</button>
          </div>
        </div>
      </div>

      <div class="options-section">
        <h2 class="section-title">Subtitles</h2>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Enable Subtitles</label>
            <span class="setting-desc">Show subtitles for dialog</span>
          </div>
          <label class="toggle-switch">
            <input type="checkbox" data-setting="subtitles" checked>
            <span class="toggle-slider"></span>
          </label>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Subtitle Size</label>
            <span class="setting-desc">Text size for subtitles</span>
          </div>
          <select class="setting-select" data-setting="subtitleSize">
            <option value="small">Small</option>
            <option value="medium" selected>Medium</option>
            <option value="large">Large</option>
            <option value="extra-large">Extra Large</option>
          </select>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Subtitle Background</label>
            <span class="setting-desc">Dark background behind text</span>
          </div>
          <label class="toggle-switch">
            <input type="checkbox" data-setting="subtitleBackground" checked>
            <span class="toggle-slider"></span>
          </label>
        </div>
      </div>
    `;
  }

  /**
   * Create controls tab content
   */
  createControlsTab() {
    return `
      <div class="options-section">
        <h2 class="section-title">Input Sensitivity</h2>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Mouse Sensitivity</label>
            <span class="setting-desc">Camera look sensitivity</span>
          </div>
          <div class="slider-control">
            <input type="range" class="setting-slider" data-setting="mouseSensitivity"
                   min="0.1" max="3.0" step="0.1" value="1.0">
            <span class="slider-value">1.0</span>
          </div>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Invert Y-Axis</label>
            <span class="setting-desc">Invert vertical camera control</span>
          </div>
          <label class="toggle-switch">
            <input type="checkbox" data-setting="invertY">
            <span class="toggle-slider"></span>
          </label>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Vibration</label>
            <span class="setting-desc">Gamepad vibration feedback</span>
          </div>
          <label class="toggle-switch">
            <input type="checkbox" data-setting="vibration" checked>
            <span class="toggle-slider"></span>
          </label>
        </div>
      </div>

      <div class="options-section">
        <h2 class="section-title">Key Bindings</h2>

        <div class="keybindings-list">
          ${Object.entries(this.getDefaultKeybindings()).map(([action, keys]) => `
            <div class="keybinding-row" data-action="${action}">
              <span class="keybinding-action">${this.formatActionName(action)}</span>
              <div class="keybinding-buttons">
                ${keys.map((key, i) => `
                  <button class="key-binding-btn" data-index="${i}">${this.formatKeyName(key)}</button>
                `).join('')}
                <button class="add-binding-btn" ${keys.length >= 2 ? 'disabled' : ''}>+</button>
              </div>
            </div>
          `).join('')}
        </div>

        <button class="reset-bindings-btn">Reset to Defaults</button>
      </div>
    `;
  }

  /**
   * Create accessibility tab content
   */
  createAccessibilityTab() {
    return `
      <div class="options-section">
        <h2 class="section-title">Visual</h2>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Colorblind Mode</label>
            <span class="setting-desc">Adjust colors for color vision deficiency</span>
          </div>
          <select class="setting-select" data-setting="colorblindMode">
            <option value="none">Off</option>
            <option value="protanopia">Protanopia (Red-Blind)</option>
            <option value="deuteranopia">Deuteranopia (Green-Blind)</option>
            <option value="tritanopia">Tritanopia (Blue-Blind)</option>
          </select>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">High Contrast</label>
            <span class="setting-desc">Increase contrast for better visibility</span>
          </div>
          <label class="toggle-switch">
            <input type="checkbox" data-setting="highContrast">
            <span class="toggle-slider"></span>
          </label>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Reduce Motion</label>
            <span class="setting-desc">Disable camera shake and motion blur</span>
          </div>
          <label class="toggle-switch">
            <input type="checkbox" data-setting="reduceMotion">
            <span class="toggle-slider"></span>
          </label>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Camera Shake</label>
            <span class="setting-desc">Camera shake during impacts</span>
          </div>
          <label class="toggle-switch">
            <input type="checkbox" data-setting="cameraShake" checked>
            <span class="toggle-slider"></span>
          </label>
        </div>
      </div>

      <div class="options-section">
        <h2 class="section-title">Gameplay</h2>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Difficulty</label>
            <span class="setting-desc">Game difficulty level</span>
          </div>
          <select class="setting-select" data-setting="gameplayDifficulty">
            <option value="easy">Easy</option>
            <option value="normal" selected>Normal</option>
            <option value="hard">Hard</option>
            <option value="nightmare">Nightmare</option>
          </select>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Tutorial Popups</label>
            <span class="setting-desc">Show helpful hints during gameplay</span>
          </div>
          <label class="toggle-switch">
            <input type="checkbox" data-setting="tutorialPopups" checked>
            <span class="toggle-slider"></span>
          </label>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <label class="setting-label">Hold to Skip</label>
            <span class="setting-desc">Hold button to skip cutscenes</span>
          </div>
          <label class="toggle-switch">
            <input type="checkbox" data-setting="holdToSkip">
            <span class="toggle-slider"></span>
          </label>
        </div>
      </div>
    `;
  }

  /**
   * Format action name for display
   */
  formatActionName(action) {
    return action.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase());
  }

  /**
   * Format key name for display
   */
  formatKeyName(key) {
    const keyNames = {
      'Space': 'SPACE',
      'ShiftLeft': 'L-SHIFT',
      'ShiftRight': 'R-SHIFT',
      'ControlLeft': 'L-CTRL',
      'ControlRight': 'R-CTRL',
      'ArrowUp': '↑',
      'ArrowDown': '↓',
      'ArrowLeft': '←',
      'ArrowRight': '→'
    };
    return keyNames[key] || key.replace(/^Key/, '');
  }

  /**
   * Initialize options menu
   */
  initialize() {
    // Load saved settings
    this.loadSettings();

    // Setup tab navigation
    this.setupTabs();

    // Setup preset buttons
    this.setupPresets();

    // Setup setting controls
    this.setupSettings();

    // Setup back button
    this.backButton.addEventListener('click', () => {
      this.uiManager.goBack();
    });
  }

  /**
   * Setup tab navigation
   */
  setupTabs() {
    this.tabButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;
        this.switchTab(tabName);
      });
    });
  }

  /**
   * Switch to a tab
   */
  switchTab(tabName) {
    // Update buttons
    this.tabButtons.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.tab === tabName);
    });

    // Update content
    this.tabContents.forEach(content => {
      content.classList.toggle('active', content.dataset.tab === tabName);
    });

    this.currentTab = tabName;
  }

  /**
   * Setup preset buttons
   */
  setupPresets() {
    this.presetButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const preset = btn.dataset.preset;
        this.applyPreset(preset);
      });
    });
  }

  /**
   * Apply a performance preset
   */
  applyPreset(presetName) {
    const preset = this.presets[presetName];
    if (!preset) return;

    // Update settings
    Object.assign(this.settings.graphics, preset);

    // Update UI controls
    this.updateGraphicsControls();

    // Apply settings
    this.applyGraphicsSettings();

    // Update preset button states
    this.presetButtons.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.preset === presetName);
    });

    // Save
    this.saveSettings();

    // Show feedback
    this.showApplyStatus(`Applied ${presetName.toUpperCase()} preset`);
  }

  /**
   * Update graphics control values to match settings
   */
  updateGraphicsControls() {
    const graphics = this.settings.graphics;

    // Update selects
    this.element.querySelectorAll('[data-setting]').forEach(input => {
      const setting = input.dataset.setting;
      const category = this.getSettingCategory(setting);

      if (category === 'graphics' && graphics[setting] !== undefined) {
        if (input.type === 'checkbox') {
          input.checked = graphics[setting];
        } else if (input.type === 'range') {
          input.value = graphics[setting];
          const valueDisplay = input.parentElement.querySelector('.slider-value');
          if (valueDisplay) {
            valueDisplay.textContent = this.formatSliderValue(setting, graphics[setting]);
          }
        } else {
          input.value = graphics[setting];
        }
      }
    });
  }

  /**
   * Setup setting controls
   */
  setupSettings() {
    // Selects and checkboxes
    this.element.querySelectorAll('[data-setting]').forEach(input => {
      const setting = input.dataset.setting;

      if (input.type === 'checkbox') {
        input.addEventListener('change', () => {
          this.updateSetting(setting, input.checked);
        });
      } else if (input.type === 'range') {
        input.addEventListener('input', () => {
          const value = parseFloat(input.value);
          const valueDisplay = input.parentElement.querySelector('.slider-value');
          if (valueDisplay) {
            valueDisplay.textContent = this.formatSliderValue(setting, value);
          }
          // Update real-time for sliders
          this.updateSetting(setting, value);
        });
      } else {
        input.addEventListener('change', () => {
          this.updateSetting(setting, input.value);
        });
      }
    });

    // Volume test buttons
    this.element.querySelectorAll('.test-sound-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        this.testSound(btn.dataset.sound);
      });
    });

    // Key binding buttons
    this.element.querySelectorAll('.key-binding-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        this.remapKey(btn);
      });
    });

    // Reset bindings button
    const resetBtn = this.element.querySelector('.reset-bindings-btn');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        this.resetKeybindings();
      });
    }
  }

  /**
   * Update a single setting
   */
  updateSetting(setting, value) {
    const category = this.getSettingCategory(setting);

    if (category && this.settings[category]) {
      this.settings[category][setting] = value;

      // Apply immediately
      this.applySetting(category, setting, value);

      // Save
      this.saveSettings();

      // Check if custom preset
      this.checkCustomPreset();
    }
  }

  /**
   * Get the category for a setting
   */
  getSettingCategory(setting) {
    if (this.settings.graphics[setting] !== undefined) return 'graphics';
    if (this.settings.audio[setting] !== undefined) return 'audio';
    if (this.settings.controls[setting] !== undefined) return 'controls';
    if (this.settings.accessibility[setting] !== undefined) return 'accessibility';
    return null;
  }

  /**
   * Apply a setting
   */
  applySetting(category, setting, value) {
    switch (category) {
      case 'graphics':
        this.applyGraphicsSetting(setting, value);
        break;
      case 'audio':
        this.applyAudioSetting(setting, value);
        break;
      case 'controls':
        this.applyControlSetting(setting, value);
        break;
      case 'accessibility':
        this.applyAccessibilitySetting(setting, value);
        break;
    }
  }

  /**
   * Apply graphics setting
   */
  applyGraphicsSetting(setting, value) {
    const renderer = this.gameManager.renderer;

    switch (setting) {
      case 'fullscreen':
        this.applyFullscreen(value);
        break;
      case 'vsync':
        // VSync is typically set at renderer initialization
        this.gameManager.emit('graphics:vsyncChanged', { enabled: value });
        break;
      case 'renderScale':
        renderer.setPixelRatio(window.devicePixelRatio * value);
        break;
      case 'antialiasing':
        this.applyAntialiasing(value);
        break;
      case 'textureQuality':
        this.gameManager.emit('graphics:textureQualityChanged', { quality: value });
        break;
      case 'shadowQuality':
        this.applyShadowQuality(value);
        break;
      case 'postProcessing':
        this.gameManager.postProcessing.enabled = value;
        break;
    }
  }

  /**
   * Apply fullscreen setting
   */
  applyFullscreen(enabled) {
    if (enabled) {
      document.documentElement.requestFullscreen().catch(err => {
        console.log('Fullscreen request failed:', err);
      });
    } else if (document.fullscreenElement) {
      document.exitFullscreen();
    }
  }

  /**
   * Apply audio setting
   */
  applyAudioSetting(setting, value) {
    const audioManager = this.gameManager.audioManager;

    switch (setting) {
      case 'masterVolume':
        audioManager.setMasterVolume(value);
        break;
      case 'musicVolume':
        audioManager.setMusicVolume(value);
        break;
      case 'sfxVolume':
        audioManager.setSFXVolume(value);
        break;
      case 'voiceVolume':
        audioManager.setVoiceVolume(value);
        break;
      case 'subtitles':
        this.gameManager.emit('subtitles:toggled', { enabled: value });
        break;
    }
  }

  /**
   * Test a sound
   */
  testSound(type) {
    const audioManager = this.gameManager.audioManager;

    switch (type) {
      case 'master':
        audioManager.playSfx('menu_click');
        break;
      case 'music':
        audioManager.playMusic('menu_music', { duration: 2000 });
        break;
      case 'sfx':
        audioManager.playSfx('explosion');
        break;
      case 'voice':
        audioManager.playSfx('voice_sample');
        break;
    }
  }

  /**
   * Remap a key binding
   */
  remapKey(button) {
    const row = button.closest('.keybinding-row');
    const action = row.dataset.action;
    const index = parseInt(button.dataset.index);

    // Show waiting state
    button.textContent = 'Press any key...';
    button.classList.add('waiting');

    // Listen for next key press
    const handler = (e) => {
      e.preventDefault();

      // Update binding
      this.settings.controls.keybindings[action][index] = e.code;

      // Update button
      button.textContent = this.formatKeyName(e.code);
      button.classList.remove('waiting');

      // Save
      this.saveSettings();

      // Remove listener
      document.removeEventListener('keydown', handler);
    };

    document.addEventListener('keydown', handler, { once: true });
  }

  /**
   * Reset keybindings to defaults
   */
  resetKeybindings() {
    this.settings.controls.keybindings = this.getDefaultKeybindings();

    // Re-render keybindings
    const keybindingsList = this.element.querySelector('.keybindings-list');
    if (keybindingsList) {
      // Rebuild the list
      keybindingsList.innerHTML = Object.entries(this.settings.controls.keybindings).map(([action, keys]) => `
        <div class="keybinding-row" data-action="${action}">
          <span class="keybinding-action">${this.formatActionName(action)}</span>
          <div class="keybinding-buttons">
            ${keys.map((key, i) => `
              <button class="key-binding-btn" data-index="${i}">${this.formatKeyName(key)}</button>
            `).join('')}
            <button class="add-binding-btn" ${keys.length >= 2 ? 'disabled' : ''}>+</button>
          </div>
        </div>
      `).join('');

      // Re-setup binding buttons
      keybindingsList.querySelectorAll('.key-binding-btn').forEach(btn => {
        btn.addEventListener('click', () => this.remapKey(btn));
      });
    }

    this.saveSettings();
  }

  /**
   * Format slider value for display
   */
  formatSliderValue(setting, value) {
    if (setting === 'renderScale') {
      return `${Math.round(value * 100)}%`;
    }
    if (setting.includes('Volume') || setting.includes('Sensitivity')) {
      return `${Math.round(value * 100)}%`;
    }
    return value.toString();
  }

  /**
   * Check if settings are now custom
   */
  checkCustomPreset() {
    // Compare current graphics settings against presets
    const current = this.settings.graphics;
    let isCustom = true;

    for (const [name, preset] of Object.entries(this.presets)) {
      if (name === 'custom') continue;

      if (this.matchesPreset(current, preset)) {
        isCustom = false;
        this.presetButtons.forEach(btn => {
          btn.classList.toggle('active', btn.dataset.preset === name);
        });
        break;
      }
    }

    if (isCustom) {
      this.presetButtons.forEach(btn => btn.classList.remove('active'));
    }
  }

  /**
   * Check if settings match a preset
   */
  matchesPreset(current, preset) {
    for (const key of Object.keys(preset)) {
      if (current[key] !== preset[key]) {
        return false;
      }
    }
    return true;
  }

  /**
   * Get current resolution string
   */
  getCurrentResolution() {
    return `${window.innerWidth}×${window.innerHeight}`;
  }

  /**
   * Show apply status message
   */
  showApplyStatus(message) {
    this.applyStatus.textContent = message;
    this.applyStatus.classList.add('show');

    setTimeout(() => {
      this.applyStatus.classList.remove('show');
    }, 2000);
  }

  /**
   * Load settings from storage
   */
  loadSettings() {
    try {
      const saved = localStorage.getItem('game_settings');
      if (saved) {
        const parsed = JSON.parse(saved);
        // Merge with defaults to handle any new settings
        this.settings = this.deepMerge(this.settings, parsed);
      }
    } catch (e) {
      console.warn('Failed to load settings:', e);
    }

    // Apply loaded settings
    this.applyAllSettings();
  }

  /**
   * Save settings to storage
   */
  saveSettings() {
    try {
      localStorage.setItem('game_settings', JSON.stringify(this.settings));
    } catch (e) {
      console.warn('Failed to save settings:', e);
    }
  }

  /**
   * Deep merge objects
   */
  deepMerge(target, source) {
    const result = { ...target };

    for (const key of Object.keys(source)) {
      if (source[key] instanceof Object && key in target) {
        result[key] = this.deepMerge(target[key], source[key]);
      } else {
        result[key] = source[key];
      }
    }

    return result;
  }

  /**
   * Apply all settings
   */
  applyAllSettings() {
    for (const category of Object.keys(this.settings)) {
      for (const [setting, value] of Object.entries(this.settings[category])) {
        this.applySetting(category, setting, value);
      }
    }
  }

  /**
   * Called when screen is shown
   */
  onShow(data) {
    // Refresh current resolution display
    if (this.resolutionInfo) {
      this.resolutionInfo.textContent = `Current: ${this.getCurrentResolution()}`;
    }

    // Load latest settings
    this.loadSettings();

    // Update controls to match
    this.updateGraphicsControls();
  }

  /**
   * Performance presets
   */
  getUltraPreset() {
    return {
      resolution: 'native',
      antialiasing: 'taa',
      textureQuality: 'ultra',
      shadowQuality: 'ultra',
      reflectionQuality: 'ultra',
      particleQuality: 'ultra',
      postProcessing: true,
      bloom: true,
      motionBlur: true,
      ambientOcclusion: true,
      renderScale: 1.0
    };
  }

  getHighPreset() {
    return {
      resolution: 'native',
      antialiasing: 'smaa',
      textureQuality: 'high',
      shadowQuality: 'high',
      reflectionQuality: 'high',
      particleQuality: 'high',
      postProcessing: true,
      bloom: true,
      motionBlur: false,
      ambientOcclusion: true,
      renderScale: 1.0
    };
  }

  getMediumPreset() {
    return {
      resolution: 'native',
      antialiasing: 'fxaa',
      textureQuality: 'medium',
      shadowQuality: 'medium',
      reflectionQuality: 'medium',
      particleQuality: 'medium',
      postProcessing: true,
      bloom: true,
      motionBlur: false,
      ambientOcclusion: false,
      renderScale: 0.9
    };
  }

  getLowPreset() {
    return {
      resolution: '1280x720',
      antialiasing: 'none',
      textureQuality: 'low',
      shadowQuality: 'low',
      reflectionQuality: 'off',
      particleQuality: 'low',
      postProcessing: false,
      bloom: false,
      motionBlur: false,
      ambientOcclusion: false,
      renderScale: 0.75
    };
  }
}

export default OptionsMenu;
```

---

## 📝 How To Build Options Like This

### Step 1: Create Simple Setting Control

```javascript
// Simple toggle
function createToggle(label, settingName, defaultValue) {
  const div = document.createElement('div');
  div.className = 'setting-row';
  div.innerHTML = `
    <label>${label}</label>
    <input type="checkbox" data-setting="${settingName}"
           ${defaultValue ? 'checked' : ''}>
  `;

  div.querySelector('input').addEventListener('change', (e) => {
    settings[settingName] = e.target.checked;
    saveSettings();
    applySetting(settingName, e.target.checked);
  });

  return div;
}

// Simple slider
function createSlider(label, settingName, min, max, value) {
  const div = document.createElement('div');
  div.className = 'setting-row';
  div.innerHTML = `
    <label>${label}</label>
    <div class="slider-container">
      <input type="range" min="${min}" max="${max}" value="${value}"
             data-setting="${settingName}">
      <span class="value">${value}</span>
    </div>
  `;

  const input = div.querySelector('input');
  input.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    div.querySelector('.value').textContent = val;
    settings[settingName] = val;
    saveSettings();
    applySetting(settingName, val);
  });

  return div;
}
```

### Step 2: Apply Settings to Game

```javascript
function applySetting(name, value) {
  switch (name) {
    case 'masterVolume':
      audioManager.gain.value = value;
      break;
    case 'fullscreen':
      if (value) {
        document.documentElement.requestFullscreen();
      }
      break;
    case 'mouseSensitivity':
      player.sensitivity = value;
      break;
    // ... more settings
  }
}
```

---

## 🔧 Common Setting Patterns

### Volume with Live Preview

```javascript
const volumeSlider = document.getElementById('volume');
volumeSlider.addEventListener('input', (e) => {
  const volume = parseFloat(e.target.value);
  audioManager.setVolume(volume);

  // Play preview sound on release
  if (!this.previewTimeout) {
    this.previewTimeout = setTimeout(() => {
      audioManager.playPreviewSound();
      this.previewTimeout = null;
    }, 500);
  }
});
```

### Key Binding Remapping

```javascript
function remapKey(action) {
  showMessage(`Press key for ${action}...`);

  const handler = (e) => {
    e.preventDefault();
    keybindings[action] = e.code;
    saveKeybindings();
    document.removeEventListener('keydown', handler);
    hideMessage();
  };

  document.addEventListener('keydown', handler, { once: true });
}
```

### Resolution with Confirmation

```javascript
async function changeResolution(resolution) {
  showConfirmDialog("Apply this resolution?", {
    onConfirm: () => {
      renderer.setSize(resolution.width, resolution.height);
      settings.resolution = resolution;
      saveSettings();
    },
    onCancel: () => {
      // Revert to previous
      cancelTimer = setTimeout(() => {
        renderer.setSize(prevResolution.width, prevResolution.height);
      }, 10000);  // 10 seconds to confirm
    }
  });
}
```

---

## Common Mistakes Beginners Make

### 1. Not Saving Settings

```javascript
// ❌ WRONG: Settings lost on refresh
setVolume(value) {
  this.volume = value;
}
// Player frustration

// ✅ CORRECT: Persist immediately
setVolume(value) {
  this.volume = value;
  localStorage.setItem('settings', JSON.stringify(settings));
}
// Settings remembered
```

### 2. Too Many Settings at Once

```javascript
// ❌ WRONG: Wall of options
<div>
  <!-- 50 settings all visible -->
</div>
// Overwhelming

// ✅ CORRECT: Categorized with tabs
<div>
  <nav>
    <button>Graphics</button>
    <button>Audio</button>
    <button>Controls</button>
  </nav>
  <div class="tab-content">
    <!-- 5-10 settings per tab -->
  </div>
</div>
// Manageable
```

### 3. No Visual Feedback

```javascript
// ❌ WRONG: Silent changes
<input type="range" onchange="updateValue()">
// Did it work?

// ✅ CORRECT: Show current value
<div class="slider-container">
  <input type="range" oninput="updateDisplay()">
  <span class="value-display">50%</span>
</div>
// Clear confirmation
```

### 4. Require Restart Too Often

```javascript
// ❌ WRONG: Restart for everything
setQuality(quality) {
  showRestartRequired();
}
// Disruptive

// ✅ CORRECT: Apply immediately when possible
setTextureQuality(quality) {
  // Can apply without restart
  renderer.textureQuality = quality;
}

setResolution(resolution) {
  // Only restart for major changes
  if (requiresRestart) {
    showRestartOption();
  }
}
```

---

## CSS for Options Menu

```css
.options-menu {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(10, 10, 26, 0.95);
  z-index: 100;
}

.options-content {
  display: flex;
  flex-direction: column;
  height: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  color: white;
}

.options-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.options-title {
  font-size: 2.5rem;
  text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
}

.preset-bar {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  margin-bottom: 2rem;
}

.preset-buttons {
  display: flex;
  gap: 0.5rem;
}

.preset-btn {
  padding: 0.5rem 1rem;
  background: rgba(255, 255, 255, 0.1);
  border: 2px solid rgba(255, 255, 255, 0.2);
  color: white;
  cursor: pointer;
  transition: all 0.3s ease;
}

.preset-btn:hover,
.preset-btn.active {
  background: rgba(0, 255, 136, 0.2);
  border-color: rgba(0, 255, 136, 0.5);
}

.options-main {
  display: flex;
  flex: 1;
  gap: 2rem;
  overflow: hidden;
}

.options-tabs {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  min-width: 150px;
}

.tab-btn {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.05);
  border: 2px solid transparent;
  border-radius: 8px;
  color: white;
  cursor: pointer;
  text-align: left;
  transition: all 0.3s ease;
}

.tab-btn:hover,
.tab-btn.active {
  background: rgba(0, 255, 136, 0.1);
  border-color: rgba(0, 255, 136, 0.3);
}

.options-tabs-content {
  flex: 1;
  overflow-y: auto;
}

.tab-content {
  display: none;
}

.tab-content.active {
  display: block;
}

.options-section {
  margin-bottom: 2rem;
}

.section-title {
  font-size: 1.5rem;
  margin-bottom: 1rem;
  color: rgba(0, 255, 136, 0.8);
}

.setting-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 8px;
  margin-bottom: 0.5rem;
}

.setting-info {
  flex: 1;
}

.setting-label {
  display: block;
  font-weight: 500;
}

.setting-desc {
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.6);
}

/* Toggle Switch */
.toggle-switch {
  position: relative;
  width: 50px;
  height: 26px;
}

.toggle-switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(255, 255, 255, 0.2);
  border-radius: 26px;
  transition: 0.3s;
}

.toggle-slider:before {
  position: absolute;
  content: "";
  height: 18px;
  width: 18px;
  left: 4px;
  bottom: 4px;
  background-color: white;
  border-radius: 50%;
  transition: 0.3s;
}

.toggle-switch input:checked + .toggle-slider {
  background-color: #00ff88;
}

.toggle-switch input:checked + .toggle-slider:before {
  transform: translateX(24px);
}

/* Slider Control */
.slider-control {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.setting-slider {
  width: 200px;
  height: 6px;
  -webkit-appearance: none;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 3px;
  outline: none;
}

.setting-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 16px;
  height: 16px;
  background: #00ff88;
  border-radius: 50%;
  cursor: pointer;
}

.slider-value {
  min-width: 50px;
  text-align: right;
}

/* Select */
.setting-select {
  padding: 0.5rem 1rem;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  color: white;
  cursor: pointer;
}

/* Key bindings */
.keybinding-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 1rem;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 8px;
  margin-bottom: 0.5rem;
}

.key-binding-btn {
  padding: 0.5rem 1rem;
  background: rgba(0, 255, 136, 0.2);
  border: 1px solid rgba(0, 255, 136, 0.3);
  border-radius: 4px;
  color: white;
  cursor: pointer;
  font-family: monospace;
}

.key-binding-btn.waiting {
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
```

---

## Related Systems

- [UIManager](./ui-manager.md) - Screen management
- [Start Screen](./start-screen.md) - Main menu
- [GameManager](../02-core/game-manager.md) - Settings application

---

## Source File Reference

**Primary Files**:
- `../src/ui/OptionsMenu.js` - Options menu component (estimated)

**Key Classes**:
- `OptionsMenu` - Settings interface
- Performance preset definitions

**Dependencies**:
- localStorage (settings persistence)
- AudioManager (volume control)
- Renderer (graphics settings)

---

## References

- [localStorage MDN](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage) - Saving settings
- [Fullscreen API](https://developer.mozilla.org/en-US/docs/Web/API/Fullscreen_API) - Fullscreen control
- [Gamepad API](https://developer.mozilla.org/en-US/docs/Web/API/Gamepad_API) - Controller support

*Documentation last updated: January 12, 2026*
