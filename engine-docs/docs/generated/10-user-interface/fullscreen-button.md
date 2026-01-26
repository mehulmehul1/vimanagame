# Fullscreen Button - First Principles Guide

## Overview

The **Fullscreen Button** is a small but important UI component that allows players to toggle fullscreen mode. While seemingly simple, it's essential for immersive gameplay—players want to experience games without browser chrome or desktop distractions. A well-implemented fullscreen button handles state correctly, provides clear feedback, and gracefully handles errors.

Think of the fullscreen button as the **"immersion switch"**—like a theater dimming its lights, it removes the barriers between player and game, allowing full focus on the experience.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Give players control over their experience boundaries. Some want maximum immersion (fullscreen), while others need quick access to other applications (windowed). Either way, the choice should be easy and obvious.

**Why a Fullscreen Toggle?**
- **Immersion**: Remove browser UI for complete focus
- **Performance**: Fullscreen often runs better
- **Player Choice**: Different preferences for different situations
- **Accessibility**: Some players need to see other windows
- **Convenience**: Easy toggle without digging through menus

**Player Experience Flow**:
```
Playing Game → "I want to be immersed" → Click Fullscreen
     ↓
Game Expands to Full Screen → "Now I'm really IN the game" → Immersion
     ↓
Playing → "Need to check something" → Press ESC or click button
     ↓
Game Returns to Windowed → "Quick access when needed" → Flexibility
```

### Design Principles

**1. Clear Visual State**
Players should know instantly whether fullscreen is active:
- Button appearance changes (icon, color)
- Tooltip shows current state and action
- Visual transition feedback

**2. Easy Access**
Don't bury the option:
- Visible from gameplay (HUD corner)
- Accessible from pause menu
- Keyboard shortcut (F11 common)

**3. Graceful Degradation**
Fullscreen can fail (browser restrictions, user denial):
- Don't break if fullscreen is denied
- Show helpful message explaining why
- Keep game playable in windowed mode

**4. State Persistence**
Remember player preference:
- Auto-enter fullscreen if last used
- Save preference to storage
- Respect system fullscreen vs exclusive

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the fullscreen button, you should know:
- **Fullscreen API** - `requestFullscreen()`, `exitFullscreen()`
- **Fullscreen events** - `fullscreenchange`, `fullscreenerror`
- **Browser restrictions** - Some browsers limit fullscreen
- **User gesture requirement** - Fullscreen must be triggered by user action
- **Different implementations** - Vendor prefixes exist

### Core Architecture

```
FULLSCREEN BUTTON ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│                FULLSCREEN BUTTON COMPONENT               │
│  - Button creation and styling                          │
│  - State tracking (isFullscreen)                        │
│  - Click handling with gesture requirement               │
│  - Error handling and fallback                          │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   ACTIONS     │  │   EVENTS     │  │   STORAGE    │
│  - Request    │  │  - On change │  │  - Save      │
│  - Exit       │  │  - On error  │  │  - Load      │
└───────────────┘  └───────────────┘  └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   BROWSER    │
                    │  API         │
                    │  Fullscreen  │
                    └───────────────┘
```

### FullscreenButton Class

```javascript
class FullscreenButton {
  constructor(options = {}) {
    this.gameManager = options.gameManager;
    this.container = options.container;
    this.position = options.position || 'bottom-right';  // top-left, top-right, bottom-left, bottom-right
    this.logger = options.logger || console;

    // State
    this.isFullscreen = false;
    this.isSupported = this.checkSupport();
    this.userPreference = false;

    // Configuration
    this.config = {
      showInHUD: options.showInHUD !== false,
      showInMenu: options.showInMenu !== false,
      autoEnter: options.autoEnter || false,
      keyboardShortcut: options.keyboardShortcut !== false,
      rememberPreference: options.rememberPreference !== false
    };

    // Icons (SVG paths)
    this.icons = {
      enter: 'M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z',  // Expand icon
      exit: 'M5 16h3v3h2v-5H5v2zm3-8H5v2h5V5H8v3zm6 11h2v-3h3v-2h-5v5zm2-11V5h-2v5h5V8h-3z',  // Shrink icon
      unsupported: 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zM4 12c0-4.42 3.58-8 8-8s8 3.58 8 8-3.58 8-8 8-8-3.58-8-8z M11 7h2v6h-2V7z'  // Not supported
    };

    // Create button
    this.createElement();

    // Setup event listeners
    this.setupEventListeners();

    // Load saved preference
    if (this.config.rememberPreference) {
      this.loadPreference();
    }

    // Auto-enter if configured and preferred
    if (this.config.autoEnter && this.userPreference) {
      this.requestFullscreen();
    }
  }

  /**
   * Check if fullscreen API is supported
   */
  checkSupport() {
    return !!(
      document.fullscreenEnabled ||
      document.webkitFullscreenEnabled ||
      document.mozFullScreenEnabled ||
      document.msFullscreenEnabled
    );
  }

  /**
   * Create button element
   */
  createElement() {
    this.button = document.createElement('button');
    this.button.className = 'fullscreen-button';
    this.button.setAttribute('aria-label', 'Toggle fullscreen');
    this.button.setAttribute('title', 'Toggle Fullscreen (F11)');

    // Create icon container
    const iconSpan = document.createElement('span');
    iconSpan.className = 'fullscreen-icon';
    iconSpan.innerHTML = this.getIconSVG('enter');
    this.button.appendChild(iconSpan);

    // Position styling
    const positions = {
      'top-left': { top: '10px', left: '10px', bottom: 'auto', right: 'auto' },
      'top-right': { top: '10px', right: '10px', bottom: 'auto', left: 'auto' },
      'bottom-left': { bottom: '10px', left: '10px', top: 'auto', right: 'auto' },
      'bottom-right': { bottom: '10px', right: '10px', top: 'auto', left: 'auto' }
    };

    const pos = positions[this.position] || positions['bottom-right'];
    this.button.style.cssText = `
      position: fixed;
      top: ${pos.top};
      left: ${pos.left};
      right: ${pos.right};
      bottom: ${pos.bottom};
      width: 40px;
      height: 40px;
      padding: 8px;
      background: rgba(0, 0, 0, 0.5);
      border: 2px solid rgba(0, 255, 136, 0.3);
      border-radius: 8px;
      cursor: pointer;
      z-index: 1000;
      opacity: 0;
      transition: opacity 0.3s ease, background 0.3s ease, transform 0.2s ease;
      backdrop-filter: blur(4px);
    `;

    // Hover effect (only on non-touch)
    if (!this.isTouchDevice()) {
      this.button.addEventListener('mouseenter', () => {
        this.button.style.background = 'rgba(0, 255, 136, 0.2)';
        this.button.style.transform = 'scale(1.1)';
      });
      this.button.addEventListener('mouseleave', () => {
        this.button.style.background = 'rgba(0, 0, 0, 0.5)';
        this.button.style.transform = 'scale(1)';
      });
    }

    // Add to container
    this.container.appendChild(this.button);

    // Update initial state
    this.updateState();
  }

  /**
   * Get SVG icon markup
   */
  getIconSVG(type) {
    const path = this.icons[type] || this.icons.enter;
    return `
      <svg viewBox="0 0 24 24" fill="white" width="100%" height="100%">
        <path d="${path}"/>
      </svg>
    `;
  }

  /**
   * Setup event listeners
   */
  setupEventListeners() {
    // Click handler
    this.button.addEventListener('click', () => this.toggle());

    // Fullscreen state change
    document.addEventListener('fullscreenchange', () => this.onFullscreenChange());
    document.addEventListener('webkitfullscreenchange', () => this.onFullscreenChange());
    document.addEventListener('mozfullscreenchange', () => this.onFullscreenChange());
    document.addEventListener('MSFullscreenChange', () => this.onFullscreenChange());

    // Fullscreen error
    document.addEventListener('fullscreenerror', (e) => this.onFullscreenError(e));
    document.addEventListener('webkitfullscreenerror', (e) => this.onFullscreenError(e));

    // Keyboard shortcut (F11)
    if (this.config.keyboardShortcut) {
      document.addEventListener('keydown', (e) => this.onKeyDown(e));
    }

    // Show on hover when in HUD
    if (this.config.showInHUD) {
      this.setupHUDVisibility();
    }
  }

  /**
   * Setup HUD visibility (show on hover)
   */
  setupHUDVisibility() {
    // Show button when mouse is near it
    const showZone = document.createElement('div');
    showZone.className = 'fullscreen-show-zone';
    showZone.style.cssText = `
      position: fixed;
      ${this.button.style.cssText
        .replace('opacity: 0', 'opacity: 0')
        .replace('z-index: 1000', 'z-index: 999')
        .replace('cursor: pointer', 'cursor: default')
        .replace(/backdrop-filter: [^;]+;?/, '')};
      width: 80px;
      height: 80px;
      background: transparent;
      border: none;
    `;

    this.button.parentNode.insertBefore(showZone, this.button);

    showZone.addEventListener('mouseenter', () => {
      this.show();
    });

    this.button.addEventListener('mouseleave', () => {
      if (!this.isFullscreen) {
        this.hide();
      }
    });
  }

  /**
   * Handle keyboard input
   */
  onKeyDown(event) {
    if (event.key === 'F11') {
      event.preventDefault();
      this.toggle();
    }
  }

  /**
   * Toggle fullscreen state
   */
  async toggle() {
    if (this.isFullscreen) {
      await this.exitFullscreen();
    } else {
      await this.requestFullscreen();
    }
  }

  /**
   * Request fullscreen
   */
  async requestFullscreen() {
    if (!this.isSupported) {
      this.showUnsupportedMessage();
      return false;
    }

    try {
      const element = document.documentElement;

      // Try each vendor prefix
      if (element.requestFullscreen) {
        await element.requestFullscreen();
      } else if (element.webkitRequestFullscreen) {
        await element.webkitRequestFullscreen();
      } else if (element.mozRequestFullScreen) {
        await element.mozRequestFullScreen();
      } else if (element.msRequestFullscreen) {
        await element.msRequestFullscreen();
      }

      // Save preference
      if (this.config.rememberPreference) {
        this.userPreference = true;
        this.savePreference();
      }

      return true;

    } catch (error) {
      this.logger.warn('Fullscreen request failed:', error);
      this.showError('Could not enter fullscreen mode');
      return false;
    }
  }

  /**
   * Exit fullscreen
   */
  async exitFullscreen() {
    if (!this.isFullscreen) return false;

    try {
      if (document.exitFullscreen) {
        await document.exitFullscreen();
      } else if (document.webkitExitFullscreen) {
        await document.webkitExitFullscreen();
      } else if (document.mozCancelFullScreen) {
        await document.mozCancelFullScreen();
      } else if (document.msExitFullscreen) {
        await document.msExitFullscreen();
      }

      // Save preference
      if (this.config.rememberPreference) {
        this.userPreference = false;
        this.savePreference();
      }

      return true;

    } catch (error) {
      this.logger.warn('Fullscreen exit failed:', error);
      return false;
    }
  }

  /**
   * Handle fullscreen state change
   */
  onFullscreenChange() {
    const wasFullscreen = this.isFullscreen;
    this.isFullscreen = this.checkIsFullscreen();

    this.updateState();

    // Emit event if state changed
    if (wasFullscreen !== this.isFullscreen) {
      this.gameManager.emit('fullscreen:changed', {
        isFullscreen: this.isFullscreen
      });
    }
  }

  /**
   * Check if currently in fullscreen
   */
  checkIsFullscreen() {
    return !!(
      document.fullscreenElement ||
      document.webkitFullscreenElement ||
      document.mozFullScreenElement ||
      document.msFullscreenElement
    );
  }

  /**
   * Handle fullscreen error
   */
  onFullscreenError(event) {
    this.logger.error('Fullscreen error:', event);

    this.showError('Fullscreen mode is not available. This may be due to browser restrictions or user permissions.');

    this.gameManager.emit('fullscreen:error', { event });
  }

  /**
   * Update button visual state
   */
  updateState() {
    const iconSpan = this.button.querySelector('.fullscreen-icon');

    if (this.isFullscreen) {
      iconSpan.innerHTML = this.getIconSVG('exit');
      this.button.setAttribute('aria-label', 'Exit fullscreen');
      this.button.setAttribute('title', 'Exit Fullscreen (F11)');
      this.button.style.background = 'rgba(0, 255, 136, 0.3)';
      this.button.classList.add('is-fullscreen');
    } else {
      iconSpan.innerHTML = this.getIconSVG('enter');
      this.button.setAttribute('aria-label', 'Enter fullscreen');
      this.button.setAttribute('title', 'Enter Fullscreen (F11)');
      this.button.style.background = 'rgba(0, 0, 0, 0.5)';
      this.button.classList.remove('is-fullscreen');
    }
  }

  /**
   * Show the button
   */
  show() {
    this.button.style.opacity = '1';
  }

  /**
   * Hide the button
   */
  hide() {
    if (!this.isFullscreen) {
      this.button.style.opacity = '0';
    }
  }

  /**
   * Show unsupported message
   */
  showUnsupportedMessage() {
    this.showError('Your browser does not support fullscreen mode. Please try a modern browser like Chrome, Firefox, or Edge.');
  }

  /**
   * Show error message
   */
  showError(message) {
    // Create toast notification
    const toast = document.createElement('div');
    toast.className = 'fullscreen-error-toast';
    toast.textContent = message;
    toast.style.cssText = `
      position: fixed;
      bottom: 80px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(255, 50, 50, 0.9);
      color: white;
      padding: 12px 24px;
      border-radius: 8px;
      z-index: 10000;
      animation: fadeInOut 3000ms forwards;
    `;

    // Add animation keyframes if not present
    if (!document.getElementById('fullscreen-toast-styles')) {
      const style = document.createElement('style');
      style.id = 'fullscreen-toast-styles';
      style.textContent = `
        @keyframes fadeInOut {
          0% { opacity: 0; transform: translateX(-50%) translateY(10px); }
          10% { opacity: 1; transform: translateX(-50%) translateY(0); }
          90% { opacity: 1; transform: translateX(-50%) translateY(0); }
          100% { opacity: 0; transform: translateX(-50%) translateY(-10px); }
        }
      `;
      document.head.appendChild(style);
    }

    document.body.appendChild(toast);

    // Remove after animation
    setTimeout(() => {
      toast.remove();
    }, 3000);
  }

  /**
   * Save user preference
   */
  savePreference() {
    try {
      localStorage.setItem('fullscreen_preference', JSON.stringify({
        enabled: this.userPreference,
        timestamp: Date.now()
      }));
    } catch (e) {
      this.logger.warn('Could not save fullscreen preference:', e);
    }
  }

  /**
   * Load user preference
   */
  loadPreference() {
    try {
      const saved = localStorage.getItem('fullscreen_preference');
      if (saved) {
        const data = JSON.parse(saved);
        this.userPreference = data.enabled || false;
      }
    } catch (e) {
      this.logger.warn('Could not load fullscreen preference:', e);
    }
  }

  /**
   * Check if device is touch
   */
  isTouchDevice() {
    return 'ontouchstart' in window ||
           navigator.maxTouchPoints > 0;
  }

  /**
   * Enable auto fullscreen (call from game start)
   */
  enableAutoFullscreen() {
    this.config.autoEnter = true;
    if (this.userPreference || this.isMobile()) {
      this.requestFullscreen();
    }
  }

  /**
   * Disable auto fullscreen
   */
  disableAutoFullscreen() {
    this.config.autoEnter = false;
  }

  /**
   * Check if currently in fullscreen
   */
  getIsFullscreen() {
    return this.isFullscreen;
  }

  /**
   * Clean up
   */
  destroy() {
    // Remove event listeners
    this.button.removeEventListener('click', this.toggle);

    // Remove element
    this.button.remove();
  }
}

/**
 * Factory for creating fullscreen buttons with common configurations
 */
class FullscreenButtonFactory {
  static createHUDButton(options = {}) {
    return new FullscreenButton({
      ...options,
      position: options.position || 'bottom-right',
      showInHUD: true,
      showInMenu: false
    });
  }

  static createMenuButton(options = {}) {
    return new FullscreenButton({
      ...options,
      position: 'static',  // Will be positioned by parent
      showInHUD: false,
      showInMenu: true
    });
  }

  static createInGameButton(options = {}) {
    return new FullscreenButton({
      ...options,
      position: options.position || 'top-right',
      showInHUD: true,
      showInMenu: true,
      autoEnter: true,
      rememberPreference: true
    });
  }
}

export default FullscreenButton;
export { FullscreenButtonFactory };
```

---

## 📝 How To Build A Fullscreen Button Like This

### Step 1: Create Simple Toggle

```javascript
// Simple fullscreen button
function createFullscreenButton() {
  const btn = document.createElement('button');
  btn.textContent = '⛶';
  btn.style.cssText = `
    position: fixed;
    bottom: 10px;
    right: 10px;
    z-index: 1000;
    padding: 10px;
  `;

  btn.addEventListener('click', () => {
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      document.documentElement.requestFullscreen();
    }
  });

  document.body.appendChild(btn);
}
```

### Step 2: Add State Tracking

```javascript
class FullscreenToggle {
  constructor() {
    this.isFullscreen = false;
    this.createButton();
    this.addListeners();
  }

  createButton() {
    this.button = document.createElement('button');
    this.updateButton();
    document.body.appendChild(this.button);
  }

  addListeners() {
    this.button.addEventListener('click', () => this.toggle());
    document.addEventListener('fullscreenchange', () => this.update());
  }

  toggle() {
    if (this.isFullscreen) {
      document.exitFullscreen();
    } else {
      document.documentElement.requestFullscreen();
    }
  }

  update() {
    this.isFullscreen = !!document.fullscreenElement;
    this.updateButton();
  }

  updateButton() {
    this.button.textContent = this.isFullscreen ? '⛶' : '⛷';
    this.button.title = this.isFullscreen ? 'Exit (F11)' : 'Fullscreen (F11)';
  }
}
```

---

## 🔧 Variations For Your Game

### Picture-in-Picture Toggle

```javascript
class PiPButton extends FullscreenButton {
  async requestPiP() {
    if (document.pictureInPictureElement) {
      await document.exitPictureInPicture();
    } else if (document.pictureInPictureEnabled) {
      // For canvas/video content
      await this.videoElement.requestPictureInPicture();
    }
  }
}
```

### Exclusive Fullscreen (Pointer Lock)

```javascript
class ExclusiveFullscreenButton extends FullscreenButton {
  async requestFullscreen() {
    // First go fullscreen
    await super.requestFullscreen();

    // Then lock pointer for true FPS-style control
    try {
      await document.documentElement.requestPointerLock({
        unadjustedMovement: false
      });
    } catch (e) {
      console.warn('Pointer lock not supported');
    }
  }

  async exitFullscreen() {
    await document.exitPointerLock();
    await super.exitFullscreen();
  }
}
```

### Windowed Mode Selector

```javascript
class WindowModeButton extends FullscreenButton {
  createElement() {
    super.createElement();

    // Add dropdown for different modes
    this.modes = ['fullscreen', 'borderless', 'windowed'];
    this.currentMode = 'fullscreen';

    const menu = document.createElement('select');
    menu.innerHTML = this.modes.map(mode =>
      `<option value="${mode}">${mode}</option>`
    ).join('');

    menu.addEventListener('change', (e) => {
      this.setMode(e.target.value);
    });

    this.button.appendChild(menu);
  }

  setMode(mode) {
    switch (mode) {
      case 'fullscreen':
        this.requestFullscreen();
        break;
      case 'borderless':
        this.setBorderless();
        break;
      case 'windowed':
        this.exitFullscreen();
        break;
    }
  }

  setBorderless() {
    // Resize window to screen size without fullscreen
    window.resizeTo(screen.width, screen.height);
    window.moveTo(0, 0);
  }
}
```

---

## Common Mistakes Beginners Make

### 1. Not Handling Browser Differences

```javascript
// ❌ WRONG: Only standard API
document.documentElement.requestFullscreen();
// Fails on some browsers

// ✅ CORRECT: Try all vendor prefixes
const elem = document.documentElement;
if (elem.requestFullscreen) {
  elem.requestFullscreen();
} else if (elem.webkitRequestFullscreen) {
  elem.webkitRequestFullscreen();
} else if (elem.mozRequestFullScreen) {
  elem.mozRequestFullScreen();
} else if (elem.msRequestFullscreen) {
  elem.msRequestFullscreen();
}
```

### 2. No Error Handling

```javascript
// ❌ WRONG: Assume it always works
async function toggleFullscreen() {
  await document.documentElement.requestFullscreen();
}
// Crashes if denied

// ✅ CORRECT: Handle errors gracefully
async function toggleFullscreen() {
  try {
    await document.documentElement.requestFullscreen();
  } catch (e) {
    if (e.name === 'NotAllowedError') {
      showMessage('Fullscreen permission denied');
    } else {
      console.error('Fullscreen failed:', e);
    }
  }
}
```

### 3. Not Tracking State

```javascript
// ❌ WRONG: No state tracking
button.onclick = () => {
  if (document.fullscreenElement) {
    document.exitFullscreen();
  } else {
    document.documentElement.requestFullscreen();
  }
};
// Doesn't handle ESC key exit

// ✅ CORRECT: Listen for changes
document.addEventListener('fullscreenchange', () => {
  updateButtonState();
});
```

### 4. Not Remembering Preference

```javascript
// ❌ WRONG: Always start windowed
startGame();

// ✅ CORRECT: Remember preference
function startGame() {
  loadPreferences();
  if (prefs.fullscreen) {
    requestFullscreen();
  }
}
```

---

## CSS for Fullscreen Button

```css
.fullscreen-button {
  position: fixed;
  bottom: 10px;
  right: 10px;
  width: 40px;
  height: 40px;
  padding: 8px;
  background: rgba(0, 0, 0, 0.6);
  border: 2px solid rgba(0, 255, 136, 0.3);
  border-radius: 8px;
  cursor: pointer;
  z-index: 1000;
  opacity: 0;
  transition: opacity 0.3s ease,
              background 0.3s ease,
              transform 0.2s ease,
              border-color 0.3s ease;
  backdrop-filter: blur(4px);
}

.fullscreen-button:hover {
  background: rgba(0, 255, 136, 0.2);
  border-color: rgba(0, 255, 136, 0.5);
  transform: scale(1.1);
}

.fullscreen-button:active {
  transform: scale(0.95);
}

.fullscreen-button.is-fullscreen {
  background: rgba(0, 255, 136, 0.3);
  border-color: rgba(0, 255, 136, 0.6);
  opacity: 1;
}

.fullscreen-icon {
  display: block;
  width: 100%;
  height: 100%;
}

.fullscreen-icon svg {
  display: block;
  width: 100%;
  height: 100%;
}

/* HUD container variant - always visible in HUD */
.hud-container .fullscreen-button {
  opacity: 0.6;
}

.hud-container:hover .fullscreen-button {
  opacity: 1;
}

/* Menu variant - always visible */
.options-menu .fullscreen-button {
  position: static;
  opacity: 1;
  width: auto;
  height: auto;
  padding: 10px 20px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.options-menu .fullscreen-button .fullscreen-icon {
  width: 20px;
  height: 20px;
}
```

---

## Performance Considerations

```
FULLSCREEN BUTTON PERFORMANCE:

DOM Elements:
├── Single button element
├── SVG icon (inline, no request)
└── Impact: Negligible

Event Listeners:
├── One click handler
├── Four fullscreen change listeners (vendor prefixed)
├── One keydown handler (optional)
└── Impact: Minimal

Fullscreen Transition:
├── Browser-controlled (GPU-accelerated)
├── Game canvas resize required
└── Impact: Frame dip during transition

Optimization:
├── Debounce rapid clicks
├── Use CSS transforms for animations
├── Avoid layout thrashing
└── Preload fullscreen assets if any
```

---

## Browser Compatibility

```
FULLSCREEN API SUPPORT:

Chrome/Edge:
├── Full support (standard API)
├── Keyboard shortcut: F11, F
└── Notes: Excellent support

Firefox:
├── Full support (moz-prefixed + standard)
├── Keyboard shortcut: F11
└── Notes: Good support

Safari:
├── Partial support (webkit-prefixed)
├── Requires user gesture
├── No keyboard shortcut support via API
└── Notes: Use F11 manually

Mobile:
├── Varies by device/browser
├── iOS Safari: Limited (must be user-initiated)
├── Android Chrome: Good support
└── Notes: Test on target devices
```

---

## Related Systems

- [UIManager](./ui-manager.md) - UI component management
- [Options Menu](./options-menu.md) - Settings integration
- [Touch Joystick](./touch-joystick.md) - Mobile controls

---

## Source File Reference

**Primary Files**:
- `../src/ui/FullscreenButton.js` - Fullscreen toggle component (estimated)

**Key Classes**:
- `FullscreenButton` - Main fullscreen toggle
- `FullscreenButtonFactory` - Pre-configured button variants

**Dependencies**:
- Fullscreen API (browser API)
- localStorage (preference persistence)
- GameManager (events)

---

## References

- [Fullscreen API MDN](https://developer.mozilla.org/en-US/docs/Web/API/Fullscreen_API) - Official API docs
- [Element.requestFullscreen()](https://developer.mozilla.org/en-US/docs/Web/API/Element/requestFullscreen) - Method reference
- [Fullscreen polyfill](https://github.com/sindresorhus/screenfull.js) - Fallback for older browsers

*Documentation last updated: January 12, 2026*
