# Vimana

A new game built on the Shadow Engine architecture.

## 🎮 Game Flow

1. **Intro Video** - Plays on startup
2. **Music Room** - Player spawns after video ends

## 📁 Project Structure

```
vimana/
├── index.html                  # Entry HTML
├── package.json                # Dependencies
├── vite.config.js              # Vite configuration
├── README.md                   # This file
└── src/
    ├── main.js                 # Main game entry point
    ├── gameData.js             # Game states (LOADING, VIDEO_INTRO, MUSIC_ROOM)
    ├── sceneData.js            # Scene/zone definitions
    ├── videoData.js            # Intro video definition
    ├── colliderData.js         # Physics colliders
    ├── lightData.js            # Light definitions
    └── styles/
        └── loadingScreen.css   # Loading screen styles
```

## 🚀 Getting Started

### 1. Install Dependencies

```bash
cd vimana
npm install
```

### 2. Add Your Assets

Place your files in `public/assets/`:

```
public/assets/
├── models/
│   └── music_room.glb          # Your 3D model
└── videos/
    └── intro.webm              # Your intro video with alpha
```

### 3. Run the Game

```bash
npm run dev
```

Open `https://localhost:5173` (HTTPS is required for some features)

## 📝 Configuration

### Adjusting Spawn Position

Edit `src/sceneData.js`:

```javascript
spawn: {
  position: { x: 0, y: 0.9, z: 2 },   // Where player appears
  rotation: { x: 0, y: 180, z: 0 },    // Direction facing (degrees)
}
```

### Adding Floor Colliders

Edit `src/colliderData.js`:

```javascript
{
  id: 'floor',
  type: 'box',
  size: { x: 20, y: 0.1, z: 20 },  // Adjust to room size
}
```

### Adjusting Lights

Edit `src/lightData.js`:

```javascript
{
  id: 'ambient',
  type: 'ambient',
  intensity: 0.4,
  color: 0x404060,  // Hex color
},
{
  id: 'sun',
  type: 'directional',
  intensity: 1.0,
  color: 0xffeedd,
}
```

## 🎬 Video Format

The intro video should be:
- **Format**: WebM (VP8/VP9 codec)
- **Alpha Channel**: Use yuva420p pixel format
- **Resolution**: 1920x1080 recommended

### Creating WebM with Alpha (FFmpeg)

```bash
ffmpeg -i input.mov -c:v libvpx-vp9 \
  -pix_fmt yuva420p -auto-alt-ref 1 \
  intro.webm
```

## 🎮 Controls

- **WASD** - Move
- **Mouse** - Look around
- **Click** - Lock pointer
- **ESC** - Release pointer

## 🔧 Debugging

Open browser console to see logs prefixed with `🎮 Vimana:`

```javascript
// Access managers from console
window.vimanaGame.gameManager
window.vimanaGame.sceneManager
window.vimanaGame.physicsManager

// Change state manually
window.gameManager.setState({ currentState: 1 }) // MUSIC_ROOM
```

## 📦 Adding New Scenes

1. Add scene to `src/sceneData.js`
2. Add state to `src/gameData.js`
3. Add colliders to `src/colliderData.js` (if needed)
4. Add lights to `src/lightData.js` (if needed)

## 🆚 Engine Architecture (from parent project)

This game uses the Shadow Engine's managers:
- **GameManager** - Central state store
- **SceneManager** - GLB/GLTF loading
- **PhysicsManager** - Rapier physics world
- **InputManager** - Keyboard/mouse/gamepad
- **VideoManager** - WebM video playback
- **MusicManager** - Background music
- **SFXManager** - Sound effects
- **LightManager** - Dynamic lighting
- **VFXManager** - Post-processing effects
