# VideoManager - First Principles Guide

## Overview

The **VideoManager** handles video playback with alpha channel (transparency) support in the Shadow Engine. It allows videos to be rendered in the 3D world as textured planes or as UI overlays, enabling cinematic storytelling elements.

Think of VideoManager as the **cinematics system** for your game - it plays pre-rendered video clips that can have transparent backgrounds, perfect for ghostly apparitions, holographic displays, or cutscene content.

## What You Need to Know First

Before understanding VideoManager, you should know:
- **Video file formats** - What codecs work in browsers
- **Alpha channels** - Transparency in video
- **Textures and materials** - How videos are mapped to 3D objects
- **Three.js scene graphs** - Adding objects to 3D scenes
- **HTML5 Video API** - Browser video playback

### Quick Refresher: Alpha Channel Video

```
Regular video:        Alpha channel video:
┌──────────────┐      ┌──────────────┐
│█████████████│      │░░░░░░░░░░░░░│
│█████████████│      │░░   ▓   ░░░░│  ▓ = actual content
│█████████████│      │░░         ░░░│  ░ = transparent
│█████████████│      │░░░░░░░░░░░░░│
└──────────────┘      └──────────────┘
Solid background     Transparent background
```

**WebM with VP9 + Alpha** is the go-to format for web games.

---

## Part 1: Why Use Video in Games?

### The Problem: Animations Are Hard

Without video, you'd need to animate complex scenes:

```javascript
// ❌ WITHOUT video - Everything must be real-time 3D
function showGhostApparition() {
  // Load character model
  const model = await loadGLTF("ghost.glb");
  scene.add(model);

  // Animate manually (complex!)
  const mixer = new THREE.AnimationMixer(model);
  mixer.clipAction("float").play();

  // Add particle effects
  // Add glow shaders
  // Add lighting changes
  // ... hundreds of lines of code
}
```

### The Solution: Pre-Rendered Video

```javascript
// ✅ WITH video - Just play a clip
videoManager.playVideo({
  file: "videos/ghost-appearance.webm",
  position: { x: 0, y: 1, z: -3 },
  scale: { x: 1.78, y: 1 },  // 16:9 aspect ratio
  hasAlpha: true,
  loop: true
});

// Video was rendered once with high-quality effects
// Now plays back as a simple video clip!
```

**Benefits:**
- Consistent quality across devices
- Complex effects without GPU overhead
- Precise timing and storytelling
- Can include effects hard to do in real-time

---

## Part 2: Video Data Structure

### Basic Video Definition

```javascript
// In videoData.js
export const videos = {
  endingScene: {
    // Video file path (WebM with alpha recommended)
    file: "video/edison-ending.webm",

    // 3D position in world
    position: { x: 0, y: 1.5, z: -2 },

    // Scale (multiply by video aspect ratio)
    scale: { x: 1.78, y: 1 },

    // Whether video has transparency
    hasAlpha: true,

    // Loop or one-shot
    loop: false,

    // Optional: Volume (for video with audio)
    volume: 1.0,

    // When this video plays
    criteria: { dialogChoice2: DIALOG_RESPONSE_TYPES.EDISON }
  },

  hologramDisplay: {
    file: "video/hologram-loop.webm",
    position: { x: 2, y: 1, z: 0 },
    scale: { x: 1, y: 1 },
    hasAlpha: true,
    loop: true,
    screenSpace: false,  // 3D world position
    criteria: { currentState: HOLOGRAM_INTRO }
  }
};
```

### Video Properties Reference

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `file` | string | Yes | Path to WebM video file |
| `position` | object | Yes | { x, y, z } world position |
| `scale` | object | Yes | { x, y } scale multiplier |
| `rotation` | object | No | { x, y, z } Euler rotation (degrees) |
| `hasAlpha` | boolean | Yes | Whether video has alpha channel |
| `loop` | boolean | No | Whether to loop the video |
| `volume` | number | No | Audio volume (0-1) |
| `playbackRate` | number | No | Playback speed (1 = normal) |
| `screenSpace` | boolean | No | If true, renders as UI overlay |
| `criteria` | object | Yes | State criteria for playback |
| `fadeIn` | number | No | Fade-in duration (ms) |
| `fadeOut` | number | No | Fade-out duration (ms) |

---

## Part 3: How VideoManager Works

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VideoManager                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Active Videos                              │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                       │   │
│  │  │ Video 1 │  │ Video 2 │  │ Video 3 │  ... (max 4-6)      │   │
│  │  │ Playing │  │ Playing │  │ Loading │                       │   │
│  │  └─────────┘  └─────────┘  └─────────┘                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Video Element Pool (DOM)                          │   │
│  │  - Reuses HTMLVideoElement instances                          │   │
│  │  - Each video needs its own element                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                Three.js Integration                            │   │
│  │  - VideoTexture for each video                               │   │
│  │  - PlaneGeometry for rendering                               │   │
│  │  - MeshBasicMaterial (transparent for alpha)                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
         │                            │
         │ listens to                 │ renders to
         ▼                            ▼
┌─────────────────┐          ┌─────────────────┐
│  GameManager    │          │   Three.js      │
│  "state:changed"│          │   Scene/Camera   │
└─────────────────┘          └─────────────────┘
```

### Playback Flow

```
1. State changes (GameManager emits "state:changed")
              │
              ▼
2. VideoManager receives event
              │
              ▼
3. Find videos matching new state
              │
              ▼
4. For each matching video:
    │
    ├──▶ Create/reuse HTMLVideoElement
    │
    ├──▶ Create Three.js VideoTexture from element
    │
    ├──▶ Create plane mesh with texture
    │
    ├──▶ Position mesh in 3D world (or add to UI layer)
    │
    ├──▶ Start video playback
    │
    └──▶ Update texture each frame
```

---

## Part 4: Creating Videos with Alpha

### Export Settings

For video with transparency:

| Setting | Value | Notes |
|---------|-------|-------|
| **Codec** | VP9 + Alpha | Best browser support |
| **Format** | WebM | Required for VP9 |
| **Resolution** | 1920x1080 or lower | Match your game's resolution |
| **Frame Rate** | 30 fps | Sufficient for most content |
| **Bitrate** | 5-10 Mbps | Balance quality and size |
| **Color** | RGBA | Alpha channel enabled |

### Tools for Creating Alpha Videos

- **Blender** - Free, open-source 3D software
- **After Effects** - Professional compositing
- **DaVinci Resolve** - Free (with Studio version for alpha export)
- **FFmpeg** - Command-line tool

### FFmpeg Command Example

```bash
# Export with alpha channel
ffmpeg -i input.mp4 \
  -c:v libvpx-vp9 \
  -b:v 5M \
  -c:a libopus \
  -auto-alt-ref 0 \
  output.webm

# With specific resolution and frame rate
ffmpeg -i input.mp4 \
  -vf scale=1920:1080 \
  -r 30 \
  -c:v libvpx-vp9 \
  -b:v 8M \
  output.webm
```

---

## Part 5: World-Space vs Screen-Space Video

### World-Space Video (3D)

Renders video as a 3D object in the scene:

```javascript
export const videos = {
  hologramGhost: {
    file: "video/ghost.webm",
    hasAlpha: true,
    screenSpace: false,  // ← Renders in 3D world

    position: { x: 0, y: 1, z: -3 },
    scale: { x: 1.78, y: 1 },

    // Player can walk around it
    criteria: { currentState: GHOST_APPEARS }
  }
};
```

**Use cases:**
- Holographic displays
- Ghostly apparitions
- In-world screens/monitors
- Projected imagery

**Visual representation:**
```
        Camera view
            ↓

    ┌─────────────┐
    │             │
    │  👤         │  ← Player can walk
    │             │     around the video
    │     [Video]  │     in 3D space
    │             │
    └─────────────┘
```

### Screen-Space Video (UI Overlay)

Renders video as a 2D UI element:

```javascript
export const videos = {
  vignetteEffect: {
    file: "video/vignette.webm",
    hasAlpha: true,
    screenSpace: true,  // ← Renders as UI

    // Optional: Position on screen
    position: "center",  // or "top", "bottom-left", etc.

    scale: 1.0,
    loop: true,

    criteria: { viewmasterInsanityIntensity: { $gt: 0.5 } }
  }
};
```

**Use cases:**
- Vignette effects
- Full-screen transitions
- UI video elements
 Picture-in-picture content

**Visual representation:**
```
┌─────────────────────────────┐
│                             │
│   [Video overlays screen]  │  ← Always on top
│                             │
│        👤                   │  ← Player movement doesn't
│                             │     affect video position
└─────────────────────────────┘
```

---

## Part 6: Three.js Video Texture Implementation

### Creating Video Textures

```javascript
class VideoManager {
  createVideoMesh(videoData) {
    // Create HTML video element
    const videoElement = document.createElement('video');
    videoElement.src = videoData.file;
    videoElement.loop = videoData.loop || false;
    videoElement.muted = !videoData.volume;  // Muted if no volume
    videoElement.playsInline = true;
    videoElement.crossOrigin = 'anonymous';

    // For alpha channel videos
    if (videoData.hasAlpha) {
      videoElement.playsInline = true;
    }

    // Create video texture
    const texture = new THREE.VideoTexture(videoElement);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;

    // For alpha videos, need proper blending
    const material = new THREE.MeshBasicMaterial({
      map: texture,
      transparent: videoData.hasAlpha,
      side: THREE.DoubleSide,
      depthWrite: false,  // Don't write to depth buffer (for transparency)
      blending: videoData.hasAlpha
        ? THREE.CustomBlending
        : THREE.NormalBlending
    });

    // Create plane geometry (aspect ratio 16:9 by default)
    const geometry = new THREE.PlaneGeometry(
      videoData.scale.x * 1.78,  // Width
      videoData.scale.y,         // Height
    );

    // Create mesh
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(
      videoData.position.x,
      videoData.position.y,
      videoData.position.z
    );

    // Optional rotation
    if (videoData.rotation) {
      mesh.rotation.set(
        THREE.MathUtils.degToRad(videoData.rotation.x),
        THREE.MathUtils.degToRad(videoData.rotation.y),
        THREE.MathUtils.degToRad(videoData.rotation.z)
      );
    }

    return { mesh, videoElement, texture };
  }
}
```

### Updating Video Textures

```javascript
class VideoManager {
  update(deltaTime) {
    // Update all active video textures
    this.activeVideos.forEach(({ texture }) => {
      if (texture.image.readyState >= texture.image.HAVE_CURRENT_DATA) {
        texture.needsUpdate = true;
      }
    });
  }
}
```

---

## Part 7: Video Performance Optimization

### Limit Concurrent Videos

Too many videos playing = poor performance:

```javascript
class VideoManager {
  constructor() {
    this.activeVideos = [];
    this.maxConcurrent = 4;  // Adjust based on testing
  }

  playVideo(videoData) {
    // Check if we can play more videos
    if (this.activeVideos.length >= this.maxConcurrent) {
      // Stop the least recently updated video
      const toStop = this.activeVideos.shift();
      this.stopVideo(toStop);
    }

    // Play new video
    const video = this.createAndPlay(videoData);
    this.activeVideos.push(video);
  }
}
```

### Resolution Strategies

```javascript
// Different resolutions for different situations

export const videos = {
  // High-quality for important scenes
  criticalScene: {
    file: "video/critical-1080p.webm",
    criteria: { currentState: FINAL_BATTLE }
  },

  // Lower quality for ambient/looping content
  ambientLoop: {
    file: "video/ambient-720p.webm",  // Lower resolution
    loop: true,
    criteria: { currentZone: "office" }
  },

  // Very low quality for background elements
  backgroundElement: {
    file: "video/bg-480p.webm",  // Lowest acceptable
    loop: true,
    criteria: { currentState: { $gte: EXPLORATION_MODE } }
  }
};
```

### Lazy Loading

```javascript
class VideoManager {
  playVideo(videoData) {
    // Create placeholder mesh first
    const placeholder = this.createPlaceholder(videoData.position);

    // Load video asynchronously
    this.loadVideoWhenVisible(videoData, placeholder);
  }

  loadVideoWhenVisible(videoData, placeholder) {
    const observer = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting) {
        // Video is visible, load it now
        this.createAndPlay(videoData);
        observer.disconnect();
      }
    });

    observer.observe(placeholder);
  }
}
```

---

## Part 8: Common Video Use Cases

### 1. Ghostly Apparition

```javascript
ghostAppearance: {
  file: "video/ghost-fade-in.webm",
  position: { x: 0, y: 1.5, z: -4 },
  scale: { x: 1, y: 1 },
  hasAlpha: true,
  loop: true,
  playbackRate: 0.8,  // Slightly slower for eerie effect
  criteria: { currentState: GHOST_ENCOUNTER }
}
```

### 2. Holographic Display

```javascript
hologramTutorial: {
  file: "video/tutorial-hologram.webm",
  position: { x: -2, y: 1.2, z: 0 },
  rotation: { y: 180 },  // Face player
  scale: { x: 0.8, y: 0.8 },
  hasAlpha: true,
  loop: true,
  criteria: { playerHasItem: "viewmaster" }
}
```

### 3. Screen Overlay Effect

```javascript
vignetteHorror: {
  file: "video/vignette-pulse.webm",
  screenSpace: true,
  position: "center",
  scale: 1.0,
  hasAlpha: true,
  loop: true,
  criteria: { sanityLevel: { $lt: 0.3 } }
}
```

### 4. Ending Sequence

```javascript
endingCredits: {
  file: "video/credits-roll.webm",
  screenSpace: true,
  position: "center",
  scale: 1.0,
  hasAlpha: true,
  loop: false,
  criteria: { currentState: ENDING_SEQUENCE }
}
```

---

## Common Mistakes Beginners Make

### 1. Wrong Video Format

```javascript
// ❌ WRONG: MP4 doesn't support alpha in most browsers
file: "video/ghost.mp4"

// ✅ CORRECT: Use WebM with VP9 + Alpha
file: "video/ghost.webm"
```

### 2. Not Setting Transparent Material

```javascript
// ❌ WRONG: Video has black background
const material = new THREE.MeshBasicMaterial({
  map: texture
});

// ✅ CORRECT: Enable transparency
const material = new THREE.MeshBasicMaterial({
  map: texture,
  transparent: true,
  side: THREE.DoubleSide,
  depthWrite: false
});
```

### 3. Forgetting crossOrigin

```javascript
// ❌ WRONG: Tainted canvas issue
videoElement.src = "video/ghost.webm";

// ✅ CORRECT: Set crossOrigin
videoElement.crossOrigin = 'anonymous';
videoElement.src = "video/ghost.webm";
```

### 4. Wrong Aspect Ratio

```javascript
// ❌ WRONG: Stretched video
const geometry = new THREE.PlaneGeometry(1, 1);  // Square!

// ✅ CORRECT: Match video aspect ratio (usually 16:9)
const geometry = new THREE.PlaneGeometry(1.78, 1);  // 16:9
```

### 5. Playing Too Many Videos

```javascript
// ❌ WRONG: All zones play videos
export const videos = {
  zone1: { file: "v1.webm", criteria: { currentZone: "zone1" } },
  zone2: { file: "v2.webm", criteria: { currentZone: "zone2" } },
  zone3: { file: "v3.webm", criteria: { currentZone: "zone3" } },
  // ... 50 more!
}

// ✅ CORRECT: Limit concurrent videos, prioritize visible areas
```

---

## Performance Considerations

### Bandwidth vs Quality

| Resolution | File Size (10 sec) | When to Use |
|------------|-------------------|--------------|
| 480p | ~2 MB | Background elements, far objects |
| 720p | ~5 MB | Standard gameplay content |
| 1080p | ~10 MB | Important story moments |
| 4K | ~40 MB | Not recommended for web |

### Memory Management

```javascript
// Clean up videos when done
class VideoManager {
  stopVideo(videoInstance) {
    // Stop playback
    videoInstance.videoElement.pause();
    videoInstance.videoElement.src = "";  // Unload

    // Dispose texture
    videoInstance.texture.dispose();

    // Remove from scene
    this.scene.remove(videoInstance.mesh);
    videoInstance.mesh.geometry.dispose();
    videoInstance.mesh.material.dispose();
  }
}
```

---

## 🎮 Game Design Perspective

### Creative Intent

Video in games serves several purposes:

1. **Cinematic storytelling** - Pre-rendered dramatic moments
2. **Atmosphere enhancement** - Visual effects without GPU cost
3. **UI communication** - Tutorial overlays, hints
4. **World-building** - In-game screens, holograms

### Design Principles

```javascript
// Principle 1: Use sparingly (special moments)
specialMoment: {
  file: "video/reveal.webm",
  criteria: { currentState: MAJOR_REVEAL }
  // Only for IMPORTANT moments
}

// Principle 2: Loop should be seamless
ambientLoop: {
  file: "video/ambient-loop.webm",
  loop: true,
  // Ensure video loops perfectly without jarring cuts
}

// Principle 3: Respect player attention
skipableCutscene: {
  file: "video/cutscene.webm",
  allowSkip: true,  // Let players skip if they've seen it
  criteria: { currentState: CUTSCENE_START }
}
```

---

## Next Steps

Now that you understand VideoManager:

- [DialogManager](./dialog-manager.md) - Audio dialog system
- [MusicManager](./music-manager.md) - Background music system
- [VFXManager](../07-visual-effects/vfx-manager.md) - Visual effects
- [SceneManager](../03-scene-rendering/scene-manager.md) - 3D content management

---

## References

- [WebM VP9 Codec](https://www.webmproject.org/docs/codec-requirements.html) - WebM format specs
- [Three.js VideoTexture](https://threejs.org/docs/#api/en/textures/VideoTexture) - Official docs
- [HTML5 Video API](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/video) - MDN reference
- [FFmpeg VP9 Encoding](https://trac.ffmpeg.org/wiki/Encode/VP9) - Encoding guide
- [Video for Web Guidelines](https://web.dev/fast/video/) - Optimization tips

*Documentation last updated: January 12, 2026*
