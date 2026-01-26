# Tech Stack Overview - Shadow Engine

## Overview

The Shadow Engine is built on a modern JavaScript stack designed specifically for creating immersive 3D web experiences using **Gaussian Splatting** - a cutting-edge rendering technique that allows photorealistic 3D scenes to run in a web browser.

This document explains each technology in the stack, why it was chosen, and how they work together to create the engine.

## What You Need to Know First

Before diving into this tech stack, you should understand:
- **Basic JavaScript knowledge** - The engine is written in JavaScript
- **What a "library" or "framework" is** - Reusable code that helps you build things faster
- **3D coordinates (X, Y, Z)** - How 3D space is represented
- **Web browsers** - Chrome, Firefox, Safari, Edge (your app runs in these)

If you're new to 3D web development, don't worry! This guide explains everything from first principles.

---

## Core Technologies

### 1. Three.js

**What Is It?**
Three.js is a JavaScript library that makes it easy to create 3D graphics in a web browser. Without Three.js, you'd need to write complex WebGL code manually. Three.js handles all the hard stuff for you.

**Why It's Used Here**
Three.js is the industry standard for 3D on the web. It provides:
- A simple API for creating 3D scenes
- Built-in geometries, materials, and lighting
- Camera controls
- loaders for 3D models
- Cross-browser compatibility

**Current Version:** v0.180.0+ (as of 2025)

**Key Concepts (Simplified)**

```
Think of Three.js like a puppet show:

Scene    = The stage where everything happens
Camera   = Where the audience (user) is sitting
Renderer = The person drawing the scene frame by frame
Mesh     = A puppet (3D object)
Material = The puppet's costume/colors
Light    = Stage lighting
```

**Basic Three.js Setup**
```javascript
// These are the core Three.js components
import * as THREE from 'three';

// 1. Create the scene (your 3D world)
const scene = new THREE.Scene();

// 2. Create the camera (your view into the world)
const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);

// 3. Create the renderer (draws everything to screen)
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
```

**In the Shadow Engine:**
Three.js provides the foundation - the 3D scene, cameras, and rendering pipeline. All other technologies build on top of Three.js.

---

### 2. SparkJS.dev (@sparkjsdev/spark)

**What Is It?**
SparkJS is a **Gaussian Splatting renderer** that integrates with Three.js. But what is Gaussian Splatting?

**Gaussian Splatting Explained (Simply)**

Traditional 3D graphics use **polygons** (triangles) to build 3D objects. Think of it like building with LEGO bricks - you need lots of small pieces to make something look smooth.

**Gaussian Splatting is different** - instead of triangles, it uses millions of tiny "splats" (like fuzzy dots) that can represent any 3D scene incredibly realistically.

```
Traditional 3D vs Gaussian Splatting:

Traditional:           Gaussian Splatting:
   ▲                         .
 ▲ ▲ ▲                      . . .
▲ ▲ ▲ ▲ ▲    vs          . . . . .
```

**Why It's Revolutionary**
- **Photorealism** - Can capture real-world scenes perfectly
- **Performance** - Complex scenes render faster than polygon-based methods
- **Web-native** - Runs directly in browsers

**How SparkJS Works**

```javascript
import { Spark } from '@sparkjsdev/spark';

// Spark integrates with your Three.js scene
const spark = new Spark();
await spark.init('./path/to/scene.sog');

// The .sog file contains millions of splat points
// Spark handles the complex sorting and rendering
```

**The .SOG File Format**
SOG (Spark Splat Object) files contain:
- **Millions of 3D points** - Each with position, color, and opacity
- **Gaussian parameters** - How each point "splats" onto the screen
- **Scene metadata** - Information about the environment

**In the Shadow Engine:**
SparkJS is the core rendering engine. All the environments (club, green room, office) are stored as .sog files and rendered through Spark.

---

### 3. Rapier Physics (@dimforge/rapier3d)

**What Is It?**
Rapier is a **physics engine** - it calculates how objects move, collide, and interact in a realistic way.

**What Does a Physics Engine Do?**

Without a physics engine, you'd have to manually calculate:
- Gravity pulling objects down
- Objects bouncing off each other
- Friction slowing things down
- Character movement and collisions

**Rapier handles all of this automatically.**

**Why Rapier?**
- **Written in Rust** - Fast and reliable
- **WebAssembly (WASM)** - Runs at near-native speed in browsers
- **Cross-platform** - Works everywhere JavaScript does
- **Modern** - Active development and improvements

**Current Version:** v0.19.0+

**Basic Physics Concepts**

```
Think of Rapier like a physics simulation:

RigidBody  = An object that can move (like a ball)
Collider   = The shape that defines the object's boundaries
Joint      = A connection between objects (like a hinge)
Force      = A push or pull on an object
```

**In the Shadow Engine:**
- **CharacterController** uses Rapier for player movement
- **Collision detection** for trigger zones
- **Physics interactions** with objects

---

### 4. Howler.js (howler)

**What Is It?**
Howler.js is an audio library for the web. It solves the many problems with browser audio.

**The Problem with Web Audio**
Different browsers handle audio differently. The native Web Audio API is powerful but complex to use. Howler.js simplifies this.

**What Howler Provides**
- **Cross-browser audio** - Works the same everywhere
- **Audio formats** - Handles MP3, OGG, WAV, WebM
- **Spatial audio** - 3D positional sound
- **Fade effects** - Smooth volume transitions
- **Sprite support** - Multiple sounds from one file

**Current Version:** v2.2.4+

**Basic Usage**
```javascript
import { Howl } from 'howler';

const sound = new Howl({
  src: ['sound.mp3'],
  volume: 0.5,
  loop: true
});

sound.play();
```

**In the Shadow Engine:**
- **DialogManager** - Spoken dialogue
- **MusicManager** - Background music with crossfades
- **SFXManager** - Sound effects

---

### 5. Vite

**What Is It?**
Vite is a **build tool** - it prepares your code for running in a browser.

**What Does a Build Tool Do?**

Modern JavaScript uses features that browsers don't natively understand. A build tool:
1. **Bundles** your code - Combines many files into one
2. **Transforms** code - Converts modern JS to browser-compatible JS
3. **Serves** during development - Hot reload when you save
4. **Optimizes** for production - Minifies and compresses

**Why Vite?**
- **Fast** - Uses native ES modules during development
- **Simple configuration** - Works out of the box
- **WASM support** - Required for Rapier physics
- **Plugin ecosystem** - Extensible

**Current Version:** v7.1.7+

---

## How These Technologies Work Together

Here's the data flow in the Shadow Engine:

```
┌─────────────────────────────────────────────────────────────┐
│                      Your Game Code                         │
│  (GameManager, CharacterController, DialogManager, etc.)     │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Three.js    │   │   SparkJS     │   │    Rapier     │
│   (3D Scene)  │   │  (Splat       │   │   (Physics)   │
│               │   │   Rendering)  │   │               │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                   ┌─────────────────┐
                   │   WebGL/WebGPU   │
                   │   (Browser GPU)  │
                   └─────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   User Screen   │
                   │   (Visual +      │
                   │    Audio)        │
                   └─────────────────┘
```

---

## Technology Versions Summary

| Technology | Version | Purpose |
|------------|---------|---------|
| **Three.js** | ^0.180.0 | 3D graphics framework |
| **@sparkjsdev/spark** | ^0.1.10 | Gaussian splat renderer |
| **@dimforge/rapier3d** | ^0.19.0 | Physics engine (WASM) |
| **howler** | ^2.2.4 | Audio playback |
| **Vite** | ^7.1.7 | Build tool & dev server |

---

## Common Mistakes Beginners Make

### 1. Forgetting to Initialize Async

Many of these libraries need to be initialized **asynchronously**:

```javascript
// ❌ WRONG - won't work
const spark = new Spark();
spark.load('./scene.sog');

// ✅ CORRECT - use await
const spark = new Spark();
await spark.load('./scene.sog');
```

### 2. Not Handling WebGL Context Loss

WebGL can crash or lose context. The Shadow Engine handles this with proper error handling.

### 3. Assuming Everything Works Everywhere

Always test on multiple browsers! The Shadow Engine includes platform detection for this reason.

### 4. Ignoring Performance

Gaussian Splatting is GPU-intensive. The Shadow Engine uses performance profiles to handle different devices.

---

## Next Steps

Now that you understand the tech stack, explore:

- [Game State System](../02-core-architecture/game-state-system.md) - How the engine manages game state
- [SceneManager Deep Dive](../03-scene-rendering/scene-manager.md) - How 3D content is loaded
- [CharacterController](../04-input-physics/character-controller.md) - How player movement works

---

## References

- [Three.js Documentation](https://threejs.org/docs/) - Official docs
- [Three.js Manual](https://threejs.org/manual/) - Tutorials and guides
- [SparkJS.dev Documentation](https://sparkjs.dev/docs/) - Gaussian splatting docs
- [SparkJS Overview](https://sparkjs.dev/docs/overview/) - Understanding Gaussian splatting
- [SparkRenderer Guide](https://sparkjs.dev/docs/spark-renderer/) - Rendering system
- [Rapier Physics Documentation](https://rapier.rs/docs/) - Official Rapier docs
- [Rapier JavaScript Guide](https://rapier.rs/docs/user_guides/javascript/getting_started_js/) - Getting started with JS
- [Howler.js Documentation](https://howlerjs.com/) - Audio library docs
- [Vite Documentation](https://vitejs.dev/) - Build tool docs

*Documentation last updated: January 2026*
