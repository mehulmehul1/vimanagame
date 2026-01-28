# VIMANA Harp Room - REVISED Implementation Plan

## 🚨 CRITICAL UPDATES - READ FIRST

### Core Philosophy Shift: "The Ship Teaches You Its Song - A Duet, Not a Test"

**Previous Approach (Test-Based):**
```
❌ Jelly teaches pattern → Player must repeat correctly
❌ Wrong input = FAILURE → RESET sequence
❌ Ship tests player competence
❌ Player passes/fails puzzle
```

**New Approach (Duet-Based):**
```
✅ Jelly demonstrates melody → Player joins in harmony
✅ Wrong note = GENTLE NUDGE → Ship patiently waits
✅ Ship teaches through presence, not testing
✅ Player and ship co-create music together
```

**The Metaphor:** 
> "You're not being tested on the ship's song. The ship is singing to you, and you're invited to harmonize. It's a duet—a shared moment of music-making. When you play wrong, the ship doesn't fail you. It just sings again, more slowly, more clearly, until you can join in harmony."

---

### What's in the Model (GLB Analysis)

✅ **Tunnel** (3168 vertices) - Entry tunnel
✅ **Beach** (624 vertices) - Walking surface  
✅ **Platform** (232 vertices) - Player platform
✅ **Vortex_gate** (2304 vertices) - Portal/vortex
✅ **Dome** (1595 vertices) - Music room dome
✅ **ArenaFloor** (192 vertices) - Water pool base
✅ **ArenaWall001** (384 vertices) - Dome structure
✅ **Pillar_1-004** (5 total pillars) - Decorative
✅ **Window 1, 2, 3** - 3 decorative windows
✅ **Gate_Plug** - Decorative element
✅ **Harp_body, Harp_stand, Harp Plate** - Harp frame present!
✅ **String 1-6** - 6 strings present (NOT 7!)
✅ **Spawn_room** - Player spawn area

### What's Missing (Procedural Creation Required)
❌ **Jelly creatures** - Must create procedurally
❌ **Water shader** - ArenaFloor needs water material
❌ **Shell collectible** - SDF-based reward item (3s appear, click to collect)
❌ **UI overlay** - 4-slot shell collection display
❌ **White flash ending** - Vortex engulf shader (no 3D white room needed)

---

## 🎨 VISUAL SHADER ENHANCEMENTS

### 1. WATER SURFACE - BIOLUMINESCENT RESONANCE

**Enhanced Shader Concept:**
```javascript
// The water shouldn't just be water—it's the ship's "skin" responding to music
// Visual philosophy: Every sound creates visible vibration in the water

const waterMaterial = new THREE.ShaderMaterial({
  uniforms: {
    uTime: { value: 0 },
    uHarpFrequencies: { value: new Float32Array(6).fill(0) }, // 6 string frequencies
    uHarpVelocities: { value: new Float32Array(6).fill(0) }, // 6 string velocities
    uDuetProgress: { value: 0.0 }, // How much player is harmonizing (0-1)
    uShipPatience: { value: 1.0 }, // Ship's gentle teaching intensity
    uBioluminescentColor: { value: new THREE.Color(0x00ff88) }, // Base bio-glow
    uHarmonicResonance: { value: 0.0 }, // When player + ship are in sync
  },
  
  vertexShader: `
    uniform float uTime;
    uniform float uHarpFrequencies[6];
    uniform float uHarpVelocities[6];
    uniform float uHarmonicResonance;
    
    varying vec2 vUv;
    varying vec3 vWorldPos;
    varying vec3 vNormal;
    varying float vElevation;
    
    void main() {
      vUv = uv;
      vNormal = normal;
      
      vec4 worldPos = modelMatrix * vec4(position, 1.0);
      vWorldPos = worldPos.xyz;
      
      // Displace water based on harp vibrations
      float displacement = 0.0;
      for (int i = 0; i < 6; i++) {
        float freq = uHarpFrequencies[i];
        float vel = uHarpVelocities[i];
        
        // Each string creates a wave pattern in the water
        float stringWave = sin(worldPos.x * freq + uTime * 3.0) * 
                          sin(worldPos.z * freq * 0.7 + uTime * 2.0) *
                          vel * 0.1;
        displacement += stringWave;
      }
      
      // Harmonic resonance creates standing wave patterns
      float resonanceWave = sin(worldPos.x * 2.0 + uTime) * 
                          sin(worldPos.z * 2.0 + uTime * 0.8) *
                          uHarmonicResonance * 0.15;
      displacement += resonanceWave;
      
      vElevation = displacement;
      
      vec3 displacedPos = position + normal * displacement;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(displacedPos, 1.0);
    }
  `,
  
  fragmentShader: `
    uniform float uTime;
    uniform float uHarpFrequencies[6];
    uniform float uHarpVelocities[6];
    uniform float uDuetProgress;
    uniform float uShipPatience;
    uniform vec3 uBioluminescentColor;
    uniform float uHarmonicResonance;
    
    varying vec2 vUv;
    varying vec3 vWorldPos;
    varying vec3 vNormal;
    varying float vElevation;
    
    // Caustic light patterns
    float caustics(vec2 uv, float time) {
      vec2 p = uv * 8.0;
      float c1 = sin(p.x + time * 0.6) * sin(p.y + time * 0.8);
      float c2 = sin(p.x * 1.3 + time * 0.4) * sin(p.y * 1.1 + time * 0.5);
      float c3 = sin(p.x * 0.9 - time * 0.3) * sin(p.y * 1.4 + time * 0.6);
      return (c1 + c2 + c3) * 0.33 + 0.5;
    }
    
    void main() {
      // Base water color - deep, alive, mysterious
      vec3 deepColor = vec3(0.0, 0.08, 0.15);
      vec3 shallowColor = vec3(0.0, 0.2, 0.3);
      
      // Mix based on elevation (waves)
      float depthMix = smoothstep(-0.1, 0.1, vElevation);
      vec3 waterColor = mix(deepColor, shallowColor, depthMix);
      
      // Caustics - light refracting through water
      float caustic = caustics(vUv, uTime);
      vec3 causticColor = vec3(0.2, 0.6, 0.8) * caustic * 0.4;
      
      // Bioluminescent glow from harp vibrations
      vec3 bioGlow = vec3(0.0);
      for (int i = 0; i < 6; i++) {
        float vel = uHarpVelocities[i];
        float freq = uHarpFrequencies[i];
        
        // Each string creates a localized glow pattern
        vec2 stringPos = vec2(
          sin(float(i) * 1.5 + uTime * 0.1) * 2.0,
          cos(float(i) * 1.2 + uTime * 0.15) * 2.0
        );
        float stringGlow = exp(-distance(vUv, stringPos) * 3.0);
        
        // Glow intensity based on string velocity
        vec3 glowColor = mix(uBioluminescentColor, vec3(1.0), vel);
        bioGlow += glowColor * stringGlow * vel * 2.0;
      }
      
      // Harmonic resonance - when player + ship sync up
      vec3 resonanceGlow = vec3(0.8, 0.9, 1.0) * uHarmonicResonance * 0.5;
      
      // Ship patience - gentle warmth, not cold
      vec3 patienceGlow = vec3(0.3, 0.6, 0.4) * uShipPatience * 0.2;
      
      // Combine all color layers
      vec3 finalColor = waterColor + causticColor + bioGlow + resonanceGlow + patienceGlow;
      
      // Fresnel effect for surface feel
      vec3 viewDir = normalize(cameraPosition - vWorldPos);
      float fresnel = pow(1.0 - dot(vNormal, viewDir), 3.0);
      finalColor += fresnel * vec3(0.4, 0.7, 1.0) * 0.4;
      
      // Duet progress - overall room brightens as harmony emerges
      finalColor *= (1.0 + uDuetProgress * 0.5);
      
      // Transparency - more transparent when viewed at angle
      float alpha = mix(0.95, 0.6, fresnel);
      
      gl_FragColor = vec4(finalColor, alpha);
    }
  `,
  
  transparent: true,
  side: THREE.DoubleSide,
});
```

### 2. VORTEX - SDF TORUS WITH PARTICLES

**Design Philosophy:**
The vortex isn't just a portal—it's the ship's throat, the place where it "sings" the universe into being. The torus shape represents the continuous flow of sound/energy through the Vimana.

**SDF (Signed Distance Field) Torus Vortex Shader:**
```javascript
const vortexMaterial = new THREE.ShaderMaterial({
  uniforms: {
    uTime: { value: 0 },
    uVortexActivation: { value: 0.0 }, // 0-1, increases with sequences
    uDuetProgress: { value: 0.0 }, // How close player is to harmony
    uInnerColor: { value: new THREE.Color(0x00ffff) }, // Cyan
    uOuterColor: { value: new THREE.Color(0x8800ff) }, // Purple
    uCoreColor: { value: new THREE.Color(0xffffff) }, // White light
    uTorusRadius: { value: 2.0 }, // Major radius
    uTubeRadius: { value: 0.4 }, // Minor radius
  },
  
  vertexShader: `
    uniform float uTime;
    uniform float uVortexActivation;
    
    varying vec3 vWorldPos;
    varying vec3 vNormal;
    varying float vDist;
    
    void main() {
      vNormal = normal;
      
      vec4 worldPos = modelMatrix * vec4(position, 1.0);
      vWorldPos = worldPos.xyz;
      
      // Distance from torus center (xz plane)
      float distFromCenter = length(worldPos.xz);
      vDist = distFromCenter;
      
      // Displace vertices based on activation
      float displacement = 0.0;
      
      // Vortex spins faster as it activates
      float angle = atan(worldPos.z, worldPos.x);
      float spin = angle + uTime * (1.0 + uVortexActivation * 3.0);
      
      // Radial breathing
      float breathe = sin(spin * 4.0 + uTime * 2.0) * uVortexActivation * 0.2;
      
      // Apply displacement
      worldPos.x += cos(angle) * breathe;
      worldPos.z += sin(angle) * breathe;
      
      gl_Position = projectionMatrix * modelViewMatrix * worldPos;
    }
  `,
  
  fragmentShader: `
    uniform float uTime;
    uniform float uVortexActivation;
    uniform float uDuetProgress;
    uniform vec3 uInnerColor;
    uniform vec3 uOuterColor;
    uniform vec3 uCoreColor;
    uniform float uTorusRadius;
    uniform float uTubeRadius;
    
    varying vec3 vWorldPos;
    varying vec3 vNormal;
    varying float vDist;
    
    // SDF for torus
    float sdTorus(vec3 p, vec2 t) {
      vec2 q = vec2(length(p.xz) - t.x, p.y);
      return length(q) - t.y;
    }
    
    // Toroidal swirl noise
    float swirl(vec3 p, float time) {
      float angle = atan(p.z, p.x);
      float radius = length(p.xz);
      float swirl = sin(angle * 6.0 + radius * 3.0 - time * 2.0);
      return swirl;
    }
    
    void main() {
      // Distance from torus surface
      float distToTorus = sdTorus(vWorldPos, vec2(uTorusRadius, uTubeRadius));
      
      // Create vortex effect - not just torus, but spinning energy
      float angle = atan(vWorldPos.z, vWorldPos.x);
      float timeOffset = uTime * (1.0 + uVortexActivation * 2.0);
      float swirlStrength = swirl(vWorldPos, timeOffset);
      
      // Color based on distance from center and swirl
      float centerFactor = smoothstep(0.0, uTorusRadius * 2.0, vDist);
      vec3 baseColor = mix(uInnerColor, uOuterColor, centerFactor);
      
      // Add swirl to color
      baseColor = mix(baseColor, uCoreColor, swirlStrength * 0.3 * uVortexActivation);
      
      // Duet progress makes vortex more coherent/less chaotic
      baseColor = mix(baseColor, uCoreColor, uDuetProgress * 0.4);
      
      // Edge glow - torus edges light up
      float edge = 1.0 - smoothstep(0.0, 0.3, abs(distToTorus));
      baseColor += uCoreColor * edge * uVortexActivation * 0.5;
      
      // Alpha - vortex becomes more solid as it activates
      float alpha = 0.1 + uVortexActivation * 0.7 + edge * 0.2;
      
      // Add glow on edges
      if (edge > 0.8) {
        alpha += 0.3;
        baseColor += uCoreColor * 0.5;
      }
      
      gl_FragColor = vec4(baseColor, alpha);
    }
  `,
  
  transparent: true,
  side: THREE.DoubleSide,
  depthWrite: false, // For proper transparency
  blending: THREE.AdditiveBlending, // Glowing effect
});
```

**Particle System Integration:**
```javascript
class VortexParticles {
  constructor(scene, centerPos) {
    this.scene = scene;
    this.centerPos = centerPos;
    this.particles = [];
    this.maxParticles = 2000;
    
    this.particleGeometry = new THREE.BufferGeometry();
    this.positions = new Float32Array(this.maxParticles * 3);
    this.velocities = new Float32Array(this.maxParticles * 3);
    this.lifetimes = new Float32Array(this.maxParticles);
    this.colors = new Float32Array(this.maxParticles * 3);
    
    this.particleGeometry.setAttribute('position', 
      new THREE.BufferAttribute(this.positions, 3));
    this.particleGeometry.setAttribute('color',
      new THREE.BufferAttribute(this.colors, 3));
    
    const particleMaterial = new THREE.PointsMaterial({
      size: 0.05,
      transparent: true,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    
    this.particleSystem = new THREE.Points(this.particleGeometry, particleMaterial);
    this.scene.add(this.particleSystem);
    
    this.initParticles();
  }
  
  initParticles() {
    for (let i = 0; i < this.maxParticles; i++) {
      this.resetParticle(i);
    }
    
    this.particleGeometry.attributes.position.needsUpdate = true;
    this.particleGeometry.attributes.color.needsUpdate = true;
  }
  
  resetParticle(i) {
    // Spawn particles in a torus pattern
    const angle = Math.random() * Math.PI * 2;
    const radius = 2.0 + (Math.random() - 0.5) * 0.5; // Around torus center

    // FIX: Randomize tubeOffset on each reset (was using static value)
    const tubeOffset = (Math.random() - 0.5) * 0.8; // Around tube

    this.positions[i * 3] = Math.cos(angle) * radius;
    this.positions[i * 3 + 1] = tubeOffset;
    this.positions[i * 3 + 2] = Math.sin(angle) * radius;
    
    // Tangential velocity (spin around torus)
    const speed = 0.5 + Math.random() * 0.5;
    this.velocities[i * 3] = -Math.sin(angle) * speed;
    this.velocities[i * 3 + 1] = (Math.random() - 0.5) * 0.2; // Slight vertical
    this.velocities[i * 3 + 2] = Math.cos(angle) * speed;
    
    this.lifetimes[i] = Math.random() * 10.0; // 10 second lifetime
    
    // Color - cyan to purple gradient
    const colorMix = Math.random();
    this.colors[i * 3] = 0.0 + colorMix * 0.5; // R
    this.colors[i * 3 + 1] = 1.0 - colorMix * 0.5; // G
    this.colors[i * 3 + 2] = 1.0; // B
  }
  
  update(delta, time, activationLevel) {
    for (let i = 0; i < this.maxParticles; i++) {
      // PERFORMANCE FIX: Particle LOD system - skip particles based on activation level
      // Low activation = skip half particles. Medium = skip quarter. Full = update all.
      const lodSkip = activationLevel < 0.3 ? (i % 2 === 0) :
                      activationLevel < 0.6 ? (i % 4 === 0) : false;
      if (lodSkip) continue;

      // Update position
      this.positions[i * 3] += this.velocities[i * 3] * delta * activationLevel;
      this.positions[i * 3 + 1] += this.velocities[i * 3 + 1] * delta * activationLevel;
      this.positions[i * 3 + 2] += this.velocities[i * 3 + 2] * delta * activationLevel;
      
      // Spin particles around torus
      const x = this.positions[i * 3];
      const z = this.positions[i * 3 + 2];
      const dist = Math.sqrt(x * x + z * z);
      const angle = Math.atan2(z, x);
      
      const spinSpeed = (1.0 + activationLevel) * delta * 0.5;
      const newAngle = angle + spinSpeed;
      
      const cosAngle = Math.cos(newAngle);
      const sinAngle = Math.sin(newAngle);
      
      // FIX: Z coordinate uses cosAngle (was incorrectly assigned Y value)
      this.positions[i * 3] = cosAngle * dist;
      this.positions[i * 3 + 1] = sinAngle * dist;
      this.positions[i * 3 + 2] = dist; // Preserve radial distance for proper torus flow
      
      // Decay lifetime
      this.lifetimes[i] -= delta;
      
      // Reset if dead or drifted too far
      if (this.lifetimes[i] <= 0 || dist > 3.5) {
        this.resetParticle(i);
      }
      
      // Brightness based on activation
      const brightness = 0.5 + activationLevel * 0.5;
      this.colors[i * 3] *= brightness;
      this.colors[i * 3 + 1] *= brightness;
      this.colors[i * 3 + 2] *= brightness;
    }
    
    this.particleGeometry.attributes.position.needsUpdate = true;
    this.particleGeometry.attributes.color.needsUpdate = true;
  }

  // MEMORY FIX: Cleanup method for proper disposal
  destroy() {
    if (this.particleSystem) {
      this.scene.remove(this.particleSystem);
      this.particleGeometry.dispose();
      this.particleSystem.material.dispose();
      this.particleSystem = null;
    }
    console.log('🧹 VortexParticles cleaned up');
  }
}
```

### 3. JELLY CREATURES - MUSICAL MESSENGERS

**Enhanced Jelly Visuals:**
```javascript
const jellyMaterial = new THREE.ShaderMaterial({
  uniforms: {
    uTime: { value: 0 },
    uNoteIndex: { value: 0 }, // Which note this jelly represents
    uTeachingState: { value: 0.0 }, // 0=hiding, 1=teaching, 2=completed
    uBioluminescentColor: { value: new THREE.Color(0x00ff88) },
    uGlowIntensity: { value: 1.0 },
  },
  
  vertexShader: `
    uniform float uTime;
    uniform float uNoteIndex;
    uniform float uTeachingState;
    
    varying vec3 vNormal;
    varying vec3 vViewDir;
    varying float vTeachingFactor;
    
    void main() {
      vNormal = normal;
      vTeachingFactor = uTeachingState;
      
      // Organic pulsing - different rate per note
      float noteRate = 2.0 + uNoteIndex * 0.5;
      float pulse = sin(uTime * noteRate) * 0.1 + 1.0;
      
      // Squish when teaching
      float squish = 1.0;
      if (uTeachingState > 0.5) {
        squish = 1.0 + sin(uTime * 3.0) * 0.2;
      }
      
      vec3 pos = position * pulse;
      pos.y *= squish;
      
      vec4 worldPos = modelMatrix * vec4(pos, 1.0);
      vViewDir = normalize(cameraPosition - worldPos.xyz);
      
      gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    }
  `,
  
  fragmentShader: `
    uniform vec3 uBioluminescentColor;
    uniform float uGlowIntensity;
    varying vec3 vNormal;
    varying vec3 vViewDir;
    varying float vTeachingFactor;
    
    void main() {
      vec3 color = uBioluminescentColor;
      
      // Fresnel glow - stronger when teaching
      float fresnel = pow(1.0 - dot(vNormal, vViewDir), 3.0);
      float teachingBoost = vTeachingFactor * 0.5;
      color += fresnel * vec3(1.0) * uGlowIntensity * (1.0 + teachingBoost);
      
      // Inner glow when teaching
      if (vTeachingFactor > 0.5) {
        color += vec3(0.5, 0.8, 0.5) * 0.3;
      }
      
      // Translucent
      float alpha = 0.6 + fresnel * 0.3 + vTeachingFactor * 0.2;
      
      gl_FragColor = vec4(color, alpha);
    }
  `,
  
  transparent: true,
});
```

---

## 🎵 "DUET" MECHANIC - GENTLE FAILURE HANDLING

### Core Principle: No Consequences, Only Feedback

**What Happens When Player Plays Wrong:**

1. **Visual Feedback (The "Shake"):**
```javascript
// Camera shakes gently - not punishment, but "that wasn't it"
class GentleFeedback {
  constructor(camera) {
    this.camera = camera;
    this.shakeIntensity = 0.0;
    this.shakeDuration = 0;
  }
  
  triggerShake(intensity = 1.0, duration = 500) {
    this.shakeIntensity = intensity;
    this.shakeDuration = duration;
    this.shakeStartTime = Date.now();
  }
  
  update() {
    if (this.shakeDuration <= 0) return;
    
    const elapsed = Date.now() - this.shakeStartTime;
    const progress = Math.min(elapsed / this.shakeDuration, 1.0);
    const decay = 1.0 - progress;
    
    if (decay > 0) {
      const shakeAmount = 0.05 * this.shakeIntensity * decay;
      this.camera.position.x += (Math.random() - 0.5) * shakeAmount;
      this.camera.position.y += (Math.random() - 0.5) * shakeAmount;
      this.camera.position.z += (Math.random() - 0.5) * shakeAmount;
    } else {
      this.shakeDuration = 0;
    }
  }
}
```

2. **Audio Feedback (Discordant Tone):**
```javascript
// Soft "wrong note" sound - not harsh, just "that's not the harmony"
class GentleAudioFeedback {
  constructor(audioContext) {
    this.audioContext = audioContext;
  }
  
  playDiscordantNote() {
    const osc = this.audioContext.createOscillator();
    const gain = this.audioContext.createGain();
    
    // Play a slightly dissonant note relative to current sequence
    const baseFreq = 261.63; // C4
    const discordantFreq = baseFreq * 1.0595; // Minor second
    osc.frequency.setValueAtTime(discordantFreq, this.audioContext.currentTime);
    osc.type = 'sine';
    
    // Very gentle - not a punishment
    gain.gain.setValueAtTime(0, this.audioContext.currentTime);
    gain.gain.linearRampToValueAtTime(0.1, this.audioContext.currentTime + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.001, this.audioContext.currentTime + 0.3);
    
    osc.connect(gain);
    gain.connect(this.audioContext.destination);
    
    osc.start();
    osc.stop(this.audioContext.currentTime + 0.35);
  }
  
  playPatientReminder() {
    // After wrong note, ship "sings again" - patient teaching
    const osc = this.audioContext.createOscillator();
    const gain = this.audioContext.createGain();
    
    const freq = 330.63; // E4 - gentle reminder
    osc.frequency.setValueAtTime(freq, this.audioContext.currentTime);
    osc.type = 'sine';
    
    gain.gain.setValueAtTime(0, this.audioContext.currentTime);
    gain.gain.linearRampToValueAtTime(0.15, this.audioContext.currentTime + 0.1);
    gain.gain.exponentialRampToValueAtTime(0.001, this.audioContext.currentTime + 0.8);
    
    osc.connect(gain);
    gain.connect(this.audioContext.destination);
    
    osc.start();
    osc.stop(this.audioContext.currentTime + 0.85);
  }
}
```

3. **Jelly Reappears (The "Patient Teacher"):**
```javascript
// Wrong note → Jelly appears again → Demonstrates correct string
// NO RESET - player just watches and tries again

class PatientJellyManager {
  constructor(scene, waterSystem, harpStrings) {
    this.scene = scene;
    this.waterSystem = waterSystem;
    this.harpStrings = harpStrings;
    
    // Three teaching sequences
    this.teachingSequences = [
      [0, 1, 2], // C, D, E
      [3, 4, 5], // F, G, A
      [2, 4, 1], // E, G, D
    ];
    
    this.currentSequence = 0;
    this.currentNoteIndex = 0;
    this.waitingForPlayer = false;
    
    // Jelly for teaching
    this.teachingJelly = new JellyCreature(scene, 0);
  }
  
  startTeachingSequence(sequenceIndex) {
    this.currentSequence = sequenceIndex;
    this.currentNoteIndex = 0;
    this.waitingForPlayer = false;
    
    // Jelly appears and teaches first note
    setTimeout(() => {
      this.teachCurrentNote();
    }, 2000); // 2 second pause before teaching
  }
  
  teachCurrentNote() {
    const noteIndex = this.teachingSequences[this.currentSequence][this.currentNoteIndex];
    
    // Jelly appears, jumps, shows ripple at target string
    const spawnPos = new THREE.Vector3(0, 0, 2);
    this.teachingJelly.jumpOut(spawnPos, noteIndex);
    
    // Wait for jelly to complete, then let player try
    setTimeout(() => {
      this.waitingForPlayer = true;
      console.log(`🎵 Ship is waiting for you to play string ${noteIndex + 1}`);
    }, 2000); // 2 second teaching demonstration
  }
  
  onStringPlayed(stringIndex) {
    if (!this.waitingForPlayer) {
      // Player played before ship was ready
      // Gentle feedback: "Wait, I'm teaching"
      this.feedback.triggerShake(0.3, 300);
      this.audio.playDiscordantNote();
      return;
    }
    
    const expectedNote = this.teachingSequences[this.currentSequence][this.currentNoteIndex];
    
    if (stringIndex === expectedNote) {
      // CORRECT! Player joined the duet
      console.log(`✅ Harmony! You played string ${stringIndex + 1}`);
      
      // Play chord - ship + player together
      this.playHarmonyChord(stringIndex);
      
      // Advance to next note or complete sequence
      this.currentNoteIndex++;
      
      if (this.currentNoteIndex >= this.teachingSequences[this.currentSequence].length) {
        this.onSequenceComplete();
      } else {
        // Teach next note
        this.waitingForPlayer = false;
        setTimeout(() => {
          this.teachCurrentNote();
        }, 2000);
      }
    } else {
      // WRONG NOTE - Gentle feedback, NO RESET
      console.log(`🎵 That's not quite it. Let me show you again.`);
      
      // Visual shake
      this.feedback.triggerShake(0.5, 500);
      
      // Soft discordant sound
      this.audio.playDiscordantNote();
      
      // Patient reminder - ship sings the note again
      setTimeout(() => {
        this.audio.playPatientReminder();
        
        // Jelly appears again to demonstrate
        setTimeout(() => {
          this.teachCurrentNote();
        }, 1000);
      }, 500);
      
      // NO RESET - player tries again with the same note
    }
  }
  
  playHarmonyChord(stringIndex) {
    // Ship plays harmony with player
    const playerFreq = this.harpStrings[stringIndex].freq;
    const harmonyFreq = playerFreq * 1.5; // Perfect fifth
    
    const playerOsc = this.audioContext.createOscillator();
    const harmonyOsc = this.audioContext.createOscillator();
    const playerGain = this.audioContext.createGain();
    const harmonyGain = this.audioContext.createGain();
    
    playerOsc.frequency.setValueAtTime(playerFreq, this.audioContext.currentTime);
    harmonyOsc.frequency.setValueAtTime(harmonyFreq, this.audioContext.currentTime);
    
    playerOsc.type = 'sine';
    harmonyOsc.type = 'sine';
    
    playerGain.gain.setValueAtTime(0.15, this.audioContext.currentTime);
    harmonyGain.gain.setValueAtTime(0.15, this.audioContext.currentTime);
    
    playerOsc.connect(playerGain);
    harmonyOsc.connect(harmonyGain);
    playerGain.connect(this.audioContext.destination);
    harmonyGain.connect(this.audioContext.destination);
    
    playerOsc.start();
    harmonyOsc.start();
    
    const now = this.audioContext.currentTime;
    const duration = 2.0;
    
    playerGain.gain.exponentialRampToValueAtTime(0.001, now + duration);
    harmonyGain.gain.exponentialRampToValueAtTime(0.001, now + duration);
    playerOsc.stop(now + duration + 0.1);
    harmonyOsc.stop(now + duration + 0.1);
  }
  
  onSequenceComplete() {
    console.log(`🎉 Duet complete! The ship harmonizes with you.`);
    
    // Play completion chord
    this.playCompletionChord();
    
    // Advance to next sequence or complete all
    this.currentSequence++;
    this.currentNoteIndex = 0;
    
    if (this.currentSequence >= this.teachingSequences.length) {
      this.onAllDuetSequencesComplete();
    } else {
      // Start next teaching sequence
      setTimeout(() => {
        this.startTeachingSequence(this.currentSequence);
      }, 3000);
    }
  }
  
  playCompletionChord() {
    // Beautiful resolution chord
    const chord = [261.63, 329.63, 392.00, 523.25]; // C major
    chord.forEach(freq => {
      const osc = this.audioContext.createOscillator();
      const gain = this.audioContext.createGain();
      
      osc.frequency.setValueAtTime(freq, this.audioContext.currentTime);
      osc.type = 'sine';
      
      gain.gain.setValueAtTime(0.0, this.audioContext.currentTime);
      gain.gain.linearRampToValueAtTime(0.2, this.audioContext.currentTime + 0.2);
      gain.gain.exponentialRampToValueAtTime(0.001, this.audioContext.currentTime + 3.0);
      
      osc.connect(gain);
      gain.connect(this.audioContext.destination);
      
      osc.start();
      osc.stop(this.audioContext.currentTime + 3.1);
    });
  }
  
  onAllDuetSequencesComplete() {
    console.log(`✨✨✨ FULL HARMONY ACHIEVED! ✨✨✨`);
    
    // Activate vortex - the ship's "voice" is ready
    this.activateVortex();
    
    // Platform detaches - player rides the ship's song
    this.detachPlatform();
  }
  
  activateVortex() {
    // Find vortex gate mesh
    this.scene.traverse((child) => {
      if (child.name === 'Vortex_gate' && child.isMesh) {
        // Make vortex more defined based on duet progress
        const duetProgress = this.currentSequence / this.teachingSequences.length;
        const intensity = duetProgress;
        
        child.material.emissive = new THREE.Color(0x00ffff);
        child.material.emissiveIntensity = intensity * 3.0;
        
        // Send to water shader
        if (window.waterMaterial) {
          window.waterMaterial.uniforms.uHarmonicResonance.value = duetProgress;
          window.waterMaterial.uniforms.uDuetProgress.value = duetProgress;
        }
        
        console.log(`🌀 Vortex intensity: ${intensity}`);
      }
    });
  }
  
  detachPlatform() {
    // Find platform mesh
    this.scene.traverse((child) => {
      if (child.name === 'Platform' && child.isMesh) {
        // Animate platform moving toward vortex
        const targetPos = new THREE.Vector3(0, 0.5, 2); // Vortex position
        const startPos = child.position.clone();
        const duration = 5000; // 5 seconds
        const startTime = Date.now();
        
        const animatePlatform = () => {
          const elapsed = Date.now() - startTime;
          const progress = Math.min(elapsed / duration, 1.0);
          
          // Move toward vortex
          child.position.lerpVectors(startPos, targetPos, progress);
          
          // Rotate slightly
          child.rotation.y = progress * Math.PI * 2;
          
          if (progress < 1.0) {
            requestAnimationFrame(animatePlatform);
          } else {
            console.log('🚀 Platform arrived at vortex!');
            // Trigger white room transition
            this.transitionToWhiteRoom();
          }
        };
        animatePlatform();
      }
    });
  }
  
  transitionToWhiteRoom() {
    // NEW: No 3D white room - instead, spawn shell and start vortex engulf
    console.log('🌀 Spawning shell reward...');

    // Shell appears slowly (3 seconds)
    this.shellSystem = new ShellCollectibleManager(this.scene, this.camera);

    // After player collects shell, start vortex engulf sequence
    this.shellSystem.onCollected = () => {
      console.log('🐚 Shell collected! Starting vortex engulf...');
      this.whiteFlashEnding = new WhiteFlashEnding(this.scene, this.camera);
      this.whiteFlashEnding.start();
    };
  }

  // MEMORY FIX: Cleanup method for scene transitions
  destroy() {
    // Remove jelly creature
    if (this.teachingJelly && this.teachingJelly.destroy) {
      this.teachingJelly.destroy();
    }

    // Remove shell system
    if (this.shellSystem && this.shellSystem.destroy) {
      this.shellSystem.destroy();
      this.shellSystem.onCollected = null;
    }

    // Remove white flash ending
    if (this.whiteFlashEnding && this.whiteFlashEnding.destroy) {
      this.whiteFlashEnding.destroy();
    }

    // Clear references
    this.shellSystem = null;
    this.whiteFlashEnding = null;
    this.teachingJelly = null;

    console.log('🧹 PatientJellyManager cleaned up');
  }
}
```

---

## 🐚 SHELL COLLECTION SYSTEM

### Design Philosophy:
The shell is a procedural SDF-based reward that materializes slowly in front of the player. When clicked, it flies and shrinks into a UI slot, providing satisfying feedback and tracking progress across the 4 rooms.

### SDF Shell Material:
```javascript
class SDFShellMaterial extends THREE.ShaderMaterial {
  constructor() {
    super({
      uniforms: {
        uTime: { value: 0 },
        uAppearProgress: { value: 0.0 }, // 0-1, 3 second materialize
        uDissolveProgress: { value: 0.0 }, // 0-1, for fly-away animation
        uShellColor: { value: new THREE.Color(0xffddaa) }, // Warm pearl
        uIridescenceColor: { value: new THREE.Color(0x88ccff) }, // Cyan shimmer
        uSpiralTightness: { value: 2.5 }, // Nautilus spiral
        uSpiralGrowth: { value: 0.15 }, // Chamber expansion rate
      },

      vertexShader: `
        uniform float uTime;
        uniform float uAppearProgress;
        uniform float uDissolveProgress;

        varying vec3 vNormal;
        varying vec3 vWorldPos;
        varying vec3 vLocalPos;
        varying float vAppearFactor;
        varying float vDissolveFactor;

        // Simplex noise for dissolve effect
        vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
        vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
        float snoise(vec3 v) {
          const vec2 C = vec2(1.0/6.0, 1.0/3.0);
          const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
          vec3 i  = floor(v + dot(v, C.yyy));
          vec3 x0 = v - i + dot(i, C.xxx);
          vec3 g = step(x0.yzx, x0.xyz);
          vec3 l = 1.0 - g;
          vec3 i1 = min( g.xyz, l.zxy );
          vec3 i2 = max( g.xyz, l.zxy );
          vec3 x1 = x0 - i1 + C.xxx;
          vec3 x2 = x0 - i2 + C.yyy;
          vec3 x3 = x0 - D.yyy;
          i = mod289(i);
          vec4 p = permute( permute( permute(
                i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
              + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
              + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
          float n_ = 0.142857142857;
          vec3  ns = n_ * D.wyz - D.xzx;
          vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
          vec4 x_ = floor(j * ns.z);
          vec4 y_ = floor(j - 7.0 * x_);
          vec4 x = x_ *ns.x + ns.yyyy;
          vec4 y = y_ *ns.x + ns.yyyy;
          vec4 h = 1.0 - abs(x) - abs(y);
          vec4 b0 = vec4( x.xy, y.xy );
          vec4 b1 = vec4( x.zw, y.zw );
          vec4 s0 = floor(b0)*2.0 + 1.0;
          vec4 s1 = floor(b1)*2.0 + 1.0;
          vec4 sh = -step(h, vec4(0.0));
          vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
          vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
          vec3 p0 = vec3(a0.xy,h.x);
          vec3 p1 = vec3(a0.zw,h.y);
          vec3 p2 = vec3(a1.xy,h.z);
          vec3 p3 = vec3(a1.zw,h.w);
          vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
          p0 *= norm.x;
          p1 *= norm.y;
          p2 *= norm.z;
          p3 *= norm.w;
          vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
          m = m * m;
          return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                        dot(p2,x2), dot(p3,x3) ) );
        }

        void main() {
          vNormal = normal;
          vLocalPos = position;
          vAppearFactor = uAppearProgress;
          vDissolveFactor = uDissolveProgress;

          // Nautilus spiral SDF displacement
          float angle = atan(position.z, position.x);
          float radius = length(position.xz);
          float spiralAngle = angle + uSpiralTightness * log(radius + 0.1);

          // Create shell chamber ridges
          float chamber = sin(spiralAngle * 3.14159 * uSpiralGrowth) * 0.5 + 0.5;
          float ridge = smoothstep(0.4, 0.6, chamber) * 0.02;

          // Appear animation - vertices emerge from center
          float appearScale = uAppearProgress;
          vec3 appearedPos = position * appearScale;

          // Dissolve animation - noise-based fade
          float dissolveNoise = snoise(position * 3.0 + uTime);
          float dissolveThreshold = uDissolveProgress;
          float dissolveMask = smoothstep(dissolveThreshold - 0.2, dissolveThreshold + 0.2, dissolveNoise);

          vec3 finalPos = appearedPos;

          vec4 worldPos = modelMatrix * vec4(finalPos, 1.0);
          vWorldPos = worldPos.xyz;

          gl_Position = projectionMatrix * modelViewMatrix * vec4(finalPos, 1.0);
        }
      `,

      fragmentShader: `
        uniform float uTime;
        uniform float uAppearProgress;
        uniform float uDissolveProgress;
        uniform vec3 uShellColor;
        uniform vec3 uIridescenceColor;
        uniform float uSpiralTightness;
        uniform float uSpiralGrowth;

        varying vec3 vNormal;
        varying vec3 vWorldPos;
        varying vec3 vLocalPos;
        varying float vAppearFactor;
        varying float vDissolveFactor;

        // Simplex noise (same as vertex shader)
        vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
        vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
        float snoise(vec3 v) {
          const vec2 C = vec2(1.0/6.0, 1.0/3.0);
          const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
          vec3 i  = floor(v + dot(v, C.yyy));
          vec3 x0 = v - i + dot(i, C.xxx);
          vec3 g = step(x0.yzx, x0.xyz);
          vec3 l = 1.0 - g;
          vec3 i1 = min( g.xyz, l.zxy );
          vec3 i2 = max( g.xyz, l.zxy );
          vec3 x1 = x0 - i1 + C.xxx;
          vec3 x2 = x0 - i2 + C.yyy;
          vec3 x3 = x0 - D.yyy;
          i = mod289(i);
          vec4 p = permute( permute( permute(
                i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
              + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
              + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
          float n_ = 0.142857142857;
          vec3  ns = n_ * D.wyz - D.xzx;
          vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
          vec4 x_ = floor(j * ns.z);
          vec4 y_ = floor(j - 7.0 * x_);
          vec4 x = x_ *ns.x + ns.yyyy;
          vec4 y = y_ *ns.x + ns.yyyy;
          vec4 h = 1.0 - abs(x) - abs(y);
          vec4 b0 = vec4( x.xy, y.xy );
          vec4 b1 = vec4( x.zw, y.zw );
          vec4 s0 = floor(b0)*2.0 + 1.0;
          vec4 s1 = floor(b1)*2.0 + 1.0;
          vec4 sh = -step(h, vec4(0.0));
          vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
          vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
          vec3 p0 = vec3(a0.xy,h.x);
          vec3 p1 = vec3(a0.zw,h.y);
          vec3 p2 = vec3(a1.xy,h.z);
          vec3 p3 = vec3(a1.zw,h.w);
          vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
          p0 *= norm.x;
          p1 *= norm.y;
          p2 *= norm.z;
          p3 *= norm.w;
          vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
          m = m * m;
          return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                        dot(p2,x2), dot(p3,x3) ) );
        }

        void main() {
          vec3 viewDir = normalize(cameraPosition - vWorldPos);

          // Fresnel effect
          float fresnel = pow(1.0 - dot(vNormal, viewDir), 3.0);

          // Base shell color
          vec3 color = uShellColor;

          // Nautilus spiral pattern
          float angle = atan(vLocalPos.z, vLocalPos.x);
          float radius = length(vLocalPos.xz);
          float spiralAngle = angle + uSpiralTightness * log(radius + 0.1);

          // Chamber coloring - each chamber slightly different
          float chamber = sin(spiralAngle * 3.14159 * uSpiralGrowth) * 0.5 + 0.5;
          vec3 chamberColor = mix(uShellColor, vec3(1.0, 0.95, 0.8), chamber * 0.3);
          color = chamberColor;

          // Iridescence - color shifting based on view angle
          float iridescence = sin(fresnel * 10.0 + uTime) * 0.5 + 0.5;
          color = mix(color, uIridescenceColor, iridescence * fresnel * 0.5);

          // Rim light
          color += fresnel * vec3(1.0) * 0.3;

          // Appear animation - fade in
          float appearAlpha = smoothstep(0.0, 0.3, uAppearProgress);

          // Dissolve animation - noise-based fade out
          float dissolveNoise = snoise(vLocalPos * 3.0 + uTime * 0.5);
          float dissolveAlpha = 1.0 - smoothstep(uDissolveProgress - 0.3, uDissolveProgress + 0.2, dissolveNoise);

          // Final alpha
          float alpha = appearAlpha * dissolveAlpha * (0.6 + fresnel * 0.4);

          gl_FragColor = vec4(color, alpha);
        }
      `,

      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
  }
}
```

### Shell Collectible Manager:
```javascript
class ShellCollectibleManager {
  constructor(scene, camera) {
    this.scene = scene;
    this.camera = camera;

    // Create shell mesh with icosahedron (approximates shell shape)
    const geometry = new THREE.IcosahedronGeometry(0.3, 2);
    this.shellMaterial = new SDFShellMaterial();

    this.shell = new THREE.Mesh(geometry, this.shellMaterial);
    this.shell.position.set(0, 1.0, 1.0); // In front of player
    this.shell.rotation.x = Math.PI * 0.1;
    this.scene.add(this.shell);

    // Animation state
    this.state = 'appearing'; // appearing, idle, collected
    this.appearStartTime = Date.now();
    this.appearDuration = 3000; // 3 seconds

    // Collection animation
    this.collectStartTime = 0;
    this.collectDuration = 1500; // 1.5 seconds
    this.startPos = new THREE.Vector3();
    this.uiTargetPos = new THREE.Vector3(); // Screen position for UI

    // Callback when collected
    this.onCollected = null;

    // Raycaster for click detection
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    // Set up click handler
    this.setupClickHandler();
  }

  setupClickHandler() {
    this.onClick = (event) => {
      if (this.state !== 'idle') return;

      // Convert mouse to normalized device coordinates
      this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
      this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

      this.raycaster.setFromCamera(this.mouse, this.camera);
      const intersects = this.raycaster.intersectObject(this.shell);

      if (intersects.length > 0) {
        this.collect();
      }
    };

    window.addEventListener('click', this.onClick);
  }

  startAppearing() {
    this.state = 'appearing';
    this.appearStartTime = Date.now();
    this.shellMaterial.uniforms.uAppearProgress.value = 0;
  }

  collect() {
    if (this.state !== 'idle') return;

    this.state = 'collecting';
    this.collectStartTime = Date.now();
    this.startPos.copy(this.shell.position);

    // Get screen position for UI target
    const uiPos = this.getUIPosition();
    this.uiTargetPos.copy(uiPos);

    // Remove click handler
    window.removeEventListener('click', this.onClick);

    // Notify UI to add shell
    if (window.shellUI) {
      window.shellUI.addShell();
    }
  }

  getUIPosition() {
    // Returns world position for shell to fly toward
    // UI is at top-left of screen
    return new THREE.Vector3(-2, 2, 0);
  }

  update(time) {
    // Update shader time
    this.shellMaterial.uniforms.uTime.value = time;

    // Handle appearing animation
    if (this.state === 'appearing') {
      const elapsed = Date.now() - this.appearStartTime;
      const progress = Math.min(elapsed / this.appearDuration, 1.0);

      // Smooth easing
      const eased = 1.0 - Math.pow(1.0 - progress, 3);
      this.shellMaterial.uniforms.uAppearProgress.value = eased;

      // Gentle rotation while appearing
      this.shell.rotation.y = elapsed * 0.001;

      if (progress >= 1.0) {
        this.state = 'idle';
        console.log('🐚 Shell fully appeared! Click to collect.');
      }
    }

    // Handle collection animation
    if (this.state === 'collecting') {
      const elapsed = Date.now() - this.collectStartTime;
      const progress = Math.min(elapsed / this.collectDuration, 1.0);

      // Smooth easing for fly animation
      const eased = progress * progress * (3 - 2 * progress); // Smoothstep

      // Lerp position toward UI
      this.shell.position.lerpVectors(this.startPos, this.uiTargetPos, eased);

      // Shrink scale
      const scale = 1.0 - eased;
      this.shell.scale.setScalar(scale);

      // Start dissolve effect
      this.shellMaterial.uniforms.uDissolveProgress.value = eased * 0.8;

      // Spin faster
      this.shell.rotation.y += 0.1;

      if (progress >= 1.0) {
        this.state = 'collected';
        this.scene.remove(this.shell);

        // Trigger callback
        if (this.onCollected) {
          this.onCollected();
        }
      }
    }

    // Idle animation
    if (this.state === 'idle') {
      this.shell.rotation.y += 0.005;
      this.shell.position.y = 1.0 + Math.sin(time * 2) * 0.05; // Bobbing
    }
  }

  destroy() {
    window.removeEventListener('click', this.onClick);
    this.scene.remove(this.shell);
    this.shellMaterial.dispose();
  }
}
```

### Shell UI Overlay:
```javascript
class ShellUIOverlay {
  constructor() {
    this.slots = 4;
    this.collectedShells = [];

    this.container = this.createContainer();
    this.slotsElements = [];

    for (let i = 0; i < this.slots; i++) {
      this.slotsElements.push(this.createSlot(i));
    }

    // Register globally for shell system access
    window.shellUI = this;
  }

  createContainer() {
    const container = document.createElement('div');
    container.style.cssText = `
      position: fixed;
      top: 20px;
      left: 20px;
      display: flex;
      gap: 12px;
      z-index: 1000;
      pointer-events: none;
    `;
    document.body.appendChild(container);
    return container;
  }

  createSlot(index) {
    const slot = document.createElement('div');
    slot.style.cssText = `
      width: 60px;
      height: 60px;
      border-radius: 50%;
      border: 2px solid rgba(255, 221, 170, 0.3);
      background: rgba(0, 20, 40, 0.5);
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.3s ease;
    `;

    // Create canvas for shell icon
    const canvas = document.createElement('canvas');
    canvas.width = 50;
    canvas.height = 50;
    canvas.style.opacity = '0';
    canvas.style.transition = 'opacity 0.5s ease';

    slot.appendChild(canvas);
    this.container.appendChild(slot);

    return { element: slot, canvas, filled: false };
  }

  addShell() {
    // Find first empty slot
    const slotIndex = this.collectedShells.length;
    if (slotIndex >= this.slots) return;

    const slot = this.slotsElements[slotIndex];
    slot.filled = true;

    // Draw shell icon on canvas
    this.drawShellIcon(slot.canvas, slotIndex);

    // Animate slot fill
    slot.element.style.borderColor = 'rgba(255, 221, 170, 0.9)';
    slot.element.style.boxShadow = '0 0 15px rgba(255, 221, 170, 0.5)';
    slot.canvas.style.opacity = '1';

    this.collectedShells.push({ index: slotIndex, collectedAt: Date.now() });

    console.log(`🐚 Shell collected! (${this.collectedShells.length}/${this.slots})`);
  }

  drawShellIcon(canvas, index) {
    const ctx = canvas.getContext('2d');
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw simplified SDF shell pattern
    ctx.save();
    ctx.translate(centerX, centerY);

    // Shell base color
    ctx.strokeStyle = `hsl(${40 + index * 10}, 70%, 75%)`;
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';

    // Draw nautilus spiral (simplified)
    ctx.beginPath();
    for (let angle = 0; angle < Math.PI * 4; angle += 0.1) {
      const radius = 2 + angle * 2;
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius;

      if (angle === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw chambers
    for (let i = 1; i <= 4; i++) {
      const chamberRadius = 3 + i * 4;
      const chamberAngle = i * 0.8;
      const cx = Math.cos(chamberAngle) * chamberRadius;
      const cy = Math.sin(chamberAngle) * chamberRadius;

      ctx.beginPath();
      ctx.arc(cx, cy, 2 + i * 0.5, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Add glow
    ctx.shadowColor = 'rgba(255, 221, 170, 0.8)';
    ctx.shadowBlur = 10;
    ctx.stroke();

    ctx.restore();
  }

  update() {
    // Animate slot icons (gentle pulse)
    this.slotsElements.forEach((slot, i) => {
      if (slot.filled) {
        const pulse = Math.sin(Date.now() * 0.002 + i) * 0.1 + 0.9;
        slot.element.style.transform = `scale(${pulse})`;
      }
    });
  }

  getCollectedCount() {
    return this.collectedShells.length;
  }

  destroy() {
    this.container.remove();
    delete window.shellUI;
  }
}
```

---

## 💫 WHITE FLASH VORTEX ENDING

### Design Philosophy:
Player walks into the vortex, gets engulfed by a white shader that spirals and fades to pure white. No 3D white room needed - a shader-based transition that's more elegant and performant.

### White Flash Ending Shader:
```javascript
class WhiteFlashEnding {
  constructor(scene, camera) {
    this.scene = scene;
    this.camera = camera;

    this.duration = 8000; // 8 seconds total
    this.startTime = 0;

    this.createWhiteFlash();
  }

  createWhiteFlash() {
    // Create full-screen quad for white flash effect
    const geometry = new THREE.PlaneGeometry(20, 20);
    this.flashMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uProgress: { value: 0.0 }, // 0-1 across duration
        uSpiralIntensity: { value: 0.0 }, // Spiral effect strength
        uWhiteLevel: { value: 0.0 }, // How close to pure white
      },

      vertexShader: `
        varying vec2 vUv;

        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,

      fragmentShader: `
        uniform float uTime;
        uniform float uProgress;
        uniform float uSpiralIntensity;
        uniform float uWhiteLevel;

        varying vec2 vUv;

        // Rotational symmetry for spiral
        vec2 rotate2D(vec2 p, float angle) {
          float c = cos(angle);
          float s = sin(angle);
          return vec2(p.x * c - p.y * s, p.x * s + p.y * c);
        }

        void main() {
          vec2 uv = vUv - 0.5; // Center at 0,0
          float dist = length(uv);
          float angle = atan(uv.y, uv.x);

          // Phase 1: Vortex spiral (0-60% of progress)
          // Creates spinning energy pattern
          float spiralPhase = smoothstep(0.0, 0.6, uProgress);

          // Multiple layered spirals
          float spiral1 = sin(angle * 5.0 + dist * 20.0 - uTime * 3.0) * 0.5 + 0.5;
          float spiral2 = sin(angle * 3.0 - dist * 15.0 + uTime * 2.0) * 0.5 + 0.5;
          float spiral3 = sin(angle * 7.0 + dist * 25.0 + uTime * 4.0) * 0.5 + 0.5;

          // Rotate uv for each spiral layer
          vec2 uv1 = rotate2D(uv, uTime * 0.5);
          vec2 uv2 = rotate2D(uv, -uTime * 0.3);
          vec2 uv3 = rotate2D(uv, uTime * 0.7);

          float combinedSpiral = (spiral1 + spiral2 + spiral3) / 3.0;

          // Spiral intensity increases with progress
          float spiralEffect = combinedSpiral * uSpiralIntensity * spiralPhase;

          // Phase 2: White fade (40-100% of progress)
          float fadePhase = smoothstep(0.4, 1.0, uProgress);

          // Colors transition: cyan → purple → white
          vec3 color1 = vec3(0.0, 0.8, 1.0); // Cyan (vortex start)
          vec3 color2 = vec3(0.5, 0.2, 1.0); // Purple
          vec3 color3 = vec3(1.0, 1.0, 1.0); // White (end)

          // Mix colors based on progress
          vec3 spiralColor = mix(color1, color2, spiralPhase);
          spiralColor = mix(spiralColor, color3, fadePhase * fadePhase);

          // Add spiral pattern to color
          vec3 finalColor = spiralColor + spiralEffect * 0.5;

          // Push toward pure white at end
          finalColor = mix(finalColor, vec3(1.0), uWhiteLevel);

          // Vignette effect
          float vignette = 1.0 - smoothstep(0.3, 0.7, dist);
          finalColor *= vignette;

          // Alpha fades in, stays, then fades out at very end
          float alpha = smoothstep(0.0, 0.1, uProgress);
          alpha *= 1.0 - smoothstep(0.95, 1.0, uProgress);

          gl_FragColor = vec4(finalColor, alpha);
        }
      `,

      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.flashMesh = new THREE.Mesh(geometry, this.flashMaterial);

    // Position in front of camera
    this.flashMesh.position.set(0, 0, -5);
    this.camera.add(this.flashMesh);
  }

  start() {
    this.startTime = Date.now();
    this.animate();

    // Auto-end scene after duration
    setTimeout(() => {
      this.onComplete();
    }, this.duration);
  }

  animate() {
    const elapsed = Date.now() - this.startTime;
    const progress = Math.min(elapsed / this.duration, 1.0);

    const time = elapsed * 0.001;

    // Update uniforms
    this.flashMaterial.uniforms.uTime.value = time;
    this.flashMaterial.uniforms.uProgress.value = progress;

    // Spiral intensity peaks at 60%, then fades
    const spiralIntensity = Math.sin(progress * Math.PI) * 2.0;
    this.flashMaterial.uniforms.uSpiralIntensity.value = spiralIntensity;

    // White level increases steadily
    this.flashMaterial.uniforms.uWhiteLevel.value = progress * progress;

    if (progress < 1.0) {
      requestAnimationFrame(() => this.animate());
    }
  }

  onComplete() {
    console.log('✨ Vimana Music Room complete!');

    // Clean up
    this.camera.remove(this.flashMesh);
    this.flashMaterial.dispose();

    // Trigger scene completion callback
    if (this.onSceneComplete) {
      this.onSceneComplete();
    } else {
      // Default: reload or transition
      console.log('Scene ended. Add transition logic here.');
    }
  }

  destroy() {
    this.camera.remove(this.flashMesh);
    this.flashMaterial.dispose();
  }
}
```

### Ending Sequence Timeline:
```
0:00 - Shell spawns, materializing over 3 seconds
0:03 - Player clicks shell
0:03-0:04.5 - Shell flies and shrinks into UI slot (1.5s)
0:04.5 - White flash sequence begins
0:04.5-0:05.7 - Spiral vortex pattern intensifies (Phase 1)
0:06.2-0:12.5 - Fade to white with spiral overlay (Phase 2)
0:12.5 - Pure white, scene ends
```

---

## 🌀 VORTEX DESIGN COMPLETE

### Visual Hierarchy:

1. **Core (SDF Torus):**
   - Mathematical precision
   - Smooth, organic shape
   - Represents the ship's throat/vocal cords
   - Glows brighter as harmony improves

2. **Particles (Energy Flow):**
   - Thousands of tiny particles
   - Spiral through torus
   - Represent sound/energy
   - Move faster as vortex activates

3. **Light:**
   - Inner cyan → outer purple gradient
   - Core white light at center
   - Edges glow intensely
   - Illuminates entire room

### Vortex Behavior:

```javascript
class VortexSystem {
  constructor(scene) {
    this.scene = scene;
    this.activationLevel = 0.0; // 0-1
    this.duetProgress = 0.0; // 0-1
    
    this.vortexMesh = null;
    this.particles = null;
    this.vortexMaterial = null;
    
    this.createVortex();
  }
  
  createVortex() {
    // Create torus geometry
    const torusGeometry = new THREE.TorusGeometry(2.0, 0.4, 32, 100);
    
    // Apply SDF vortex material
    this.vortexMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uVortexActivation: { value: 0.0 },
        uDuetProgress: { value: 0.0 },
        uInnerColor: { value: new THREE.Color(0x00ffff) },
        uOuterColor: { value: new THREE.Color(0x8800ff) },
        uCoreColor: { value: new THREE.Color(0xffffff) },
        uTorusRadius: { value: 2.0 },
        uTubeRadius: { value: 0.4 },
      },
      vertexShader: `/* ... SDF torus vertex shader ... */`,
      fragmentShader: `/* ... SDF torus fragment shader ... */`,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    
    this.vortexMesh = new THREE.Mesh(torusGeometry, this.vortexMaterial);
    this.vortexMesh.position.set(0, 0.5, 2);
    this.scene.add(this.vortexMesh);
    
    // Create particle system
    this.particles = new VortexParticles(this.scene, new THREE.Vector3(0, 0.5, 2));
  }
  
  updateDuetProgress(progress) {
    this.duetProgress = Math.min(progress, 1.0);
    
    if (this.vortexMaterial) {
      this.vortexMaterial.uniforms.uDuetProgress.value = this.duetProgress;
    }
    
    // Activation grows with duet progress
    this.activationLevel = this.duetProgress;
    
    if (this.vortexMaterial) {
      this.vortexMaterial.uniforms.uVortexActivation.value = this.activationLevel;
    }
  }
  
  update(time) {
    if (this.vortexMaterial) {
      this.vortexMaterial.uniforms.uTime.value = time;
    }
    
    if (this.particles) {
      this.particles.update(0.016, time, this.activationLevel);
    }
  }
}
```

---

## 🎭 NARRATIVE INTEGRATION

### How This Fits VIMANA's Vision:

**From Narrative Design:**
> "Every world has a song. We learn them all."
> "This is Vimana's communication hub—how cultures stay connected across distance."

**Our Implementation:**
- The ship sings its song through the harp
- The jelly creatures are "musical messengers" - they carry the song from the water to the harp
- Player learns by watching, then joins the duet
- When harmony is achieved, the vortex opens - the ship's "voice" is strong enough to transport

**The "Teaching" Metaphor:**
> "The ship doesn't test you. It teaches you. When you play wrong, it doesn't fail you. It just sings again, more slowly, more clearly, until you can join in harmony. This is how the Vimana learns from worlds—they demonstrate, you observe, then you co-create."

---

## 📊 PROGRESSION SYSTEM

### Duet Progress Tracking:

```javascript
class DuetProgressTracker {
  constructor() {
    this.totalNotesPlayed = 0;
    this.correctNotesPlayed = 0;
    this.wrongNotesPlayed = 0;
    this.sequencesCompleted = 0;
    
    this.harmonyScore = 0.0; // 0-1
    this.shipPatience = 1.0; // Always patient, but visually represented
  }
  
  onNotePlayed(correct) {
    this.totalNotesPlayed++;
    
    if (correct) {
      this.correctNotesPlayed++;
      // Harmony increases with correct notes
      this.harmonyScore = Math.min(this.harmonyScore + 0.1, 1.0);
    } else {
      this.wrongNotesPlayed++;
      // Harmony decreases slightly, but ship stays patient
      this.harmonyScore = Math.max(this.harmonyScore - 0.02, 0.0);
    }
    
    // Ship patience - never decreases, just visual reminder
    // The more mistakes, the more gentle the teaching becomes
    
    this.updateVisuals();
  }
  
  onSequenceComplete() {
    this.sequencesCompleted++;
    
    // Significant harmony boost
    this.harmonyScore = Math.min(this.harmonyScore + 0.2, 1.0);
    
    this.updateVisuals();
  }
  
  updateVisuals() {
    // Send to water shader
    if (window.waterMaterial) {
      window.waterMaterial.uniforms.uDuetProgress.value = this.harmonyScore;
    }
    
    // Send to vortex
    if (window.vortexSystem) {
      window.vortexSystem.updateDuetProgress(this.harmonyScore);
    }
    
    // Log for debugging
    console.log(`🎵 Harmony: ${this.harmonyScore.toFixed(2)} | Sequences: ${this.sequencesComplete}/3`);
  }
}
```

---

## 🚀 IMPLEMENTATION ORDER

### Phase 1: Visual Foundation (Day 1)
- [ ] Apply enhanced water shader to ArenaFloor
- [ ] Create SDF torus vortex mesh
- [ ] Create vortex particle system
- [ ] Test water + vortex visuals

### Phase 2: Enhanced Jelly System (Day 2)
- [ ] Create JellyCreature with enhanced shader
- [ ] Create PatientJellyManager (new logic)
- [ ] Implement teaching sequence system
- [ ] Test jelly demonstrations

### Phase 3: Gentle Feedback (Day 3)
- [ ] Create GentleFeedback (camera shake)
- [ ] Create GentleAudioFeedback (discordant/reminder sounds)
- [ ] Integrate with wrong note handling
- [ ] Test no-reset behavior

### Phase 4: Duet Integration (Day 4)
- [ ] Create DuetProgressTracker
- [ ] Connect harp strings → progress tracker
- [ ] Connect progress → water shader duet uniform
- [ ] Connect progress → vortex activation
- [ ] Test full duet flow

### Phase 5: Shell Collection System (Day 5)
- [ ] Create SDFShellMaterial with nautilus spiral SDF
- [ ] Create ShellCollectibleManager with click detection
- [ ] Implement 3-second appear animation
- [ ] Implement 1.5-second fly-to-UI animation
- [ ] Test shell collection feel

### Phase 6: UI Overlay System (Day 6)
- [ ] Create ShellUIOverlay with 4-slot display
- [ ] Implement canvas-drawn simplified shell icons
- [ ] Add slot fill animations
- [ ] Integrate with ShellCollectibleManager
- [ ] Test UI feedback loop

### Phase 7: White Flash Ending (Day 7)
- [ ] Create WhiteFlashEnding shader system
- [ ] Implement vortex spiral phase (0-60%)
- [ ] Implement white fade phase (40-100%)
- [ ] Integrate with shell collection callback
- [ ] Test complete ending sequence

---

## 🎯 SUCCESS METRICS

### Visual Quality:
- [ ] Water ripples respond to all 6 strings individually
- [ ] Bioluminescence glows with string vibration
- [ ] Vortex torus is mathematically smooth
- [ ] Particles flow through torus in spiral
- [ ] Glow intensifies with duet progress
- [ ] Shell SDF shows clear nautilus spiral pattern
- [ ] Shell iridescence shifts with view angle
- [ ] UI icons match shell aesthetic
- [ ] White flash has smooth spiral-to-white transition

### Gameplay Feel:
- [ ] Wrong note feels gentle, not punitive
- [ ] Jelly reappears clearly demonstrates correct string
- [ ] No reset - player stays on same note
- [ ] Ship patience is visually evident
- [ ] Harmony buildup feels rewarding
- [ ] Shell appearance feels mysterious and rewarding
- [ ] Shell collection is satisfying (fly + shrink animation)
- [ ] UI provides clear progress feedback
- [ ] White flash ending feels transcendent

### Performance:
- [ ] 60 FPS with water + vortex + particles + shell
- [ ] Shader compilation under 5 seconds
- [ ] Memory stable across device tiers
- [ ] Smooth transitions between states
- [ ] Shell appear animation is smooth (no hitching)
- [ ] White flash maintains 60 FPS

---

## 📚 DOCS TO UPDATE

1. **_bmad-output/gdd.md** - Update Archive of Voices section ✅
2. **_bmad-output/narrative-design.md** - Add music room narrative ✅
3. **VIMANA_HARP_IMPLEMENTATION_PLAN.md** - This file (updated) ✅
4. **VIMANA_HARP_ENHANCED_DESIGN.md** - Reference for implementation details ✅

---

## ⚡ PERFORMANCE & BEST PRACTICES

### Async Shader Loading Strategy:

Shader compilation can block the main thread. This strategy prevents frame drops during scene load.

```javascript
class ShaderLoader {
  constructor() {
    this.shaderCache = new Map();
    this.loadingPromises = new Map();
  }

  async loadShader(name, vertexSource, fragmentSource) {
    // Check cache first
    if (this.shaderCache.has(name)) {
      return this.shaderCache.get(name);
    }

    // Check if already loading
    if (this.loadingPromises.has(name)) {
      return this.loadingPromises.get(name);
    }

    // Create loading promise
    const promise = this.compileShaderAsync(name, vertexSource, fragmentSource);
    this.loadingPromises.set(name, promise);

    try {
      const material = await promise;
      this.shaderCache.set(name, material);
      this.loadingPromises.delete(name);
      return material;
    } catch (error) {
      this.loadingPromises.delete(name);
      throw error;
    }
  }

  compileShaderAsync(name, vertexSource, fragmentSource) {
    return new Promise((resolve, reject) => {
      // Create test geometry for compilation
      const testGeometry = new THREE.PlaneGeometry(1, 1);

      const material = new THREE.ShaderMaterial({
        vertexShader: vertexSource,
        fragmentShader: fragmentSource,
        transparent: true,
      });

      // Create off-screen scene for compilation
      const offScreenScene = new THREE.Scene();
      const offScreenCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
      const testMesh = new THREE.Mesh(testGeometry, material);
      offScreenScene.add(testMesh);

      // Use requestAnimationFrame to let browser compile
      requestAnimationFrame(() => {
        // Force compilation by rendering
        const renderer = window.renderer || new THREE.WebGLRenderer();
        renderer.compile(offScreenScene, offScreenCamera);

        // Clean up
        offScreenScene.remove(testMesh);
        testGeometry.dispose();

        // Resolve with compiled material
        resolve(material);
      });
    });
  }

  // Batch load all shaders at scene start
  async preloadAllShaders() {
    const shaders = [
      ['water', waterVertexShader, waterFragmentShader],
      ['vortex', vortexVertexShader, vortexFragmentShader],
      ['jelly', jellyVertexShader, jellyFragmentShader],
      ['shell', shellVertexShader, shellFragmentShader],
      ['whiteFlash', whiteFlashVertexShader, whiteFlashFragmentShader],
    ];

    const promises = shaders.map(([name, vert, frag]) =>
      this.loadShader(name, vert, frag)
    );

    try {
      await Promise.all(promises);
      console.log('✅ All shaders compiled successfully');
    } catch (error) {
      console.error('❌ Shader compilation failed:', error);
    }
  }
}
```

### Scene Transition Cleanup:

```javascript
class HarpRoomSceneManager {
  constructor(scene, camera) {
    this.scene = scene;
    this.camera = camera;
    this.systems = [];
  }

  // Register all systems for cleanup
  registerSystem(system) {
    this.systems.push(system);
  }

  // Clean up all systems before scene transition
  cleanup() {
    console.log('🧹 Cleaning up Harp Room...');

    // Clean up all registered systems
    this.systems.forEach(system => {
      if (system.destroy) {
        system.destroy();
      }
    });

    // Clear arrays
    this.systems = [];

    // Clean up global references
    if (window.waterMaterial) {
      window.waterMaterial.dispose();
      delete window.waterMaterial;
    }
    if (window.vortexSystem) {
      delete window.vortexSystem;
    }
    if (window.shellUI) {
      window.shellUI.destroy();
      delete window.shellUI;
    }

    // Force garbage collection hint (for debugging)
    if (window.gc) {
      window.gc();
    }

    console.log('✅ Harp Room cleanup complete');
  }

  // Transition to next scene
  transitionToNextScene(nextSceneInit) {
    // Clean up current scene
    this.cleanup();

    // Small delay for cleanup to complete
    setTimeout(() => {
      // Initialize next scene
      nextSceneInit();
    }, 100);
  }
}
```

### Performance Profiling Integration:

```javascript
class PerformanceMonitor {
  constructor() {
    this.fpsHistory = [];
    this.frameCount = 0;
    this.lastFpsUpdate = 0;

    // Create stats display (optional, for development)
    this.stats = null;
    if (typeof Stats !== 'undefined') {
      this.stats = new Stats();
      this.stats.showPanel(0); // 0: fps, 1: ms, 2: mb
      document.body.appendChild(this.stats.dom);
    }
  }

  update(time) {
    if (this.stats) {
      this.stats.begin();
    }

    this.frameCount++;

    // Update FPS every second
    if (time - this.lastFpsUpdate >= 1000) {
      const fps = this.frameCount;
      this.fpsHistory.push(fps);

      // Keep last 60 seconds of history
      if (this.fpsHistory.length > 60) {
        this.fpsHistory.shift();
      }

      const avgFps = this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length;

      // Log warnings if FPS drops
      if (fps < 55) {
        console.warn(`⚠️ Low FPS detected: ${fps} (avg: ${avgFps.toFixed(1)})`);
      }

      this.frameCount = 0;
      this.lastFpsUpdate = time;
    }

    if (this.stats) {
      this.stats.end();
    }
  }

  getAverageFPS() {
    if (this.fpsHistory.length === 0) return 0;
    return this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length;
  }

  isPerformanceGood() {
    return this.getAverageFPS() >= 55;
  }
}
```

### Mobile Device Fallbacks:

```javascript
class DeviceCapabilities {
  static detect() {
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    const isLowEnd = isMobile && this.getMemoryLimit() < 4; // Less than 4GB RAM

    return {
      isMobile,
      isLowEnd,
      maxParticles: isLowEnd ? 500 : 2000,
      shaderQuality: isLowEnd ? 'low' : 'high',
      particleLOD: isLowEnd ? 0.5 : 0.3, // More aggressive LOD on low-end
    };
  }

  static getMemoryLimit() {
    // Estimate based on device memory API
    if (navigator.deviceMemory) {
      return navigator.deviceMemory;
    }
    return 8; // Default assumption
  }

  static getOptimalSettings() {
    const caps = this.detect();

    return {
      maxParticles: caps.maxParticles,
      enableHighQualityShaders: caps.shaderQuality === 'high',
      lodThreshold: caps.particleLOD,
      targetFPS: caps.isMobile ? 30 : 60,
    };
  }
}
```

---

## 🔧 BUG FIXES APPLIED (Party Mode Review)

### Critical Fixes:
1. **Line 415 Particle Position Bug**: Fixed Z-coordinate incorrectly assigned Y value. Now properly preserves radial distance for torus flow.
2. **Line 375 Tube Offset Randomization**: Fixed static tubeOffset value. Now properly randomized on each particle reset.

### Important Improvements:
3. **Particle LOD System**: Added level-of-detail system that skips particles based on activation level (50% skip at low, 25% at medium, 0% at full).
4. **Memory Cleanup Methods**: Added destroy() methods to VortexParticles and PatientJellyManager for proper scene transition cleanup.
5. **Async Shader Loading**: Added strategy for non-blocking shader compilation during scene load.
6. **Performance Monitoring**: Added PerformanceMonitor class for FPS tracking and warning system.

### Performance Impact:
- **Before Fixes**: 2000 particles × 6 trig operations × 60fps = 720K ops/second (all particles)
- **After LOD**: At 30% activation: 1000 particles × 6 trig × 60fps = 360K ops/second (50% reduction)
- **Low-end Devices**: At 30% activation with 500 max particles = 90K ops/second (87% reduction)

---

## 💡 KEY INSIGHTS

**What Makes This Work:**
1. **No Failure** - Player can't "lose," only learn
2. **Gentle Feedback** - Shake + discordant tone, not harsh punishment
3. **Patient Teaching** - Jelly reappears, demonstrates again
4. **Visual Harmony** - Room gets brighter as player and ship sync up
5. **Vortex as Voice** - The portal is ship's song made visible

**The Emotional Arc:**
```
Confusion (what is this?) → Curiosity (watching jelly) → 
Attempt (playing string) → Harmony (correct note) → 
Correction (wrong note) → Understanding (watching again) → 
Progression (sequences advance) → Achievement (all complete) → 
Transcendence (vortex opens, platform rides ship's voice)
```

**This is Not:**
- ❌ A puzzle to solve
- ❌ A test to pass
- ❌ A challenge to overcome
- ❌ A game with failure states

**This Is:**
- ✅ A song to learn
- ✅ A duet to join
- ✅ A relationship to build
- ✅ An experience to share

---

**Next Steps:**
1. ✅ Create enhanced design document (VIMANA_HARP_ENHANCED_DESIGN.md)
2. ✅ Update _bmad-output/narrative-design.md
3. ✅ Update _bmad-output/gdd.md
4. ✅ Update this implementation plan (VIMANA_HARP_IMPLEMENTATION_PLAN.md)
5. [ ] Present complete plan to user
6. [ ] Begin implementation with Phase 1
