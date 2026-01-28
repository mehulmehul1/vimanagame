# VIMANA Harp Room - ENHANCED DESIGN
## "The Ship Teaches You Its Song" - A Duet, Not a Test

**Date:** 2026-01-24
**Context:** Prototyping the Music Room (Archive of Voices - Culture Chamber)

---

## 🎯 CORE PHILOSOPHY SHIFT

### Previous Approach (Test-Based)
```
❌ Jelly teaches pattern → Player must repeat correctly
❌ Wrong input = FAILURE → RESET sequence
❌ Ship tests player competence
❌ Player passes/fails puzzle
```

### New Approach (Duet-Based)
```
✅ Jelly demonstrates melody → Player joins in harmony
✅ Wrong note = GENTLE NUDGE → Ship patiently waits
✅ Ship teaches through presence, not testing
✅ Player and ship co-create music together
```

**The Metaphor:** 
> "You're not being tested on the ship's song. The ship is singing to you, and you're invited to harmonize. It's a duet—a shared moment of music-making. When you play wrong, the ship doesn't fail you. It just sings again, more slowly, more clearly, until you can join in harmony."

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
  }
  
  resetParticle(i) {
    // Spawn particles in a torus pattern
    const angle = Math.random() * Math.PI * 2;
    const radius = 2.0 + (Math.random() - 0.5) * 0.5; // Around torus center
    const tubeOffset = (Math.random() - 0.5) * 0.8; // Around tube
    
    this.positions[i * 3] = Math.cos(angle) * radius + (Math.random() - 0.5) * 0.2;
    this.positions[i * 3 + 1] = tubeOffset;
    this.positions[i * 3 + 2] = Math.sin(angle) * radius + (Math.random() - 0.5) * 0.2;
    
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
      
      this.positions[i * 3] = Math.cos(newAngle) * dist;
      this.positions[i * 3 + 2] = Math.sin(newAngle) * dist;
      
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
    gain.gain.setValueAtTime(0.0, this.audioContext.currentTime);
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
    
    gain.gain.setValueAtTime(0.0, this.audioContext.currentTime);
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
    
    playerGain.gain.exponentialRampToValueAtTime(0.001, this.audioContext.currentTime + 2.0);
    harmonyGain.gain.exponentialRampToValueAtTime(0.001, this.audioContext.currentTime + 2.0);
    
    playerOsc.connect(playerGain);
    harmonyOsc.connect(harmonyGain);
    playerGain.connect(this.audioContext.destination);
    harmonyGain.connect(this.audioContext.destination);
    
    playerOsc.start();
    harmonyOsc.start();
    playerOsc.stop(this.audioContext.currentTime + 2.1);
    harmonyOsc.stop(this.audioContext.currentTime + 2.1);
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
}
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

## 🐚 SHELL COLLECTION SYSTEM

### Design Philosophy:
The shell is a procedural SDF-based collectible that serves as both reward and progress tracker. Each room grants one shell; collecting all 4 activates the Vimana.

### Visual Design:
- **Nautilus Spiral SDF:** Mathematical formula creates organic shell shape
- **Iridescence:** Color shifts based on view angle (warm pearl → cyan shimmer)
- **3-Second Appear:** Shell materializes slowly from nothing
- **Click to Collect:** Raycast detection for satisfying interaction
- **Fly-to-UI Animation:** 1.5 second lerp + shrink + dissolve

### Shell States:
```
APPEARING (3s)
  ├─ Vertices emerge from center
  ├─ Shader opacity fades in
  └─ Gentle rotation

IDLE (indefinite)
  ├─ Slow rotation
  ├─ Gentle bobbing
  └─ Waiting for player click

COLLECTING (1.5s)
  ├─ Fly toward UI position
  ├─ Scale shrinks to 0
  ├─ Dissolve noise fades alpha
  └─ Spin accelerates

COLLECTED
  └─ Removed from scene, UI slot filled
```

### UI Overlay:
- **4 Circular Slots** arranged horizontally (top-left screen position)
- **Empty Slot:** Dark semi-transparent with faint border
- **Filled Slot:** Glowing border + canvas-drawn shell icon
- **Icon Design:** Simplified 2D nautilus spiral (matches 3D aesthetic)
- **Animation:** Gentle pulse on filled slots

### Technical Notes:
```javascript
// Integration points
- ShellCollectibleManager.onCollected → ShellUIOverlay.addShell()
- PatientJellyManager.onAllDuetSequencesComplete → spawn shell
- ShellCollectibleManager.onCollected → WhiteFlashEnding.start()
```

---

## 💫 WHITE FLASH VORTEX ENDING

### NEW Ending Flow (Simplified):

**Previous (Complex):**
```
Platform rides to vortex → Enter white room → See shell → Collect → Exit door → Next room
```

**New (Elegant):**
```
Platform rides to vortex → Shell appears → Collect → White flash engulf → Scene ends
```

### White Flash Sequence:

**Phase 1: Vortex Spiral (0-60% of 8s)**
- Multi-layered spiral patterns
- Cyan → Purple color shift
- Spinning energy intensity
- Creates "entering the ship's voice" feeling

**Phase 2: White Fade (40-100% of 8s)**
- Spiral overlay persists
- Gradual shift to pure white
- Vignette effect focuses attention
- Creates transcendence feeling

**Final Moment:**
- Pure white screen
- Scene fades out
- Triggers completion callback

### Technical Implementation:
- Full-screen quad with custom shader
- Parented to camera (follows player)
- Additive blending for glow effect
- 8-second total duration
- Auto-triggers completion

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
3. **VIMANA_HARP_IMPLEMENTATION_PLAN.md** - Complete implementation plan ✅
4. **VIMANA_HARP_ENHANCED_DESIGN.md** - This file (updated) ✅

---

## 💡 KEY INSIGHTS

**What Makes This Work:**
1. **No Failure** - Player can't "lose," only learn
2. **Gentle Feedback** - Shake + discordant tone, not harsh punishment
3. **Patient Teaching** - Jelly reappears, demonstrates again
4. **Visual Harmony** - Room gets brighter as player and ship sync up
5. **Vortex as Voice** - The portal is the ship's song made visible
6. **Shell as Reward** - Procedural SDF object feels precious and mysterious
7. **UI as Progress** - Four slots give clear sense of journey scope
8. **White Flash Ending** - Elegant shader transition beats complex 3D room

**The Emotional Arc:**
```
Confusion (what is this?) → Curiosity (watching jelly) →
Attempt (playing string) → Gentle Correction (wrong note) →
Understanding (watching again) → Harmony (correct note) →
Progress (sequence advances) → Achievement (all sequences complete) →
Reward (shell appears) → Collection (satisfying fly-to-UI) →
Transcendence (white flash engulf, scene ends)
```

**This is Not:**
- ❌ A puzzle to solve
- ❌ A test to pass
- ❌ A challenge to overcome
- ❌ A game with failure states
- ❌ A complex multi-room navigation

**This Is:**
- ✅ A song to learn
- ✅ A duet to join
- ✅ A relationship to build
- ✅ An experience to share
- ✅ A contained 5-10 minute journey
- ✅ One of four chambers in a larger experience

---

**Next Steps:**
1. ✅ Design approved - shell system, UI, and white flash ending integrated
2. ✅ Documentation updated (implementation plan + this enhanced design)
3. [ ] Begin implementation with Phase 1
4. [ ] Test and iterate based on feel
5. [ ] Polish visuals until they sing
