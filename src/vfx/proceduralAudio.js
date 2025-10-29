import { Logger } from "../utils/logger.js";

/**
 * ProceduralAudio - Modular Web Audio API synthesizer for VFX
 *
 * Creates dynamic procedural sounds that sync with visual effects.
 * Highly configurable for different effect types.
 */
export class ProceduralAudio {
  constructor(config = {}) {
    this.name = config.name || "ProceduralAudio";
    this.logger = new Logger(this.name, false);

    // Audio context and nodes
    this.audioContext = null;
    this.masterGain = null;

    // Oscillators
    this.baseOscillator = null;
    this.subOscillator = null;
    this.modulatorOscillator = null;
    this.sweepOscillator = null; // High-frequency sweep for "radio tuning" effect
    this.sweepLFO = null; // LFO to modulate sweep frequency

    // Effects nodes
    this.filter = null;
    this.distortion = null;
    this.delay = null;
    this.delayFeedback = null;
    this.compressor = null;
    this.panner = null;

    // LFO for subtle movement
    this.lfo = null;
    this.lfoGain = null;

    // State
    this.isPlaying = false;
    this.startTime = 0;
    this.targetVolume = 0; // Track target volume for ramping
    this.isFadingOut = false;
    this.sweepGain = null; // Gain node for sweep oscillator (if enabled)

    // Configuration (can be overridden)
    this.config = {
      baseFrequency: config.baseFrequency || 155.56, // Eb3
      volume: config.volume || 0.3,
      baseOscType: config.baseOscType || "sawtooth", // sawtooth, sine, square, triangle
      subOscType: config.subOscType || "sine",
      modOscType: config.modOscType || "sine",
      filterType: config.filterType || "lowpass",
      filterFreq: config.filterFreq || 2000,
      filterQ: config.filterQ || 5,
      distortionAmount: config.distortionAmount || 20,
      delayTime: config.delayTime || 0.15,
      delayFeedback: config.delayFeedback || 0.4,
      lfoFreq: config.lfoFreq || 0.3,
      lfoDepth: config.lfoDepth || 50,
      fadeInTime: config.fadeInTime || 0.08, // Fade in duration (seconds)
      fadeOutTime: config.fadeOutTime || 0.1, // Fade out duration (seconds)
      fadeInCurve: config.fadeInCurve || "linear", // 'linear' or 'exponential'
      fadeOutCurve: config.fadeOutCurve || "linear", // 'linear' or 'exponential'
      // Radio tuning sweep effect
      enableSweep: config.enableSweep || false,
      sweepBaseFreq: config.sweepBaseFreq || 3000, // High frequency base
      sweepRange: config.sweepRange || 1500, // Hz up and down
      sweepRate: config.sweepRate || 2.5, // Sweeps per second
      sweepGain: config.sweepGain || 0.15, // Volume of sweep (relative to main)
      ...config,
    };

    this.logger.log(`${this.name} created with config:`, this.config);
  }

  /**
   * Initialize the Web Audio API context and nodes
   */
  async initialize() {
    if (this.audioContext) {
      this.logger.log("Already initialized");
      return;
    }

    try {
      // Create audio context
      this.audioContext = new (window.AudioContext ||
        window.webkitAudioContext)();

      // Resume context if suspended (browser autoplay policy)
      if (this.audioContext.state === "suspended") {
        await this.audioContext.resume();
      }

      // Create the audio graph
      this._createAudioGraph();

      this.logger.log("Audio initialized", {
        sampleRate: this.audioContext.sampleRate,
        state: this.audioContext.state,
      });
    } catch (error) {
      this.logger.error("Failed to initialize audio:", error);
    }
  }

  /**
   * Create the audio processing graph
   * @private
   */
  _createAudioGraph() {
    const ctx = this.audioContext;

    // Master gain (volume control)
    this.masterGain = ctx.createGain();
    this.masterGain.gain.value = 0; // Start silent
    this.masterGain.connect(ctx.destination);

    // Compressor to smooth out dynamics
    this.compressor = ctx.createDynamicsCompressor();
    this.compressor.threshold.value = -20;
    this.compressor.knee.value = 10;
    this.compressor.ratio.value = 4;
    this.compressor.attack.value = 0.003;
    this.compressor.release.value = 0.25;
    this.compressor.connect(this.masterGain);

    // Stereo panner for spatial movement
    this.panner = ctx.createStereoPanner();
    this.panner.pan.value = 0;
    this.panner.connect(this.compressor);

    // Delay with feedback
    this.delay = ctx.createDelay(1.0);
    this.delay.delayTime.value = this.config.delayTime;
    this.delayFeedback = ctx.createGain();
    this.delayFeedback.gain.value = this.config.delayFeedback;
    this.delay.connect(this.delayFeedback);
    this.delayFeedback.connect(this.delay);
    this.delay.connect(this.panner);

    // Distortion (waveshaper)
    this.distortion = ctx.createWaveShaper();
    this.distortion.curve = this._makeDistortionCurve(
      this.config.distortionAmount
    );
    this.distortion.oversample = "2x";
    this.distortion.connect(this.delay);
    this.distortion.connect(this.panner); // Also send dry signal

    // Filter for frequency sweeps
    this.filter = ctx.createBiquadFilter();
    this.filter.type = this.config.filterType;
    this.filter.frequency.value = this.config.filterFreq;
    this.filter.Q.value = this.config.filterQ;
    this.filter.connect(this.distortion);

    // LFO (Low Frequency Oscillator) for subtle modulation
    this.lfo = ctx.createOscillator();
    this.lfo.frequency.value = this.config.lfoFreq;
    this.lfo.type = "sine";
    this.lfoGain = ctx.createGain();
    this.lfoGain.gain.value = this.config.lfoDepth;
    this.lfo.connect(this.lfoGain);
    this.lfo.start();

    this.logger.log("Audio graph created");
  }

  /**
   * Create distortion curve for waveshaper
   * @private
   */
  _makeDistortionCurve(amount) {
    const samples = 44100;
    const curve = new Float32Array(samples);
    const deg = Math.PI / 180;

    for (let i = 0; i < samples; i++) {
      const x = (i * 2) / samples - 1;
      curve[i] =
        ((3 + amount) * x * 20 * deg) / (Math.PI + amount * Math.abs(x));
    }

    return curve;
  }

  /**
   * Start the sound
   */
  start() {
    if (!this.audioContext) {
      this.logger.warn("Cannot start - audio not initialized");
      return;
    }

    if (this.isPlaying) {
      this.stop(); // Stop existing oscillators first
    }

    const ctx = this.audioContext;
    const now = ctx.currentTime;
    this.startTime = now;

    // Create oscillators
    // Base oscillator - main tone
    this.baseOscillator = ctx.createOscillator();
    this.baseOscillator.type = this.config.baseOscType;
    this.baseOscillator.frequency.value = this.config.baseFrequency;

    const baseGain = ctx.createGain();
    baseGain.gain.value = 0.4;
    this.baseOscillator.connect(baseGain);
    baseGain.connect(this.filter);

    // Sub oscillator - one octave down for depth
    this.subOscillator = ctx.createOscillator();
    this.subOscillator.type = this.config.subOscType;
    this.subOscillator.frequency.value = this.config.baseFrequency * 0.5;

    const subGain = ctx.createGain();
    subGain.gain.value = 0.3;
    this.subOscillator.connect(subGain);
    subGain.connect(this.filter);

    // Modulator oscillator - FM synthesis for complexity
    this.modulatorOscillator = ctx.createOscillator();
    this.modulatorOscillator.type = this.config.modOscType;
    this.modulatorOscillator.frequency.value = this.config.baseFrequency * 2;

    const modGain = ctx.createGain();
    modGain.gain.value = 100; // FM modulation amount
    this.modulatorOscillator.connect(modGain);
    modGain.connect(this.baseOscillator.frequency); // Modulate base frequency

    // Connect LFO to filter frequency for subtle movement
    this.lfoGain.connect(this.filter.frequency);

    // Create sweep oscillator if enabled (high-frequency radio tuning effect)
    if (this.config.enableSweep) {
      this.sweepOscillator = ctx.createOscillator();
      this.sweepOscillator.type = "sine"; // Pure tone for sweep
      this.sweepOscillator.frequency.value = this.config.sweepBaseFreq;

      // Create LFO to modulate sweep frequency (creates the scanning effect)
      this.sweepLFO = ctx.createOscillator();
      this.sweepLFO.type = "sine";
      this.sweepLFO.frequency.value = this.config.sweepRate;

      const sweepLFOGain = ctx.createGain();
      sweepLFOGain.gain.value = this.config.sweepRange; // Modulation depth
      this.sweepLFO.connect(sweepLFOGain);
      sweepLFOGain.connect(this.sweepOscillator.frequency); // Modulate sweep freq

      const sweepGain = ctx.createGain();
      sweepGain.gain.value = 0; // Start silent, will be controlled dynamically
      this.sweepGain = sweepGain; // Store reference for dynamic control
      this.sweepOscillator.connect(sweepGain);
      sweepGain.connect(this.filter); // Send through filter chain

      this.sweepLFO.start(now);
      this.sweepOscillator.start(now);
    }

    // Start oscillators
    this.baseOscillator.start(now);
    this.subOscillator.start(now);
    this.modulatorOscillator.start(now);

    // Always fade in from 0 to avoid clicks/pops
    const fadeInTime = this.config.fadeInTime;
    this.masterGain.gain.cancelScheduledValues(now);

    if (this.config.fadeInCurve === "exponential") {
      // Exponential fade in: start near 0 and exponentially ramp up
      const epsilon = 0.001; // Can't start at exactly 0 for exponential
      this.masterGain.gain.setValueAtTime(epsilon, now);
      this.masterGain.gain.exponentialRampToValueAtTime(
        this.config.volume,
        now + fadeInTime
      );
      this.logger.log(`Started sound - exponential fade in (${fadeInTime}s)`);
    } else {
      // Linear fade in
      this.masterGain.gain.setValueAtTime(0, now);
      this.masterGain.gain.linearRampToValueAtTime(
        this.config.volume,
        now + fadeInTime
      );
      this.logger.log(`Started sound - linear fade in (${fadeInTime}s)`);
    }

    this.targetVolume = this.config.volume;
    this.isPlaying = true;
    this.isFadingOut = false;
  }

  /**
   * Stop the sound
   */
  stop() {
    if (!this.isPlaying || !this.audioContext) return;

    const ctx = this.audioContext;
    const now = ctx.currentTime;
    const fadeOutTime = this.config.fadeOutTime;

    // Mark as fading out to prevent updateParams from changing volume
    this.isFadingOut = true;

    // Always fade out to 0 to avoid clicks/pops
    this.masterGain.gain.cancelScheduledValues(now);
    this.masterGain.gain.setValueAtTime(this.masterGain.gain.value, now);

    if (this.config.fadeOutCurve === "exponential") {
      // Exponential fade out: exponentially approach near-zero, then linear to 0
      const epsilon = 0.001;
      const expTime = fadeOutTime * 0.9; // 90% exponential
      const linTime = fadeOutTime * 0.1; // 10% linear to finish
      this.masterGain.gain.exponentialRampToValueAtTime(epsilon, now + expTime);
      this.masterGain.gain.linearRampToValueAtTime(0, now + fadeOutTime);
      this.logger.log(`Stopped sound - exponential fade out (${fadeOutTime}s)`);
    } else {
      // Linear fade out
      this.masterGain.gain.linearRampToValueAtTime(0, now + fadeOutTime);
      this.logger.log(`Stopped sound - linear fade out (${fadeOutTime}s)`);
    }

    // Stop oscillators after fade
    if (this.baseOscillator) {
      this.baseOscillator.stop(now + fadeOutTime);
      this.baseOscillator = null;
    }
    if (this.subOscillator) {
      this.subOscillator.stop(now + fadeOutTime);
      this.subOscillator = null;
    }
    if (this.modulatorOscillator) {
      this.modulatorOscillator.stop(now + fadeOutTime);
      this.modulatorOscillator = null;
    }
    if (this.sweepOscillator) {
      this.sweepOscillator.stop(now + fadeOutTime);
      this.sweepOscillator = null;
    }
    if (this.sweepLFO) {
      this.sweepLFO.stop(now + fadeOutTime);
      this.sweepLFO = null;
    }

    this.sweepGain = null;
    this.targetVolume = 0;
    this.isPlaying = false;
  }

  /**
   * Update audio parameters smoothly
   * @param {Object} params - Parameters to update
   * @param {number} params.filterFreq - Target filter frequency
   * @param {number} params.filterQ - Target filter Q (resonance)
   * @param {number} params.delayFeedback - Target delay feedback
   * @param {number} params.pan - Target stereo pan (-1 to 1)
   * @param {number} params.volume - Target volume (0 to 1)
   * @param {number} params.pitchMultiplier - Base frequency multiplier
   * @param {number} params.sweepAmount - Sweep oscillator volume (0 to 1, if enabled)
   * @param {number} params.transitionTime - Time to reach target (seconds)
   */
  updateParams(params) {
    if (!this.isPlaying || !this.audioContext) return;

    const ctx = this.audioContext;
    const now = ctx.currentTime;
    const transitionTime = params.transitionTime || 0.05;

    // Check if we're in the initial fade-in period
    const timeSinceStart = now - this.startTime;
    const isFadingIn = timeSinceStart < this.config.fadeInTime;

    // Filter frequency
    if (params.filterFreq !== undefined) {
      this.filter.frequency.cancelScheduledValues(now);
      this.filter.frequency.setValueAtTime(this.filter.frequency.value, now);
      this.filter.frequency.linearRampToValueAtTime(
        params.filterFreq,
        now + transitionTime
      );
    }

    // Filter Q
    if (params.filterQ !== undefined) {
      this.filter.Q.cancelScheduledValues(now);
      this.filter.Q.setValueAtTime(this.filter.Q.value, now);
      this.filter.Q.linearRampToValueAtTime(
        params.filterQ,
        now + transitionTime
      );
    }

    // Pitch (base frequency)
    if (params.pitchMultiplier !== undefined && this.baseOscillator) {
      const targetFreq = this.config.baseFrequency * params.pitchMultiplier;
      this.baseOscillator.frequency.cancelScheduledValues(now);
      this.baseOscillator.frequency.setValueAtTime(
        this.baseOscillator.frequency.value,
        now
      );
      this.baseOscillator.frequency.linearRampToValueAtTime(
        targetFreq,
        now + transitionTime
      );

      // Update sub oscillator (stay one octave below)
      if (this.subOscillator) {
        this.subOscillator.frequency.cancelScheduledValues(now);
        this.subOscillator.frequency.setValueAtTime(
          this.subOscillator.frequency.value,
          now
        );
        this.subOscillator.frequency.linearRampToValueAtTime(
          targetFreq * 0.5,
          now + transitionTime
        );
      }
    }

    // Delay feedback
    if (params.delayFeedback !== undefined) {
      this.delayFeedback.gain.cancelScheduledValues(now);
      this.delayFeedback.gain.setValueAtTime(
        this.delayFeedback.gain.value,
        now
      );
      this.delayFeedback.gain.linearRampToValueAtTime(
        params.delayFeedback,
        now + transitionTime
      );
    }

    // Stereo panning
    if (params.pan !== undefined) {
      this.panner.pan.cancelScheduledValues(now);
      this.panner.pan.setValueAtTime(this.panner.pan.value, now);
      this.panner.pan.linearRampToValueAtTime(params.pan, now + transitionTime);
    }

    // Volume - skip if we're in fade-in or fade-out period
    if (params.volume !== undefined && !isFadingIn && !this.isFadingOut) {
      this.targetVolume = params.volume;
      this.masterGain.gain.cancelScheduledValues(now);
      this.masterGain.gain.setValueAtTime(this.masterGain.gain.value, now);
      this.masterGain.gain.linearRampToValueAtTime(
        params.volume,
        now + transitionTime
      );
    }

    // Sweep amount (radio tuning effect volume)
    if (params.sweepAmount !== undefined && this.sweepGain) {
      const targetSweepVol = params.sweepAmount * this.config.sweepGain;
      this.sweepGain.gain.cancelScheduledValues(now);
      this.sweepGain.gain.setValueAtTime(this.sweepGain.gain.value, now);
      this.sweepGain.gain.linearRampToValueAtTime(
        targetSweepVol,
        now + transitionTime
      );
    }
  }

  /**
   * Set master volume
   * @param {number} volume - Volume (0-1)
   */
  setVolume(volume) {
    this.config.volume = Math.max(0, Math.min(1, volume));

    if (this.masterGain && this.isPlaying && !this.isFadingOut) {
      const ctx = this.audioContext;
      const now = ctx.currentTime;
      const timeSinceStart = now - this.startTime;
      const isFadingIn = timeSinceStart < this.config.fadeInTime;

      // Only change volume if not in fade-in period
      if (!isFadingIn) {
        this.targetVolume = this.config.volume;
        this.masterGain.gain.cancelScheduledValues(now);
        this.masterGain.gain.setValueAtTime(this.masterGain.gain.value, now);
        this.masterGain.gain.linearRampToValueAtTime(
          this.config.volume,
          now + 0.1
        );
      }
    }
  }

  /**
   * Cleanup
   */
  dispose() {
    this.logger.log("Disposing audio");

    this.stop();

    if (this.lfo) {
      this.lfo.stop();
      this.lfo.disconnect();
      this.lfo = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    this.masterGain = null;
    this.filter = null;
    this.distortion = null;
    this.delay = null;
    this.delayFeedback = null;
    this.compressor = null;
    this.panner = null;
    this.lfoGain = null;
  }
}

export default ProceduralAudio;
