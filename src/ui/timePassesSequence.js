import * as THREE from "three";
import { createParticleImage } from "../vfx/titleText.js";
import { TitleSequence } from "../vfx/titleSequence.js";
import { GAME_STATES } from "../gameData.js";
import { Logger } from "../utils/logger.js";

/**
 * TimePassesSequence - Manages the "SOME TIME PASSES..." particle image animation during LIGHTS_OUT
 */
export class TimePassesSequence {
  constructor(camera, options = {}) {
    this.camera = camera;
    this.uiManager = options.uiManager || null;
    this.gameManager = options.gameManager || null;
    this.isActive = false;
    this.sequenceStarted = false;

    this.logger = new Logger("TimePassesSequence", true);

    this.titleSequence = null;
    this.timePassesText = null;

    this.createTimePassesText();
  }

  /**
   * Create the "SOME TIME PASSES..." image particles
   */
  createTimePassesText() {
    this.textScene = new THREE.Scene();
    this.textCamera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1.0,
      20.0
    );

    const isMobile = this.gameManager?.getState?.()?.isMobile || false;
    const textZ = isMobile ? -4.0 : -2.25;

    const imageData = createParticleImage(this.textScene, {
      imageUrl: "/images/SomeTimePasses.png",
      position: { x: 0, y: 0, z: textZ },
      scale: 0.03125,
      animate: true,
      particleDensity: 0.5,
      alphaThreshold: 0.1,
      useImageColor: false,
      tintColor: new THREE.Color(0xffffff),
    });

    this.textScene.remove(imageData.mesh);
    this.textCamera.add(imageData.mesh);
    this.textScene.add(this.textCamera);
    imageData.mesh.userData.baseScale = 0.0125;
    imageData.mesh.visible = false;

    this.timePassesText = {
      mesh: imageData.mesh,
      particles: imageData.particles,
      update: imageData.update,
    };

    // Track when particles are loaded so we can initialize TitleSequence
    this.particlesLoaded = false;

    // Check for particles after image loads (check multiple times since loading is async)
    const checkParticles = () => {
      if (imageData.particles && imageData.particles.length > 0) {
        this.logger.log(`Particles created: ${imageData.particles.length}`);
        this.logger.log(
          `Mesh geometry vertices: ${imageData.mesh.geometry.attributes.position.count}`
        );
        this.particlesLoaded = true;

        // Initialize particles to hidden state before TitleSequence
        this.timePassesText.particles.forEach((p) => {
          p.opacity = 0.0;
          p.scale = 0.2;
        });

        // Update geometry immediately to reflect hidden state
        const opacities = this.timePassesText.mesh.geometry.attributes.opacity;
        const sizes = this.timePassesText.mesh.geometry.attributes.size;
        if (opacities && sizes) {
          this.timePassesText.particles.forEach((p, idx) => {
            opacities.array[idx] = 0.0;
            sizes.array[idx] = 0.28 * 0.2;
          });
          opacities.needsUpdate = true;
          sizes.needsUpdate = true;
        }

        // If sequence was already started but particles weren't loaded, create it now
        if (this.sequenceStarted && !this.titleSequence) {
          this.logger.log(
            "Creating TitleSequence now that particles are loaded"
          );
          this.createTitleSequence();
        }
      } else {
        setTimeout(checkParticles, 500);
      }
    };
    setTimeout(checkParticles, 500);
  }

  /**
   * Create the TitleSequence (separated so we can recreate it when particles load)
   */
  createTitleSequence() {
    // Ensure particles are hidden before TitleSequence initializes them
    if (this.timePassesText.particles) {
      this.timePassesText.particles.forEach((p) => {
        p.opacity = 0.0;
        p.scale = 0.2;
      });
    }

    this.titleSequence = new TitleSequence([this.timePassesText], {
      introDuration: 2.0,
      staggerDelay: 0,
      holdDuration: 3.0,
      outroDuration: 4.0,
      disperseDistance: 5.0,
      basePointSize: 0.15,
      onComplete: () => {
        this.logger.log("Time passes sequence complete");
        if (this.gameManager) {
          this.gameManager.setState({
            currentState: GAME_STATES.WAKING_UP,
          });
        }
        this.isActive = false;
      },
    });

    // Update geometry immediately to reflect initial hidden state
    if (this.titleSequence && typeof this.titleSequence.update === "function") {
      this.titleSequence.update(0);
    }
  }

  /**
   * Start the sequence when LIGHTS_OUT state is reached
   */
  start() {
    if (this.sequenceStarted) return;
    this.sequenceStarted = true;
    this.isActive = true;

    this.logger.log("Starting time passes sequence");

    // Check if particles are loaded
    const particleCount = this.timePassesText?.particles?.length || 0;
    this.logger.log(`Particles available: ${particleCount}`);

    if (particleCount === 0) {
      this.logger.warn(
        "No particles loaded yet - TitleSequence will be created when particles load"
      );
    } else {
      this.createTitleSequence();
    }

    this.timePassesText.mesh.visible = true;
    this.logger.log(`Mesh visible set to: ${this.timePassesText.mesh.visible}`);
  }

  /**
   * Update the sequence (called from main loop)
   */
  update(dt) {
    if (!this.isActive || !this.titleSequence) return;

    if (this.textCamera) {
      this.textCamera.position.copy(this.camera.position);
      this.textCamera.quaternion.copy(this.camera.quaternion);
      this.textCamera.aspect = this.camera.aspect;
      this.textCamera.updateProjectionMatrix();
    }

    this.titleSequence.update(dt);
  }

  /**
   * Get the text scene and camera for separate rendering
   */
  getTextRenderInfo() {
    if (!this.isActive) return null;
    return {
      scene: this.textScene,
      camera: this.textCamera,
    };
  }

  /**
   * Clean up resources
   */
  cleanup() {
    this.isActive = false;
    if (this.timePassesText && this.timePassesText.mesh) {
      this.timePassesText.mesh.visible = false;
    }
  }
}
