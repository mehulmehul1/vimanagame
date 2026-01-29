import * as THREE from 'three';
import { JellyMaterial } from './JellyMaterial';

/**
 * JellyCreature - Procedural jellyfish that teaches the player the song
 *
 * These bioluminescent creatures emerge from the water, demonstrate a note,
 * and submerge again. They are the "musical messengers" of the Vimana.
 */
export class JellyCreature extends THREE.Mesh {
    // Story 1.2 spec values
    private static readonly BASE_SCALE = 0.3; // Story spec: sphere radius 0.3
    private static readonly TARGET_STRING_Y = 1.5; // Story spec: y = 1.5 when teaching
    private static readonly SUBMERGED_Y_OFFSET = -0.2; // Story spec: y = -0.2 (below water surface)
    private static readonly ARC_HEIGHT = 0.5; // Story spec: arc height offset

    private material: JellyMaterial;
    private spawnPosition: THREE.Vector3;
    private targetPosition: THREE.Vector3;
    private homePosition: THREE.Vector3;
    private jellyLight: THREE.PointLight; // Point light for visibility
    private glowSprite: THREE.Sprite; // Emissive glow sprite

    private state: 'spawning' | 'teaching' | 'submerging' | 'idle' | 'hidden' = 'hidden';
    private animTime: number = 0;
    private pulseRate: number = 2.0;
    private targetString: number = 0;
    private isTeaching: boolean = false;
    private teachingIntensity: number = 0;

    // Swim physics properties
    private velocity: THREE.Vector3;
    private angularVelocity: THREE.Vector3;
    private swimPhase: number = 0;
    private waterSurfaceY: number = 0;
    private isInWater: boolean = false;

    constructor(spawnPosition: THREE.Vector3 = new THREE.Vector3(0, 0, 0), noteIndex: number = 0) {
        // Create geometry and material FIRST (before accessing 'this')
        // Story spec: radius 0.3, 64x64 segments for smooth displacement
        const geometry = new THREE.SphereGeometry(JellyCreature.BASE_SCALE, 64, 64);
        const pulseRate = 2.0; // Local variable - use this instead of this.pulseRate

        // Use JellyMaterial wrapper (selects TSL/WebGPU or GLSL/WebGL2 automatically)
        const material = new JellyMaterial();
        material.setPulseRate(pulseRate);

        super(geometry, material.getMaterial());
        this.material = material;

        this.spawnPosition = spawnPosition.clone();
        this.homePosition = new THREE.Vector3(
            spawnPosition.x,
            spawnPosition.y + JellyCreature.SUBMERGED_Y_OFFSET, // Below water surface
            spawnPosition.z
        );
        this.targetPosition = new THREE.Vector3(
            spawnPosition.x,
            spawnPosition.y + JellyCreature.TARGET_STRING_Y,
            spawnPosition.z
        );
        this.position.copy(this.homePosition);
        this.scale.setScalar(0);

        // Initialize swim physics
        this.velocity = new THREE.Vector3();
        this.angularVelocity = new THREE.Vector3();

        // Create point light for visibility
        this.jellyLight = new THREE.PointLight(0x00ffff, 3, 3); // Smaller radius for smaller jelly
        this.jellyLight.position.set(0, 0.15, 0); // Scaled down for 0.3 radius
        this.add(this.jellyLight);

        // Create glow sprite (billboard halo)
        const glowTexture = this.createGlowTexture();
        const glowMaterial = new THREE.SpriteMaterial({
            map: glowTexture,
            color: 0x00ffff,
            transparent: true,
            opacity: 0.4,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });
        this.glowSprite = new THREE.Sprite(glowMaterial);
        this.glowSprite.scale.set(0.75, 0.75, 1); // Scaled down (2.5 * 0.3)
        this.add(this.glowSprite);
    }

    /**
     * Create a soft glow texture for the sprite
     */
    private createGlowTexture(): THREE.CanvasTexture {
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 128;
        const ctx = canvas.getContext('2d')!;

        // Create radial gradient for soft glow
        const gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
        gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
        gradient.addColorStop(0.2, 'rgba(200, 255, 255, 0.8)');
        gradient.addColorStop(0.5, 'rgba(100, 255, 255, 0.3)');
        gradient.addColorStop(1, 'rgba(0, 255, 255, 0)');

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 128, 128);

        return new THREE.CanvasTexture(canvas);
    }

    public spawn(targetString: number): void {
        this.targetString = targetString;
        this.state = 'spawning';
        this.animTime = 0;
        this.position.copy(this.homePosition);
        this.scale.setScalar(0);

        // Enable light for visibility
        this.jellyLight.intensity = 3;
        this.glowSprite.visible = true;
    }

    public beginTeaching(): void {
        this.state = 'teaching';
        this.isTeaching = true;
        this.teachingIntensity = 1.0;
        this.material.setTeaching(true);

        // Intensify light during teaching
        this.jellyLight.intensity = 6;
        this.jellyLight.color.setHex(0x00ffff);
        this.glowSprite.material.opacity = 0.8;
        (this.glowSprite.material as THREE.SpriteMaterial).color.setHex(0x00ffff);
    }

    public submerge(): void {
        this.state = 'submerging';
        this.isTeaching = false;
        this.material.setTeaching(false);

        // Dim light during submersion
        this.jellyLight.intensity = 1;
    }

    public update(deltaTime: number, time: number): void {
        this.material.setTime(time);
        const pulse = Math.sin(time * this.pulseRate) * 0.1 + 1.0;

        // Update swim physics (only during idle, not during scripted animations)
        if (this.state === 'idle') {
            this.updateSwimPhysics(deltaTime);
        }

        switch (this.state) {
            case 'spawning':
                this.updateSpawning(deltaTime);
                break;
            case 'teaching':
                this.updateTeaching(deltaTime, pulse);
                break;
            case 'submerging':
                this.updateSubmerging(deltaTime);
                break;
            case 'idle':
                this.scale.setScalar(JellyCreature.BASE_SCALE * pulse);
                break;
        }
    }

    private updateSpawning(deltaTime: number): void {
        this.animTime += deltaTime;
        const duration = 1.5; // Story spec: 1.5 seconds emerge
        const t = Math.min(this.animTime / duration, 1.0);

        // Story spec: easeOutQuad for ascent
        const easeT = t * (2.0 - t);

        // Scale in smoothly
        this.scale.setScalar(JellyCreature.BASE_SCALE * easeT);

        // Story spec: Parabolic arc: y = 4 * arcHeight * (t - t²)
        const arcY = JellyCreature.ARC_HEIGHT * 4.0 * (t - t * t);

        // Lerp X and Z linearly
        this.position.x = this.homePosition.x +
            (this.targetPosition.x - this.homePosition.x) * t;
        this.position.z = this.homePosition.z +
            (this.targetPosition.z - this.homePosition.z) * t;

        // Y follows parabola from home to target
        this.position.y = this.homePosition.y +
            (this.targetPosition.y - this.homePosition.y) * t + arcY;

        if (t >= 1.0) {
            this.state = 'idle';
        }
    }

    private updateTeaching(deltaTime: number, pulse: number): void {
        const teachingPulse = pulse * (1.0 + this.teachingIntensity);
        this.scale.setScalar(JellyCreature.BASE_SCALE * teachingPulse);

        this.teachingIntensity = Math.max(0.5, Math.sin(this.animTime * 3) * 0.5 + 0.5);
        this.material.setTeachingIntensity(this.teachingIntensity);

        // Animate glow sprite with teaching pulse (scaled for smaller jelly)
        const glowScale = 0.75 + this.teachingIntensity * 0.45;
        this.glowSprite.scale.set(glowScale, glowScale, 1);
        (this.glowSprite.material as THREE.SpriteMaterial).opacity = 0.4 + this.teachingIntensity * 0.4;

        // Pulse light intensity
        this.jellyLight.intensity = 4 + this.teachingIntensity * 3;

        this.animTime += deltaTime;
    }

    private updateSubmerging(deltaTime: number): void {
        this.animTime += deltaTime;
        const duration = 1.5; // Story spec: 1.5 seconds submerge
        const t = Math.min(this.animTime / duration, 1.0);

        const scale = 1.0 - t;
        this.scale.setScalar(JellyCreature.BASE_SCALE * scale);
        this.position.lerp(this.homePosition, t * 2);

        // Fade out light and glow
        this.jellyLight.intensity = 1.0 - t;
        this.glowSprite.material.opacity = (0.4 + this.teachingIntensity * 0.4) * (1 - t);

        if (t >= 1.0) {
            this.state = 'hidden';
            this.position.copy(this.homePosition);
            // Disable light when hidden
            this.jellyLight.intensity = 0;
            this.glowSprite.visible = false;
        }
    }

    public setPulseRate(rate: number): void {
        this.pulseRate = rate;
        this.material.setPulseRate(rate);
    }

    public setColor(color: THREE.Color): void {
        this.material.setColor(color);
    }

    public setCameraPosition(position: THREE.Vector3): void {
        this.material.setCameraPosition(position);
    }

    public getState(): string {
        return this.state;
    }

    public isReady(): boolean {
        return this.state === 'idle' || this.state === 'teaching';
    }

    public isHidden(): boolean {
        return this.state === 'hidden';
    }

    public setHomePosition(position: THREE.Vector3): void {
        // Update home position (where jelly spawns from/submerges to)
        this.homePosition = position.clone();
        // Apply submerged offset (below water surface)
        this.homePosition.y = position.y + JellyCreature.SUBMERGED_Y_OFFSET;

        // Update target position (where jelly goes when teaching)
        this.targetPosition = new THREE.Vector3(
            position.x,
            position.y + JellyCreature.TARGET_STRING_Y,
            position.z
        );

        // If jelly is hidden, idle, or spawning, update its current position too
        if (this.state === 'hidden' || this.state === 'idle' || this.state === 'spawning') {
            this.position.copy(this.homePosition);
        }

        console.log(`[JellyCreature] Home position set to:`, this.homePosition);
    }

    /**
     * Update water surface Y for swim physics
     * Called by JellyManager after water surface detection
     */
    public setWaterSurface(y: number): void {
        this.waterSurfaceY = y;
    }

    /**
     * Update swim physics (buoyancy, drag, organic movement)
     * Only applies when jelly is idle (not during scripted animations)
     */
    private updateSwimPhysics(deltaTime: number): void {
        // Check if in water (jelly bottom vs water surface)
        const jellyBottomY = this.position.y - (JellyCreature.BASE_SCALE * 0.5);
        this.isInWater = jellyBottomY < this.waterSurfaceY;

        if (this.isInWater) {
            // Submersion depth (0 = at surface, positive = underwater)
            const depth = this.waterSurfaceY - jellyBottomY;
            const submergedRatio = Math.min(depth / JellyCreature.BASE_SCALE, 1.0);

            // Buoyancy: upward force proportional to submerged volume
            const buoyancyForce = 2.0 * submergedRatio;
            this.velocity.y += buoyancyForce * deltaTime;

            // Water drag (slows movement)
            const dragFactor = 1.0 - (submergedRatio * 0.15);
            this.velocity.multiplyScalar(Math.max(0.85, dragFactor));
            this.angularVelocity.multiplyScalar(Math.max(0.85, dragFactor));

            // Organic bobbing/swimming motion
            this.swimPhase += deltaTime * 2.0;
            const bobAmount = Math.sin(this.swimPhase) * 0.02 * submergedRatio;
            this.position.y += bobAmount;

            // Gentle wobble rotation
            this.angularVelocity.x = Math.sin(this.swimPhase * 0.7) * 0.1;
            this.angularVelocity.z = Math.cos(this.swimPhase * 0.9) * 0.1;
        } else {
            // In air - gentle gravity applies
            this.velocity.y -= 0.5 * deltaTime;
        }

        // Apply velocity
        this.position.addScaledVector(this.velocity, deltaTime);
        this.rotation.x += this.angularVelocity.x * deltaTime;
        this.rotation.z += this.angularVelocity.z * deltaTime;

        // Clamp to not go too deep
        const maxDepth = this.waterSurfaceY - 0.5;
        if (this.position.y < maxDepth) {
            this.position.y = maxDepth;
            this.velocity.y = Math.max(0, this.velocity.y);
        }

        // Decay velocity to prevent explosion
        this.velocity.multiplyScalar(0.98);
        this.angularVelocity.multiplyScalar(0.95);
    }

    public destroy(): void {
        this.geometry.dispose();
        this.material.destroy();
    }
}
