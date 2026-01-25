import * as THREE from 'three';
import { jellyVertexShader, jellyFragmentShader } from '../shaders';

/**
 * JellyCreature - Procedural jellyfish that teaches the player the song
 *
 * These bioluminescent creatures emerge from the water, demonstrate a note,
 * and submerge again. They are the "musical messengers" of the Vimana.
 */
export class JellyCreature extends THREE.Mesh {
    private static readonly BASE_SCALE = 1.0; // Increased for better visibility (was 0.4)
    private static readonly TARGET_STRING_Y = 1.6; // Higher for better visibility (was 0.8)
    private static readonly WATER_LEVEL_Y = 0.2; // Above water level for visibility

    private material: THREE.ShaderMaterial;
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

    constructor(spawnPosition: THREE.Vector3 = new THREE.Vector3(0, 0, 0), noteIndex: number = 0) {
        // Create geometry and material FIRST (before accessing 'this')
        const geometry = new THREE.SphereGeometry(1, 32, 32);
        const pulseRate = 2.0; // Local variable - use this instead of this.pulseRate

        const uniforms = {
            uTime: { value: 0 },
            uPulseRate: { value: pulseRate },
            uIsTeaching: { value: 0.0 },
            uTeachingIntensity: { value: 0.0 },
            uBioluminescentColor: { value: new THREE.Color(0x00ffaa) },
            uCameraPosition: { value: new THREE.Vector3() }
        };

        const material = new THREE.ShaderMaterial({
            vertexShader: jellyVertexShader,
            fragmentShader: jellyFragmentShader,
            uniforms: uniforms,
            transparent: true,
            side: THREE.DoubleSide
        });

        super(geometry, material);
        this.material = material;

        this.spawnPosition = spawnPosition.clone();
        this.homePosition = new THREE.Vector3(
            spawnPosition.x,
            JellyCreature.WATER_LEVEL_Y,
            spawnPosition.z
        );
        this.targetPosition = new THREE.Vector3(
            spawnPosition.x,
            JellyCreature.TARGET_STRING_Y,
            spawnPosition.z
        );
        this.position.copy(this.homePosition);
        this.scale.setScalar(0);

        // Create point light for visibility
        this.jellyLight = new THREE.PointLight(0x00ffff, 3, 8);
        this.jellyLight.position.set(0, 0.5, 0);
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
        this.glowSprite.scale.set(2.5, 2.5, 1);
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
        this.material.uniforms.uIsTeaching.value = 1.0;

        // Intensify light during teaching
        this.jellyLight.intensity = 6;
        this.jellyLight.color.setHex(0x00ffff);
        this.glowSprite.material.opacity = 0.8;
        (this.glowSprite.material as THREE.SpriteMaterial).color.setHex(0x00ffff);
    }

    public submerge(): void {
        this.state = 'submerging';
        this.isTeaching = false;
        this.material.uniforms.uIsTeaching.value = 0.0;

        // Dim light during submersion
        this.jellyLight.intensity = 1;
    }

    public update(deltaTime: number, time: number): void {
        this.material.uniforms.uTime.value = time;
        const pulse = Math.sin(time * this.pulseRate) * 0.1 + 1.0;

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
        const duration = 1.0;
        const t = Math.min(this.animTime / duration, 1.0);

        const easeT = 1.0 - Math.pow(1.0 - t, 3);
        const bounce = 1.0 + Math.sin(t * Math.PI) * 0.2;

        this.scale.setScalar(JellyCreature.BASE_SCALE * easeT * bounce);

        const arcHeight = Math.sin(t * Math.PI) * 0.5;
        this.position.y = this.homePosition.y +
            (this.targetPosition.y - this.homePosition.y) * t + arcHeight;

        if (t >= 1.0) {
            this.state = 'idle';
        }
    }

    private updateTeaching(deltaTime: number, pulse: number): void {
        const teachingPulse = pulse * (1.0 + this.teachingIntensity);
        this.scale.setScalar(JellyCreature.BASE_SCALE * teachingPulse);

        this.teachingIntensity = Math.max(0.5, Math.sin(this.animTime * 3) * 0.5 + 0.5);
        this.material.uniforms.uTeachingIntensity.value = this.teachingIntensity;

        // Animate glow sprite with teaching pulse
        const glowScale = 2.5 + this.teachingIntensity * 1.5;
        this.glowSprite.scale.set(glowScale, glowScale, 1);
        (this.glowSprite.material as THREE.SpriteMaterial).opacity = 0.4 + this.teachingIntensity * 0.4;

        // Pulse light intensity
        this.jellyLight.intensity = 4 + this.teachingIntensity * 3;

        this.animTime += deltaTime;
    }

    private updateSubmerging(deltaTime: number): void {
        this.animTime += deltaTime;
        const duration = 0.8;
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
        this.material.uniforms.uPulseRate.value = rate;
    }

    public setColor(color: THREE.Color): void {
        this.material.uniforms.uBioluminescentColor.value.copy(color);
    }

    public setCameraPosition(position: THREE.Vector3): void {
        this.material.uniforms.uCameraPosition.value.copy(position);
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
        this.homePosition.y = JellyCreature.WATER_LEVEL_Y;

        // Update target position (where jelly goes when teaching)
        this.targetPosition = new THREE.Vector3(
            position.x,
            JellyCreature.TARGET_STRING_Y,
            position.z
        );

        // If jelly is hidden, idle, or spawning, update its current position too
        if (this.state === 'hidden' || this.state === 'idle' || this.state === 'spawning') {
            this.position.copy(this.homePosition);
        }

        console.log(`[JellyCreature] Home position set to:`, this.homePosition);
    }

    public destroy(): void {
        this.geometry.dispose();
        this.material.dispose();
    }
}
