/**
 * ResourceManager - Manages Three.js resource lifecycle
 *
 * Tracks created resources for proper cleanup and disposal.
 * Prevents memory leaks through centralized resource management.
 */

import * as THREE from 'three';

type ResourceType = 'geometry' | 'material' | 'texture' | 'cubeTexture' | 'renderTarget' | 'object';

interface TrackedResource {
    type: ResourceType;
    resource: any;
    createdAt: number;
    disposed: boolean;
}

export class ResourceManager {
    private static instance: ResourceManager;
    private resources: Map<string, TrackedResource>;
    private resourceCounter: number;

    private constructor() {
        this.resources = new Map();
        this.resourceCounter = 0;
    }

    /**
     * Get singleton instance
     */
    public static getInstance(): ResourceManager {
        if (!ResourceManager.instance) {
            ResourceManager.instance = new ResourceManager();
        }
        return ResourceManager.instance;
    }

    /**
     * Track a geometry
     */
    public trackGeometry(geometry: THREE.BufferGeometry, id?: string): string {
        const resourceId = id || `geometry_${this.resourceCounter++}`;
        this.resources.set(resourceId, {
            type: 'geometry',
            resource: geometry,
            createdAt: Date.now(),
            disposed: false
        });
        return resourceId;
    }

    /**
     * Track a material
     */
    public trackMaterial(material: THREE.Material, id?: string): string {
        const resourceId = id || `material_${this.resourceCounter++}`;
        this.resources.set(resourceId, {
            type: 'material',
            resource: material,
            createdAt: Date.now(),
            disposed: false
        });
        return resourceId;
    }

    /**
     * Track a texture
     */
    public trackTexture(texture: THREE.Texture, id?: string): string {
        const resourceId = id || `texture_${this.resourceCounter++}`;
        this.resources.set(resourceId, {
            type: 'texture',
            resource: texture,
            createdAt: Date.now(),
            disposed: false
        });
        return resourceId;
    }

    /**
     * Track a cube texture
     */
    public trackCubeTexture(texture: THREE.CubeTexture, id?: string): string {
        const resourceId = id || `cubeTexture_${this.resourceCounter++}`;
        this.resources.set(resourceId, {
            type: 'cubeTexture',
            resource: texture,
            createdAt: Date.now(),
            disposed: false
        });
        return resourceId;
    }

    /**
     * Track a render target
     */
    public trackRenderTarget(target: THREE.WebGLRenderTarget, id?: string): string {
        const resourceId = id || `renderTarget_${this.resourceCounter++}`;
        this.resources.set(resourceId, {
            type: 'renderTarget',
            resource: target,
            createdAt: Date.now(),
            disposed: false
        });
        return resourceId;
    }

    /**
     * Track a 3D object (and its children recursively)
     */
    public trackObject(object: THREE.Object3D, id?: string): string {
        const resourceId = id || `object_${this.resourceCounter++}`;
        this.resources.set(resourceId, {
            type: 'object',
            resource: object,
            createdAt: Date.now(),
            disposed: false
        });

        // Recursively track geometries and materials
        object.traverse((child) => {
            if (child instanceof THREE.Mesh) {
                if (child.geometry) {
                    this.trackGeometry(child.geometry, `${resourceId}_geo_${child.uuid}`);
                }
                if (child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach((mat, i) => {
                            this.trackMaterial(mat, `${resourceId}_mat_${i}_${child.uuid}`);
                        });
                    } else {
                        this.trackMaterial(child.material, `${resourceId}_mat_${child.uuid}`);
                    }
                }
            }
        });

        return resourceId;
    }

    /**
     * Dispose a specific resource by ID
     */
    public dispose(resourceId: string): boolean {
        const tracked = this.resources.get(resourceId);
        if (!tracked || tracked.disposed) return false;

        this.disposeResource(tracked);
        this.resources.delete(resourceId);
        return true;
    }

    /**
     * Dispose resource based on type
     */
    private disposeResource(tracked: TrackedResource): void {
        if (tracked.disposed) return;

        switch (tracked.type) {
            case 'geometry':
                if (tracked.resource.dispose) {
                    tracked.resource.dispose();
                }
                break;
            case 'material':
                if (tracked.resource.dispose) {
                    tracked.resource.dispose();
                }
                break;
            case 'texture':
            case 'cubeTexture':
                if (tracked.resource.dispose) {
                    tracked.resource.dispose();
                }
                break;
            case 'renderTarget':
                if (tracked.resource.dispose) {
                    tracked.resource.dispose();
                }
                break;
            case 'object':
                this.disposeObject(tracked.resource);
                break;
        }

        tracked.disposed = true;
    }

    /**
     * Dispose object and all children
     */
    private disposeObject(object: THREE.Object3D): void {
        object.traverse((child) => {
            if (child instanceof THREE.Mesh) {
                if (child.geometry) {
                    child.geometry.dispose();
                }
                if (child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(mat => mat.dispose());
                    } else {
                        child.material.dispose();
                    }
                }
            }
        });
    }

    /**
     * Dispose all resources
     */
    public disposeAll(): void {
        this.resources.forEach((tracked) => {
            this.disposeResource(tracked);
        });
        this.resources.clear();
    }

    /**
     * Dispose all resources of a specific type
     */
    public disposeByType(type: ResourceType): number {
        let disposed = 0;
        this.resources.forEach((tracked, id) => {
            if (tracked.type === type && !tracked.disposed) {
                this.disposeResource(tracked);
                this.resources.delete(id);
                disposed++;
            }
        });
        return disposed;
    }

    /**
     * Get resource count by type
     */
    public getCountByType(type?: ResourceType): number {
        if (type) {
            let count = 0;
            this.resources.forEach(tracked => {
                if (tracked.type === type && !tracked.disposed) count++;
            });
            return count;
        }
        return this.resources.size;
    }

    /**
     * Get resource by ID
     */
    public getResource(id: string): any | null {
        const tracked = this.resources.get(id);
        return tracked && !tracked.disposed ? tracked.resource : null;
    }

    /**
     * Get memory usage estimate (in bytes)
     */
    public getMemoryEstimate(): { geometries: number; materials: number; textures: number } {
        let geometries = 0;
        let materials = 0;
        let textures = 0;

        this.resources.forEach(tracked => {
            if (tracked.disposed) return;

            switch (tracked.type) {
                case 'geometry':
                    if (tracked.resource.attributes) {
                        tracked.resource.attributes.forEach((attr: any) => {
                            geometries += attr.array?.byteLength || 0;
                        });
                    }
                    if (tracked.resource.index) {
                        geometries += tracked.resource.index.array?.byteLength || 0;
                    }
                    break;
                case 'material':
                    materials += 1024; // Approximate
                    break;
                case 'texture':
                case 'cubeTexture':
                    if (tracked.resource.image) {
                        const w = tracked.resource.image.width || 0;
                        const h = tracked.resource.image.height || 0;
                        textures += w * h * 4; // RGBA
                    }
                    if (tracked.resource.type === 'CubeTexture') {
                        textures *= 6;
                    }
                    break;
            }
        });

        return { geometries, materials, textures };
    }

    /**
     * Log resource statistics
     */
    public logStats(): void {
        const byType: Record<ResourceType, number> = {
            geometry: 0,
            material: 0,
            texture: 0,
            cubeTexture: 0,
            renderTarget: 0,
            object: 0
        };

        this.resources.forEach(tracked => {
            if (!tracked.disposed) {
                byType[tracked.type]++;
            }
        });

        const memory = this.getMemoryEstimate();

        console.group('[ResourceManager Stats]');
        console.log('Geometries:', byType.geometry);
        console.log('Materials:', byType.material);
        console.log('Textures:', byType.texture);
        console.log('Cube Textures:', byType.cubeTexture);
        console.log('Render Targets:', byType.renderTarget);
        console.log('Objects:', byType.object);
        console.log('---');
        console.log('Geometry Memory:', (memory.geometries / 1024).toFixed(2), 'KB');
        console.log('Texture Memory:', (memory.textures / 1024 / 1024).toFixed(2), 'MB');
        console.log('Total Tracked:', this.getCountByType());
        console.groupEnd();
    }

    /**
     * Cleanup old resources (older than specified milliseconds)
     */
    public cleanupOld(maxAge: number): number {
        const now = Date.now();
        let cleaned = 0;

        this.resources.forEach((tracked, id) => {
            if (!tracked.disposed && now - tracked.createdAt > maxAge) {
                this.disposeResource(tracked);
                this.resources.delete(id);
                cleaned++;
            }
        });

        return cleaned;
    }
}
