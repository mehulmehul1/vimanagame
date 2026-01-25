import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import AssetImporter from './AssetImporter.js';

/**
 * Thumbnail cache entry
 */
interface ThumbnailEntry {
    url: string; // data URL or blob URL
    timestamp: number; // Generation timestamp
    assetMtime: number; // Asset file modification time
    width: number;
    height: number;
}

/**
 * Thumbnail cache storage format
 */
interface ThumbnailCacheData {
    version: string;
    entries: Record<string, ThumbnailEntry>;
}

/**
 * ThumbnailCache - Manages thumbnail generation and caching
 *
 * Features:
 * - Generate thumbnails on asset import
 * - Save to editor/.cache/ folder as PNG
 * - Index cache on editor load
 * - Invalidate cache when asset changes (mtime check)
 *
 * Usage:
 * const cache = ThumbnailCache.getInstance();
 * await cache.initialize();
 * const thumbnail = await cache.getThumbnail(assetPath);
 */
class ThumbnailCache {
    private static instance: ThumbnailCache;

    // Cache storage
    private cache: Map<string, ThumbnailEntry> = new Map();
    private cachePath: string = '.cache/thumbnails';
    private cacheIndexFile: string = '.cache/thumbnails/index.json';

    // Settings
    private thumbnailSize: number = 128;
    private thumbnailQuality: number = 0.7;
    private maxCacheSize: number = 100 * 1024 * 1024; // 100MB

    // Loader
    private gltfLoader: GLTFLoader;

    // Canvas context for thumbnail generation
    private canvas: HTMLCanvasElement | null = null;
    private ctx: CanvasRenderingContext2D | null = null;

    // IndexedDB for larger cache storage
    private db: IDBDatabase | null = null;
    private DB_NAME = 'ThumbnailCache';
    private DB_VERSION = 1;
    private STORE_NAME = 'thumbnails';

    // Event listeners
    private eventListeners: Map<string, Set<Function>> = new Map();

    private constructor() {
        this.gltfLoader = new GLTFLoader();
        console.log('ThumbnailCache: Constructor complete');
    }

    public static getInstance(): ThumbnailCache {
        if (!ThumbnailCache.instance) {
            ThumbnailCache.instance = new ThumbnailCache();
        }
        return ThumbnailCache.instance;
    }

    /**
     * Initialize the thumbnail cache
     */
    public async initialize(): Promise<void> {
        console.log('ThumbnailCache: Initializing...');

        try {
            // Initialize IndexedDB
            await this.initIndexedDB();

            // Load cache index from IndexedDB
            await this.loadCacheIndex();

            // Setup canvas for thumbnail generation
            this.canvas = document.createElement('canvas');
            this.canvas.width = this.thumbnailSize;
            this.canvas.height = this.thumbnailSize;
            this.ctx = this.canvas.getContext('2d');

            console.log('ThumbnailCache: Initialization complete');
            this.emit('initialized');
        } catch (error) {
            console.error('ThumbnailCache: Initialization failed:', error);
            // Continue without cache
            this.emit('initialized');
        }
    }

    /**
     * Initialize IndexedDB for thumbnail storage
     */
    private initIndexedDB(): Promise<void> {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.DB_NAME, this.DB_VERSION);

            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this.db = request.result;
                resolve();
            };

            request.onupgradeneeded = (event) => {
                const db = (event.target as IDBOpenDBRequest).result;
                if (!db.objectStoreNames.contains(this.STORE_NAME)) {
                    const store = db.createObjectStore(this.STORE_NAME, { keyPath: 'assetPath' });
                    store.createIndex('timestamp', 'timestamp', { unique: false });
                }
            };
        });
    }

    /**
     * Load cache index from IndexedDB
     */
    private async loadCacheIndex(): Promise<void> {
        if (!this.db) return;

        return new Promise((resolve, reject) => {
            const transaction = this.db!.transaction([this.STORE_NAME], 'readonly');
            const store = transaction.objectStore(this.STORE_NAME);
            const request = store.getAll();

            request.onsuccess = () => {
                const entries = request.result as Array<{ assetPath: string } & ThumbnailEntry>;
                entries.forEach(entry => {
                    this.cache.set(entry.assetPath, entry);
                });
                console.log(`ThumbnailCache: Loaded ${entries.length} cached thumbnails`);
                resolve();
            };

            request.onerror = () => reject(request.error);
        });
    }

    /**
     * Get thumbnail for an asset
     */
    public async getThumbnail(assetPath: string, assetMtime?: number): Promise<string | null> {
        // Check if cache has entry and is valid
        const cached = this.cache.get(assetPath);

        if (cached) {
            // Check if cache is still valid (asset hasn't changed)
            if (assetMtime === undefined || cached.assetMtime === assetMtime) {
                return cached.url;
            }
            // Invalidate stale cache
            this.invalidate(assetPath);
        }

        return null;
    }

    /**
     * Generate thumbnail for an asset
     */
    public async generateThumbnail(
        assetPath: string,
        assetType: 'image' | 'model' | 'splat',
        assetData?: string | ArrayBuffer
    ): Promise<string> {
        const startTime = Date.now();

        try {
            let thumbnailUrl: string;

            switch (assetType) {
                case 'image':
                    thumbnailUrl = await this.generateImageThumbnail(assetData as string);
                    break;
                case 'model':
                    thumbnailUrl = await this.generateModelThumbnail(assetPath, assetData as ArrayBuffer);
                    break;
                case 'splat':
                    thumbnailUrl = await this.generateSplatThumbnail(assetPath);
                    break;
                default:
                    throw new Error(`Unknown asset type: ${assetType}`);
            }

            // Create cache entry
            const entry: ThumbnailEntry = {
                url: thumbnailUrl,
                timestamp: Date.now(),
                assetMtime: Date.now(),
                width: this.thumbnailSize,
                height: this.thumbnailSize
            };

            // Save to cache
            await this.saveToCache(assetPath, entry);

            console.log(`ThumbnailCache: Generated thumbnail for ${assetPath} in ${Date.now() - startTime}ms`);
            this.emit('thumbnailGenerated', { assetPath, entry });

            return thumbnailUrl;
        } catch (error) {
            console.error(`ThumbnailCache: Failed to generate thumbnail for ${assetPath}:`, error);
            // Return placeholder
            return this.generatePlaceholderThumbnail(assetType);
        }
    }

    /**
     * Generate thumbnail for image
     */
    private async generateImageThumbnail(imageData: string): Promise<string> {
        return new Promise((resolve, reject) => {
            const img = new Image();

            img.onload = () => {
                if (!this.ctx || !this.canvas) {
                    reject(new Error('Canvas not initialized'));
                    return;
                }

                // Clear canvas
                this.ctx.fillStyle = '#1a1a1a';
                this.ctx.fillRect(0, 0, this.thumbnailSize, this.thumbnailSize);

                // Calculate aspect ratio
                const scale = Math.min(
                    this.thumbnailSize / img.width,
                    this.thumbnailSize / img.height
                );

                const width = img.width * scale;
                const height = img.height * scale;
                const x = (this.thumbnailSize - width) / 2;
                const y = (this.thumbnailSize - height) / 2;

                this.ctx.drawImage(img, x, y, width, height);

                resolve(this.canvas.toDataURL('image/jpeg', this.thumbnailQuality));
            };

            img.onerror = () => reject(new Error('Failed to load image'));
            img.src = imageData;
        });
    }

    /**
     * Generate thumbnail for 3D model
     */
    private async generateModelThumbnail(assetPath: string, modelData?: ArrayBuffer): Promise<string> {
        return new Promise((resolve, reject) => {
            // Create a temporary scene for rendering
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a1a);

            const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
            camera.position.set(0, 0, 3);

            const renderer = new THREE.WebGLRenderer({
                canvas: this.canvas!,
                antialias: true
            });
            renderer.setSize(this.thumbnailSize, this.thumbnailSize);

            // Add lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);

            // Load model
            const onLoad = (gltf: any) => {
                const model = gltf.scene;

                // Center and scale model
                const box = new THREE.Box3().setFromObject(model);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());

                model.position.sub(center);
                const maxDim = Math.max(size.x, size.y, size.z);
                const scale = 2 / maxDim;
                model.scale.setScalar(scale);

                scene.add(model);

                // Render
                renderer.render(scene, camera);

                // Get thumbnail
                const thumbnail = this.canvas!.toDataURL('image/jpeg', this.thumbnailQuality);

                // Cleanup
                scene.remove(model);
                renderer.dispose();

                resolve(thumbnail);
            };

            const onError = (error: any) => {
                renderer.dispose();
                reject(error);
            };

            if (modelData) {
                // Load from array buffer
                this.gltfLoader.parse(modelData as ArrayBuffer, '', onLoad, onError);
            } else {
                // Load from URL
                this.gltfLoader.load(assetPath, onLoad, undefined, onError);
            }
        });
    }

    /**
     * Generate thumbnail for splat (placeholder)
     */
    private async generateSplatThumbnail(assetPath: string): Promise<string> {
        // For splats, we create a stylized placeholder
        if (!this.ctx || !this.canvas) {
            return this.generatePlaceholderThumbnail('splat');
        }

        // Draw gradient background
        const gradient = this.ctx.createRadialGradient(
            this.thumbnailSize / 2, this.thumbnailSize / 2, 0,
            this.thumbnailSize / 2, this.thumbnailSize / 2, this.thumbnailSize / 2
        );
        gradient.addColorStop(0, '#4a9eff');
        gradient.addColorStop(1, '#1a3a5f');

        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.thumbnailSize, this.thumbnailSize);

        // Draw "SPLAT" text
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = 'bold 16px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText('SPLAT', this.thumbnailSize / 2, this.thumbnailSize / 2);

        return this.canvas.toDataURL('image/jpeg', this.thumbnailQuality);
    }

    /**
     * Generate placeholder thumbnail
     */
    private generatePlaceholderThumbnail(assetType: string): string {
        if (!this.ctx || !this.canvas) {
            return 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128"><rect fill="#1a1a1a" width="128" height="128"/></svg>');
        }

        // Draw background
        this.ctx.fillStyle = '#1a1a1a';
        this.ctx.fillRect(0, 0, this.thumbnailSize, this.thumbnailSize);

        // Draw border
        this.ctx.strokeStyle = '#4d4d4d';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(2, 2, this.thumbnailSize - 4, this.thumbnailSize - 4);

        // Draw type label
        this.ctx.fillStyle = '#888';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(assetType.toUpperCase(), this.thumbnailSize / 2, this.thumbnailSize / 2);

        return this.canvas.toDataURL('image/jpeg', this.thumbnailQuality);
    }

    /**
     * Save thumbnail to cache
     */
    private async saveToCache(assetPath: string, entry: ThumbnailEntry): Promise<void> {
        // Store in memory cache
        this.cache.set(assetPath, entry);

        // Store in IndexedDB
        if (!this.db) return;

        return new Promise((resolve, reject) => {
            const transaction = this.db!.transaction([this.STORE_NAME], 'readwrite');
            const store = transaction.objectStore(this.STORE_NAME);

            const record = {
                assetPath,
                ...entry
            };

            const request = store.put(record);

            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    /**
     * Invalidate cached thumbnail for an asset
     */
    public invalidate(assetPath: string): void {
        this.cache.delete(assetPath);

        if (this.db) {
            const transaction = this.db.transaction([this.STORE_NAME], 'readwrite');
            const store = transaction.objectStore(this.STORE_NAME);
            store.delete(assetPath);
        }

        console.log(`ThumbnailCache: Invalidated thumbnail for ${assetPath}`);
        this.emit('thumbnailInvalidated', { assetPath });
    }

    /**
     * Clear all cached thumbnails
     */
    public async clearAll(): Promise<void> {
        this.cache.clear();

        if (this.db) {
            return new Promise((resolve, reject) => {
                const transaction = this.db!.transaction([this.STORE_NAME], 'readwrite');
                const store = transaction.objectStore(this.STORE_NAME);
                const request = store.clear();

                request.onsuccess = () => {
                    console.log('ThumbnailCache: Cleared all thumbnails');
                    this.emit('cacheCleared');
                    resolve();
                };
                request.onerror = () => reject(request.error);
            });
        }
    }

    /**
     * Get cache statistics
     */
    public getCacheStats(): {
        count: number;
        size: number; // Estimated
    } {
        return {
            count: this.cache.size,
            size: this.cache.size * (this.thumbnailSize * this.thumbnailSize * 4) // Rough estimate
        };
    }

    /**
     * Set thumbnail size
     */
    public setThumbnailSize(size: number): void {
        this.thumbnailSize = Math.max(64, Math.min(512, size));
        if (this.canvas) {
            this.canvas.width = this.thumbnailSize;
            this.canvas.height = this.thumbnailSize;
        }
    }

    /**
     * Set thumbnail quality
     */
    public setThumbnailQuality(quality: number): void {
        this.thumbnailQuality = Math.max(0.1, Math.min(1.0, quality));
    }

    /**
     * Register event listener
     */
    public on(eventName: string, callback: Function): void {
        if (!this.eventListeners.has(eventName)) {
            this.eventListeners.set(eventName, new Set());
        }
        this.eventListeners.get(eventName)!.add(callback);
    }

    /**
     * Unregister event listener
     */
    public off(eventName: string, callback: Function): void {
        const listeners = this.eventListeners.get(eventName);
        if (listeners) {
            listeners.delete(callback);
        }
    }

    /**
     * Emit event
     */
    private emit(eventName: string, data?: any): void {
        const listeners = this.eventListeners.get(eventName);
        if (listeners) {
            listeners.forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`ThumbnailCache: Error in event listener for ${eventName}:`, error);
                }
            });
        }
    }

    /**
     * Clean up
     */
    public destroy(): void {
        if (this.db) {
            this.db.close();
        }
        this.cache.clear();
        this.eventListeners.clear();
        console.log('ThumbnailCache: Destroyed');
    }
}

export default ThumbnailCache;
