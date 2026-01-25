import React, { useState, useEffect, useRef } from 'react';
import SceneLoader from '../core/SceneLoader';
import EditorManager from '../core/EditorManager';
import type { SceneObjectData } from '../core/SceneLoader';
// Import scene data from parent directory
// @ts-ignore - Import from parent game src
import { sceneObjects as gameSceneObjects } from '@game-src/sceneData.js';

interface LoadShadowCzarSceneProps {
    // No props needed currently
}

/**
 * LoadShadowCzarScene - Button and logic to load Shadow Czar game scenes
 *
 * This component:
 * - Loads the sceneData.js file from ../src/data/sceneData.js
 * - Parses scene objects (splats, GLTF models, primitives)
 * - Loads them using SceneLoader (Spark.js for splats, GLTFLoader for models)
 */
const LoadShadowCzarScene: React.FC<LoadShadowCzarSceneProps> = () => {
    const [isLoading, setIsLoading] = useState(false);
    const [loadedCount, setLoadedCount] = useState(0);
    const [totalCount, setTotalCount] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [sceneLoaded, setSceneLoaded] = useState(false);
    const [splatsVisible, setSplatsVisible] = useState(true);
    const [splatCount, setSplatCount] = useState(0);

    // Track loaded splat objects for show/hide functionality
    const splatObjects = useRef<any[]>([]);

    // Get SceneLoader instance
    const [sceneLoader] = useState<SceneLoader>(() => SceneLoader.getInstance());

    // Initialize SceneLoader when mounted
    useEffect(() => {
        const initLoader = async () => {
            try {
                await sceneLoader.initialize();
                console.log('LoadShadowCzarScene: SceneLoader initialized');
            } catch (err) {
                console.error('LoadShadowCzarScene: Failed to initialize SceneLoader:', err);
                setError('Failed to initialize scene loader');
            }
        };
        initLoader();
    }, [sceneLoader]);

    /**
     * Load the Shadow Czar scene from sceneData.js
     */
    const loadShadowCzarScene = async () => {
        setIsLoading(true);
        setError(null);
        setLoadedCount(0);
        setSceneLoaded(false);

        // Clear previous splat tracking
        splatObjects.current = [];
        setSplatCount(0);

        // Track loading results
        let successCount = 0;
        let errorCount = 0;
        let splatsLoaded = 0;

        try {
            console.log('LoadShadowCzarScene: Loading scene data...');

            // Use imported scene data directly
            const sceneObjects = gameSceneObjects;
            const objectEntries = Object.entries(sceneObjects) as [string, SceneObjectData][];

            setTotalCount(objectEntries.length);
            console.log(`LoadShadowCzarScene: Found ${objectEntries.length} scene objects`);

            // Load each object
            // Note: successCount and errorCount already declared at function start

            // Sort by priority (highest first) and preload
            const sortedObjects = objectEntries.sort((a, b) => {
                const aPriority = a[1].priority || 0;
                const bPriority = b[1].priority || 0;
                return bPriority - aPriority;
            });

            // First, load only preload objects for initial scene
            // Filter out device-specific variants (Laptop, Desktop, Mobile) for performance
            const preloadObjects = sortedObjects.filter(([id, obj]) => {
                if (obj.preload !== true) return false;
                // Skip device-specific quality variants
                if (id.endsWith('Laptop') || id.endsWith('Desktop') || id.endsWith('Mobile')) {
                    console.log(`LoadShadowCzarScene: Skipping device variant: ${id}`);
                    return false;
                }
                return true;
            });
            console.log(`LoadShadowCzarScene: Loading ${preloadObjects.length} preload objects (device variants filtered)`);

            // Get EditorManager for scene access (available for future use)
            void EditorManager.getInstance();

            for (const [id, objectData] of preloadObjects) {
                try {
                    console.log(`LoadShadowCzarScene: Loading ${id} (${objectData.type})`);
                    const loadedObject = await sceneLoader.loadSceneObject(objectData);

                    // Track splat objects for show/hide functionality
                    if (objectData.type === 'splat' && loadedObject) {
                        splatObjects.current.push(loadedObject);
                        splatsLoaded++;
                        setSplatCount(splatsLoaded);
                    }

                    successCount++;
                    setLoadedCount(successCount);
                } catch (err) {
                    console.error(`LoadShadowCzarScene: Failed to load ${id}:`, err);
                    errorCount++;
                }
            }

            console.log(`LoadShadowCzarScene: Loaded ${successCount} objects (${splatsLoaded} splats), ${errorCount} errors`);
            setSceneLoaded(true);

        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Unknown error';
            console.error('LoadShadowCzarScene: Failed to load scene:', err);
            setError(errorMessage);
        } finally {
            setIsLoading(false);
        }
    };

    /**
     * Clear all loaded scene objects
     */
    const clearScene = () => {
        sceneLoader.unloadAll();
        splatObjects.current = [];
        setSplatCount(0);
        setSplatsVisible(true);
        setSceneLoaded(false);
        setLoadedCount(0);
        setTotalCount(0);
        setError(null);
        console.log('LoadShadowCzarScene: Scene cleared');
    };

    /**
     * Toggle visibility of all splat objects
     */
    const toggleSplats = () => {
        const newVisibility = !splatsVisible;
        setSplatsVisible(newVisibility);

        splatObjects.current.forEach(splat => {
            if (splat) {
                splat.visible = newVisibility;
            }
        });

        console.log(`LoadShadowCzarScene: ${newVisibility ? 'Showing' : 'Hiding'} ${splatObjects.current.length} splats`);
    };

    return (
        <div className="load-scene-container">
            <button
                className="load-scene-button"
                onClick={loadShadowCzarScene}
                disabled={isLoading}
            >
                {isLoading ? 'Loading...' : 'Load Shadow Czar Scene'}
            </button>

            {sceneLoaded && (
                <button
                    className="clear-scene-button"
                    onClick={clearScene}
                >
                    Clear Scene
                </button>
            )}

            {splatCount > 0 && (
                <button
                    className={`splat-toggle-button ${splatsVisible ? 'active' : ''}`}
                    onClick={toggleSplats}
                    title={`${splatsVisible ? 'Hide' : 'Show'} all ${splatCount} splat objects`}
                >
                    {splatsVisible ? '🌐 Hide Splats' : '👁️ Show Splats'}
                </button>
            )}

            {isLoading && (
                <span className="load-progress">
                    {loadedCount} / {totalCount} objects
                </span>
            )}

            {sceneLoaded && !isLoading && (
                <span className="load-success">
                    {loadedCount} objects loaded ({splatCount} splats)
                </span>
            )}

            {error && (
                <span className="load-error" title={error}>
                    Error loading scene
                </span>
            )}
        </div>
    );
};

export default LoadShadowCzarScene;
