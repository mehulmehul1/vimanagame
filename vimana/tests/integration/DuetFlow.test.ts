import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as THREE from 'three';
import { HarpRoom } from '../../src/scenes/HarpRoom';

// Mock PlatformRideAnimator to avoid THREE.Mesh dependency issues
// Define the mock class inline in the vi.mock call to avoid hoisting issues
vi.mock('../../src/entities/PlatformRideAnimator', () => {
    return {
        PlatformRideAnimator: vi.fn().mockImplementation(() => ({
            update: vi.fn(),
            startRide: vi.fn(),
            hasStarted: vi.fn().mockReturnValue(false),
            isComplete: vi.fn().mockReturnValue(false),
            reset: vi.fn()
        }))
    };
});

describe('DuetFlow Integration', () => {
    let harpRoom: HarpRoom;
    let scene: THREE.Scene;
    let camera: THREE.PerspectiveCamera;
    let renderer: THREE.WebGLRenderer;

    beforeEach(() => {
        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera();

        // Create a mock renderer that satisfies the type requirements
        renderer = {
            domElement: document.createElement('canvas'),
            setSize: vi.fn(),
            setPixelRatio: vi.fn(),
            render: vi.fn(),
            clear: vi.fn(),
            dispose: vi.fn()
        } as any;

        harpRoom = new HarpRoom(scene, camera, renderer);
    });

    afterEach(() => {
        if (harpRoom) {
            try {
                harpRoom.destroy();
            } catch (e) {
                // Ignore cleanup errors in tests
            }
        }
        vi.clearAllMocks();
    });

    it('should initialize all systems correctly', async () => {
        await harpRoom.initialize(new THREE.Group()); // Pass empty group as mock scene

        expect(harpRoom.getScene()).toBeDefined();
    });

    it('should handle harp string interaction', async () => {
        await harpRoom.initialize(new THREE.Group());

        // Mock raycaster intersection
        const raycaster = new THREE.Raycaster();

        // Mock a string object in the scene
        const stringObj = new THREE.Mesh();
        stringObj.userData = { isHarpString: true, stringIndex: 0 };
        scene.add(stringObj);

        // Spy on raycaster
        const intersectSpy = vi.spyOn(raycaster, 'intersectObjects').mockReturnValue([
            { object: stringObj, distance: 1, point: new THREE.Vector3() } as any
        ]);

        // Manually click
        expect(() => harpRoom.onClick(raycaster)).not.toThrow();
        expect(intersectSpy).toHaveBeenCalled();
    });
});
