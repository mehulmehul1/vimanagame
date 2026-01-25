import { vi } from 'vitest';

// Mock Three.js
vi.mock('three', async () => {
    const actual = await vi.importActual('three');
    return {
        ...actual,
        WebGLRenderer: vi.fn().mockImplementation(() => ({
            setSize: vi.fn(),
            setPixelRatio: vi.fn(),
            render: vi.fn(),
            domElement: document.createElement('canvas'),
            dispose: vi.fn(),
            shadowMap: { enabled: false, type: 0 },
            toneMapping: 0,
            outputColorSpace: '',
        })),
        AudioListener: vi.fn().mockImplementation(() => ({
            position: { set: vi.fn(), copy: vi.fn() },
            rotation: { set: vi.fn(), copy: vi.fn() },
            add: vi.fn(),
            remove: vi.fn(),
        })),
        // Mock other heavy classes if needed
    };
});

// Mock Canvas and Window properties
HTMLCanvasElement.prototype.getContext = vi.fn();
(window as any).pointerLockElement = null;
window.innerWidth = 1920;
window.innerHeight = 1080;
window.devicePixelRatio = 1;

// Mock Pointer Lock API
document.exitPointerLock = vi.fn();
Element.prototype.requestPointerLock = vi.fn();
