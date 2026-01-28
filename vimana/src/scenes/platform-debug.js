/**
 * Platform Debug Helper
 * Run this in browser console to find and test the platform mesh
 */

// Add to window for console access
window.findPlatform = function() {
    const scene = window.gameManager?.currentScene;
    if (!scene) {
        console.error('No scene found. Make sure game is loaded.');
        return;
    }

    console.log('=== Scanning scene for platform mesh ===');

    const platforms: { name: string; pos: THREE.Vector3; type: string }[] = [];

    scene.traverse((child) => {
        if (child instanceof THREE.Mesh) {
            const name = child.name || '(unnamed)';
            const pos = child.position.clone();

            // Look for likely platform names
            const nameLower = name.toLowerCase();
            if (nameLower.includes('platform') ||
                nameLower.includes('disc') ||
                nameLower.includes('floor') ||
                nameLower.includes('stage') ||
                nameLower.includes('base') ||
                nameLower.includes('platter')) {

                platforms.push({ name, pos, type: 'PLATFORM_CANDIDATE' });
                console.log(`🎯 Found: "${name}" at (${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)})`);
            }
        }
    });

    if (platforms.length === 0) {
        console.log('❌ No platform candidates found.');
        console.log('Dumping ALL meshes in scene:');
        let count = 0;
        scene.traverse((child) => {
            if (child instanceof THREE.Mesh) {
                count++;
                console.log(`  [${count}] "${child.name || '(unnamed)'}" at (${child.position.x.toFixed(2)}, ${child.position.y.toFixed(2)}, ${child.position.z.toFixed(2)})`);
            }
        });
    } else {
        console.log(`\n✅ Found ${platforms.length} platform candidates.`);
    }

    return platforms;
};

// Test platform ride animation
window.testPlatformRide = function(duration = 5) {
    const scene = window.gameManager?.currentScene;
    if (!scene) {
        console.error('No scene found.');
        return;
    }

    // Find the platform
    const platforms = window.findPlatform();
    if (platforms.length === 0) {
        console.error('No platform found to test!');
        return;
    }

    const platformName = platforms[0].name;
    const platform = scene.getObjectByName(platformName);

    if (!platform) {
        console.error(`Could not get platform mesh: ${platformName}`);
        return;
    }

    console.log(`🎢 Testing platform ride on "${platformName}"`);
    console.log(`   Start: (${platform.position.x.toFixed(2)}, ${platform.position.y.toFixed(2)}, ${platform.position.z.toFixed(2)})`);

    // Vortex position (from Vortex_gate)
    const targetPos = new THREE.Vector3(0, 0.5, 2);
    console.log(`   Target: ${targetPos.toArray().map(v => v.toFixed(2))}`);

    // Animate
    const startPos = platform.position.clone();
    const startTime = Date.now();

    function animate() {
        const elapsed = (Date.now() - startTime) / 1000;
        const progress = Math.min(elapsed / duration, 1);

        // Easing
        const eased = progress < 0.5 ? 4 * progress * progress * progress : 1 - Math.pow(-2 * progress + 2, 3) / 2;

        // Arc height
        const arcOffset = Math.sin(eased * Math.PI) * 0.3;

        // Move
        platform.position.lerpVectors(startPos, targetPos, eased);
        platform.position.y += arcOffset;

        if (progress < 1) {
            requestAnimationFrame(animate);
        } else {
            console.log('✅ Platform ride complete!');
            console.log(`   End: (${platform.position.x.toFixed(2)}, ${platform.position.y.toFixed(2)}, ${platform.position.z.toFixed(2)})`);
        }
    }

    animate();
};

// Reset platform position
window.resetPlatform = function() {
    // TODO: Store original position when finding platform
    console.log('Reset platform - needs original position storage');
};

// Trigger duet complete (tests vortex activation → platform ride)
window.triggerDuetComplete = function() {
    console.log('🎵 Triggering duet complete...');

    // Dispatch duet progress event
    window.dispatchEvent(new CustomEvent('duet-progress', {
        detail: { progress: 1.0 }
    }));

    console.log('✅ Duet complete event dispatched.');
    console.log('   This should trigger vortex activation and platform ride when activation >= 0.99');
};

// Quick test: Find platform NOW
window.findPlatform();
