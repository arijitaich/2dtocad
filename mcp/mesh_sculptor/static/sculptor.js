/**
 * Mesh Sculptor - Cities: Skylines Style 3D Mesh Editing
 * 
 * Features:
 * - Raise/Lower terrain-like sculpting
 * - Flatten areas
 * - Smooth surfaces
 * - Configurable brush size, strength, falloff
 * - Undo/Redo history
 * - Export to OBJ/STL for wax printing
 */

// ============================================================
// DEBUG LOGGING
// ============================================================

function debugLog(message, type = 'info') {
    const logDiv = document.getElementById('debug-log');
    if (!logDiv) return;
    
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logDiv.appendChild(entry);
    logDiv.scrollTop = logDiv.scrollHeight;
    
    // Also log to console
    console.log(`[${type}] ${message}`);
}

function clearDebugLog() {
    const logDiv = document.getElementById('debug-log');
    if (logDiv) logDiv.innerHTML = '';
}

// ============================================================
// GLOBAL STATE
// ============================================================

let scene, camera, renderer, controls;
let mesh = null;
let meshData = {
    vertices: null,
    faces: null,
    normals: null,
    originalVertices: null
};

// Sculpting state
let currentTool = 'raise';
let brushSize = 2.0;
let brushStrength = 0.3;
let brushFalloff = 0.5;
let isSculpting = false;
let shiftPressed = false;
let meshScale = 1.0; // Scale factor based on mesh size

// Grid mode state
let gridMode = true; // Default to grid mode for matrix skin
let gridCellSize = 0.18; // Default 0.18mm for matrix skin (0.01 * 18 scale)
let gridBrushCells = 3; // How many grid cells the brush covers (3x3 default)
let gridSquareIndicator = null; // Visual indicator for grid brush

// Selection mode state
let isSelectMode = false;
let selectedCells = new Map(); // Map of "x,y" -> { bounds, vertices, visualMesh }
let selectionVisuals = []; // Array of visual meshes showing selection
let duplicateMesh = null; // The floating duplicate mesh
let isDraggingDuplicate = false; // Whether we're placing a duplicate
let duplicateVertices = null; // Vertices of the duplicate
let duplicateFaces = null; // Faces of the duplicate

// Elongate tool state
let isElongateMode = false;
let elongateStartY = 0; // Mouse Y position when elongate started
let elongateNormal = null; // The average normal direction for elongation
let elongateOriginalPositions = null; // Original vertex positions before elongate
let elongateSelectedVertices = null; // Vertices being elongated

// Carve block tool state
let isCarveMode = false;
let carveBlockMesh = null; // Visual 3D block for carving
let carveBlockWidth = 2.0;
let carveBlockHeight = 2.0;
let carveBlockDepth = 1.0;

// View state
let wireframeMode = false;
let showNormalsMode = false;
let normalsHelper = null;

// History for undo/redo
const MAX_HISTORY = 50;
let history = [];
let historyIndex = -1;

// Raycaster for brush
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let brushIndicator = null;

// ============================================================
// INITIALIZATION
// ============================================================

function init() {
    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    
    // Camera
    const canvas = document.getElementById('canvas');
    const aspect = canvas.clientWidth / canvas.clientHeight;
    camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
    camera.position.set(30, 20, 30);
    
    // Renderer
    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    
    // Set up our mouse event handlers FIRST (before OrbitControls)
    // Use pointerdown instead of mousedown for better compatibility
    canvas.addEventListener('pointerdown', onPointerDown, true);  // capture phase
    canvas.addEventListener('pointermove', onPointerMove, true);  // capture phase
    canvas.addEventListener('pointerup', onPointerUp, true);      // capture phase
    
    // Orbit controls - only use right mouse button for rotation
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.mouseButtons = {
        LEFT: null, // Disabled - we handle left-click for sculpting
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.ROTATE
    };
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 20, 10);
    scene.add(directionalLight);
    
    const backLight = new THREE.DirectionalLight(0x4ecdc4, 0.3);
    backLight.position.set(-10, -10, -10);
    scene.add(backLight);
    
    // Grid helper
    const gridHelper = new THREE.GridHelper(50, 50, 0x444444, 0x222222);
    scene.add(gridHelper);
    
    // Brush indicator - flat circular disc that follows surface
    const brushGroup = new THREE.Group();
    
    // Outer ring
    const ringGeometry = new THREE.RingGeometry(0.9, 1.0, 64);
    const ringMaterial = new THREE.MeshBasicMaterial({
        color: 0xe94560,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.8
    });
    const brushRing = new THREE.Mesh(ringGeometry, ringMaterial);
    brushGroup.add(brushRing);
    
    // Inner fill (semi-transparent)
    const fillGeometry = new THREE.CircleGeometry(0.9, 64);
    const fillMaterial = new THREE.MeshBasicMaterial({
        color: 0xe94560,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.15
    });
    const brushFill = new THREE.Mesh(fillGeometry, fillMaterial);
    brushGroup.add(brushFill);
    
    // Center dot
    const dotGeometry = new THREE.CircleGeometry(0.05, 16);
    const dotMaterial = new THREE.MeshBasicMaterial({
        color: 0xffffff,
        side: THREE.DoubleSide
    });
    const brushDot = new THREE.Mesh(dotGeometry, dotMaterial);
    brushDot.position.z = 0.001; // Slightly above to prevent z-fighting
    brushGroup.add(brushDot);
    
    brushIndicator = brushGroup;
    brushIndicator.visible = false;
    scene.add(brushIndicator);
    
    // Grid square brush indicator (for grid mode)
    createGridSquareIndicator();
    
    // Event listeners
    setupEventListeners();
    
    // Start render loop
    animate();
    
    // Hide loading
    hideLoading();
    
    debugLog('Mesh Sculptor initialized - Ready!', 'success');
}

function setupEventListeners() {
    const canvas = document.getElementById('canvas');
    
    debugLog(`Event listeners already set up in init()`, 'info');
    
    // Keyboard events (these can stay on document)
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    
    // Window resize
    window.addEventListener('resize', onWindowResize);
    
    // Prevent context menu on canvas
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
}

// New pointer event handlers (more reliable than mouse events)
function onPointerDown(event) {
    debugLog(`Pointer down! button=${event.button}, selectMode=${isSelectMode}, carveMode=${isCarveMode}, elongateMode=${isElongateMode}`, 'info');
    
    if (event.button === 0 && mesh) { // Left click
        updateMousePosition(event);
        
        // Handle elongate mode - drag to stretch
        if (isElongateMode) {
            elongateStartY = event.clientY;
            isSculpting = true; // Reuse for "is dragging"
            return;
        }
        
        // Handle duplicate placement FIRST
        if (isDraggingDuplicate) {
            placeDuplicate();
            return;
        }
        
        // Handle selection mode - MUST come before sculpting
        if (isSelectMode) {
            debugLog('Selection mode active - selecting cells', 'info');
            isSculpting = true; // Reuse this flag for "is selecting/dragging"
            
            // REMOVED auto-clear: Selection now persists until "Clear Selection" is clicked
            // This allows selecting across multiple view angles
            
            // Select at current position
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(mesh);
            if (intersects.length > 0) {
                selectCellsAtPoint(intersects[0].point);
            }
            setStatus('Selecting cells... drag to select more');
            return; // IMPORTANT: Don't continue to sculpting
        }
        
        // Handle carve mode
        if (isCarveMode) {
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(mesh);
            if (intersects.length > 0) {
                const hit = intersects[0];
                const normalMatrix = new THREE.Matrix3().getNormalMatrix(mesh.matrixWorld);
                const worldNormal = hit.face.normal.clone().applyMatrix3(normalMatrix).normalize();
                carveAtPoint(hit.point, worldNormal);
            }
            return;
        }
        
        // Normal sculpting (only if NOT in select mode and NOT in carve mode)
        isSculpting = true;
        debugLog('🎨 Starting sculpt session...', 'success');
        
        saveToHistory();
        performSculpt();
        setStatus(`Sculpting with ${currentTool}...`);
    }
}

function onPointerMove(event) {
    updateMousePosition(event);
    
    raycaster.setFromCamera(mouse, camera);
    
    // Update carve block position - always visible, free movement
    if (isCarveMode && carveBlockMesh) {
        const intersects = mesh ? raycaster.intersectObject(mesh) : [];
        if (intersects.length > 0) {
            // On mesh - position at hit point with surface normal orientation
            const hit = intersects[0];
            const normalMatrix = new THREE.Matrix3().getNormalMatrix(mesh.matrixWorld);
            const worldNormal = hit.face.normal.clone().applyMatrix3(normalMatrix).normalize();
            updateCarveBlockPosition(hit.point, worldNormal);
            carveBlockMesh.material.color.setHex(0xff4444); // Red when on mesh
        } else {
            // Off mesh - project to a plane at the mesh's Z level or default plane
            const planeZ = mesh ? mesh.position.z : 0;
            const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -planeZ);
            const intersectPoint = new THREE.Vector3();
            raycaster.ray.intersectPlane(plane, intersectPoint);
            if (intersectPoint) {
                updateCarveBlockPosition(intersectPoint, new THREE.Vector3(0, 0, 1));
            }
            carveBlockMesh.material.color.setHex(0x888888); // Gray when off mesh
        }
        carveBlockMesh.visible = true; // Always visible in carve mode
    }
    
    // Update brush indicator (for non-carve modes)
    if (!isCarveMode) {
        updateBrushIndicator();
    }
    
    // Handle elongate dragging - stretch along normal
    if (isElongateMode && isSculpting) {
        const deltaY = event.clientY - elongateStartY;
        updateElongate(deltaY);
        return;
    }
    
    // Update duplicate position if dragging - FREE 3D SPACE movement
    if (isDraggingDuplicate && duplicateMesh) {
        // Create a plane perpendicular to camera at the duplicate's current position
        const cameraDirection = new THREE.Vector3();
        camera.getWorldDirection(cameraDirection);
        const dragPlane = new THREE.Plane();
        dragPlane.setFromNormalAndCoplanarPoint(cameraDirection, duplicateMesh.position);
        
        // Find intersection with drag plane
        const intersectPoint = new THREE.Vector3();
        if (raycaster.ray.intersectPlane(dragPlane, intersectPoint)) {
            updateDuplicatePosition(intersectPoint);
        }
    }
    
    // Select while dragging in select mode
    if (isSelectMode && isSculpting && mesh) {
        const intersects = raycaster.intersectObject(mesh);
        if (intersects.length > 0) {
            selectCellsAtPoint(intersects[0].point);
        }
        // Don't continue to sculpting!
    }
    // Sculpt while dragging - ONLY if NOT in select mode, NOT in carve mode, NOT in elongate mode
    else if (isSculpting && mesh && !isSelectMode && !isCarveMode && !isElongateMode) {
        performSculpt();
    }
    
    // Update coordinates display
    if (mesh) {
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObject(mesh);
        if (intersects.length > 0) {
            const point = intersects[0].point;
            document.getElementById('status-coords').textContent = 
                `X: ${point.x.toFixed(2)} Y: ${point.y.toFixed(2)} Z: ${point.z.toFixed(2)}`;
        }
    }
}
function onPointerUp(event) {
    if (event.button === 0) {
        if (isSculpting) {
            debugLog('🎨 Sculpt session ended', 'info');
            setStatus('Ready');
        }
        isSculpting = false;
    }
}

function updateMousePosition(event) {
    const canvas = document.getElementById('canvas');
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

// ============================================================
// MOUSE/KEYBOARD EVENTS (kept for keyboard and wheel)
// ============================================================

function onMouseDown(event) {
    // Now handled by onPointerDown
}

function onMouseMove(event) {
    // Now handled by onPointerMove
}

function onMouseUp(event) {
    // Now handled by onPointerUp
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    
    // Window resize
    window.addEventListener('resize', onWindowResize);
    
    // Prevent context menu on canvas
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
}

// ============================================================
// MESH LOADING
// ============================================================

async function loadSampleMesh() {
    showLoading('Loading sample ring mesh...');
    
    try {
        const response = await fetch('/api/load-sample');
        const data = await response.json();
        
        if (data.success) {
            loadMeshData(data);
            setStatus(`Loaded: ${data.filename} (${data.info.vertex_count.toLocaleString()} vertices)`);
        } else {
            throw new Error(data.error || 'Failed to load');
        }
    } catch (error) {
        console.error('Error loading sample:', error);
        setStatus('Error loading sample mesh: ' + error.message);
    }
    
    hideLoading();
}

async function uploadFile(input) {
    if (!input.files || !input.files[0]) return;
    
    const file = input.files[0];
    showLoading(`Uploading ${file.name}...`);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.success) {
            loadMeshData(data);
            setStatus(`Uploaded: ${data.filename} (${data.info.vertex_count.toLocaleString()} vertices)`);
        } else {
            throw new Error(data.error || 'Upload failed');
        }
    } catch (error) {
        console.error('Error uploading:', error);
        setStatus('Error uploading file: ' + error.message);
    }
    
    hideLoading();
    input.value = ''; // Reset file input
}

function loadMeshData(data) {
    // Remove existing mesh
    if (mesh) {
        scene.remove(mesh);
        mesh.geometry.dispose();
        mesh.material.dispose();
    }
    
    // Store mesh data
    meshData.vertices = new Float32Array(data.mesh_data.vertices.flat());
    meshData.faces = data.mesh_data.faces;
    meshData.normals = new Float32Array(data.mesh_data.normals.flat());
    meshData.originalVertices = meshData.vertices.slice(); // Copy for reset
    
    // Create Three.js geometry
    const geometry = new THREE.BufferGeometry();
    
    // Set vertices
    geometry.setAttribute('position', new THREE.BufferAttribute(meshData.vertices, 3));
    
    // Set indices (faces)
    const indices = [];
    for (const face of meshData.faces) {
        indices.push(face[0], face[1], face[2]);
    }
    geometry.setIndex(indices);
    
    // Set normals
    geometry.setAttribute('normal', new THREE.BufferAttribute(meshData.normals, 3));
    
    // Material
    const material = new THREE.MeshStandardMaterial({
        color: 0xd4af37, // Gold color
        metalness: 0.8,
        roughness: 0.3,
        side: THREE.DoubleSide,
        flatShading: false
    });
    
    // Create mesh
    mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);
    
    // Center and fit camera
    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    const center = new THREE.Vector3();
    box.getCenter(center);
    
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    
    // Calculate mesh scale for brush sizing
    meshScale = maxDim;
    debugLog(`Mesh loaded - Scale: ${meshScale.toFixed(2)}mm`, 'success');
    debugLog(`Vertices: ${data.mesh_data.vertices.length}, Faces: ${data.mesh_data.faces.length}`, 'info');
    
    // Auto-detect grid cell size from mesh
    detectGridCellSize();
    
    // Set brush size relative to mesh (default to 10% of mesh size)
    const defaultBrushSize = maxDim * 0.1;
    updateBrushSize(defaultBrushSize);
    document.getElementById('brush-size').value = defaultBrushSize;
    document.getElementById('brush-size').max = maxDim * 0.5;
    document.getElementById('brush-size').min = maxDim * 0.01;
    document.getElementById('brush-size').step = maxDim * 0.01;
    
    // Position mesh at origin
    mesh.position.sub(center);
    
    // Adjust camera
    camera.position.set(maxDim * 2, maxDim * 1.5, maxDim * 2);
    controls.target.set(0, 0, 0);
    controls.update();
    
    // Update info panel
    updateMeshInfo(data.info);
    
    // Reset history
    history = [];
    historyIndex = -1;
    saveToHistory();
}

// ============================================================
// GRID MODE FUNCTIONS
// ============================================================

function createGridSquareIndicator() {
    // Create a square outline for grid mode brush
    const geometry = new THREE.BufferGeometry();
    
    // Square outline vertices (will be scaled based on grid cells)
    const size = 1;
    const vertices = new Float32Array([
        -size, -size, 0,  size, -size, 0,  // Bottom edge
        size, -size, 0,   size, size, 0,   // Right edge
        size, size, 0,    -size, size, 0,  // Top edge
        -size, size, 0,   -size, -size, 0  // Left edge
    ]);
    
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    
    const material = new THREE.LineBasicMaterial({
        color: 0x00ff88,
        linewidth: 2,
        transparent: true,
        opacity: 0.9
    });
    
    gridSquareIndicator = new THREE.LineSegments(geometry, material);
    gridSquareIndicator.visible = false;
    scene.add(gridSquareIndicator);
}

function detectGridCellSize() {
    if (!meshData.vertices || meshData.vertices.length < 12) return;
    
    // Sample vertices to find typical spacing (grid cell size)
    // Look at distances between nearby vertices
    const vertices = meshData.vertices;
    const vertexCount = vertices.length / 3;
    
    // Sample first 1000 vertices and find minimum non-zero distances
    const distances = [];
    const sampleSize = Math.min(1000, vertexCount);
    
    for (let i = 0; i < sampleSize; i++) {
        const v1x = vertices[i * 3];
        const v1y = vertices[i * 3 + 1];
        const v1z = vertices[i * 3 + 2];
        
        // Compare with next few vertices
        for (let j = i + 1; j < Math.min(i + 50, sampleSize); j++) {
            const v2x = vertices[j * 3];
            const v2y = vertices[j * 3 + 1];
            const v2z = vertices[j * 3 + 2];
            
            const dx = v2x - v1x;
            const dy = v2y - v1y;
            const dz = v2z - v1z;
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
            
            // Only consider small distances (likely grid edges)
            if (dist > 0.001 && dist < meshScale * 0.1) {
                distances.push(dist);
            }
        }
    }
    
    if (distances.length > 0) {
        // Sort and take the median of smaller distances
        distances.sort((a, b) => a - b);
        const medianIdx = Math.floor(distances.length * 0.2); // Take 20th percentile
        gridCellSize = distances[medianIdx];
        
        debugLog(`Grid cell size detected: ${gridCellSize.toFixed(4)}mm`, 'success');
        
        // Update grid brush size slider
        updateGridBrushUI();
        updateCellSizeUI();
    } else {
        // For matrix skin meshes, use 0.18mm as default (0.01 * 18 scale factor)
        gridCellSize = 0.18;
        debugLog(`Grid cell size default: ${gridCellSize.toFixed(4)}mm`, 'warn');
        updateCellSizeUI();
    }
}

function updateCellSize(value) {
    gridCellSize = parseFloat(value);
    document.getElementById('cell-size-value').textContent = gridCellSize.toFixed(2);
    debugLog(`Cell size set to: ${gridCellSize.toFixed(3)}mm`, 'info');
    
    // Update visual indicators if any
    updateBrushIndicator();
}

function updateCellSizeUI() {
    const cellSizeSlider = document.getElementById('cell-size');
    const cellSizeValue = document.getElementById('cell-size-value');
    if (cellSizeSlider) {
        cellSizeSlider.value = gridCellSize;
    }
    if (cellSizeValue) {
        cellSizeValue.textContent = gridCellSize.toFixed(2);
    }
}

function updateGridBrushUI() {
    const gridSizeSlider = document.getElementById('grid-brush-size');
    if (gridSizeSlider) {
        gridSizeSlider.value = gridBrushCells;
        const gridSizeValue = document.getElementById('grid-brush-size-value');
        if (gridSizeValue) {
            gridSizeValue.textContent = `${gridBrushCells}x${gridBrushCells}`;
        }
    }
}

function toggleGridMode() {
    gridMode = !gridMode;
    
    // Update UI button
    const gridModeBtn = document.getElementById('grid-mode-btn');
    if (gridModeBtn) {
        gridModeBtn.classList.toggle('active', gridMode);
        gridModeBtn.textContent = gridMode ? '▦ Grid: ON' : '○ Grid: OFF';
    }
    
    // Show/hide appropriate controls
    const gridSizeGroup = document.getElementById('grid-size-group');
    const circularSizeGroup = document.getElementById('circular-size-group');
    const falloffGroup = document.getElementById('falloff-group');
    
    if (gridSizeGroup) gridSizeGroup.style.display = gridMode ? 'block' : 'none';
    if (circularSizeGroup) circularSizeGroup.style.display = gridMode ? 'none' : 'block';
    if (falloffGroup) falloffGroup.style.display = gridMode ? 'none' : 'block';
    
    // Show/hide appropriate brush indicator
    if (brushIndicator) brushIndicator.visible = false;
    if (gridSquareIndicator) gridSquareIndicator.visible = false;
    
    debugLog(`Grid mode: ${gridMode ? 'ON (square brush)' : 'OFF (circular brush)'}`, 'info');
    setStatus(gridMode ? 'Grid Mode - Square brush aligned to matrix cells' : 'Free Mode - Circular brush');
}

function updateGridBrushSize(value) {
    gridBrushCells = parseInt(value);
    const gridSizeValue = document.getElementById('grid-brush-size-value');
    if (gridSizeValue) {
        gridSizeValue.textContent = `${gridBrushCells}x${gridBrushCells}`;
    }
    debugLog(`Grid brush: ${gridBrushCells}x${gridBrushCells} cells`, 'info');
}

function getGridAlignedBounds(point) {
    // Align the brush to grid cells
    const halfSize = gridCellSize * gridBrushCells / 2;
    
    // Snap to nearest grid cell center
    const snappedX = Math.round(point.x / gridCellSize) * gridCellSize;
    const snappedY = Math.round(point.y / gridCellSize) * gridCellSize;
    const snappedZ = Math.round(point.z / gridCellSize) * gridCellSize;
    
    return {
        center: { x: snappedX, y: snappedY, z: snappedZ },
        min: { 
            x: snappedX - halfSize, 
            y: snappedY - halfSize, 
            z: snappedZ - halfSize 
        },
        max: { 
            x: snappedX + halfSize, 
            y: snappedY + halfSize, 
            z: snappedZ + halfSize 
        },
        size: gridCellSize * gridBrushCells
    };
}

// ============================================================
// SELECTION & DUPLICATE TOOLS
// ============================================================

function toggleSelectMode() {
    isSelectMode = !isSelectMode;
    
    // Turn off carve mode if entering select mode
    if (isSelectMode && isCarveMode) {
        isCarveMode = false;
        if (carveBlockMesh) carveBlockMesh.visible = false;
        const carveBtn = document.getElementById('carve-mode-btn');
        if (carveBtn) {
            carveBtn.classList.remove('active');
            carveBtn.textContent = '🔲 Carve Block';
        }
        const carveControls = document.getElementById('carve-controls');
        if (carveControls) carveControls.style.display = 'none';
    }
    
    // Update UI
    const selectBtn = document.getElementById('select-mode-btn');
    if (selectBtn) {
        selectBtn.classList.toggle('active', isSelectMode);
        selectBtn.textContent = isSelectMode ? '✓ Select Mode ON' : '☐ Select Mode';
    }
    
    // Show/hide duplicate button based on selection
    updateDuplicateButtonVisibility();
    
    if (isSelectMode) {
        debugLog(`Select mode ON - Drag to select ${gridBrushCells}x${gridBrushCells} cell areas`, 'info');
        setStatus(`Select Mode - Drag to select cells (${gridBrushCells}x${gridBrushCells} brush)`);
    } else {
        debugLog('Select mode OFF', 'info');
        setStatus('Ready');
    }
}

function selectCellsAtPoint(point) {
    if (!mesh || !meshData.vertices || !meshData.faces) return;
    
    const localPoint = point.clone().sub(mesh.position);
    
    // Raycast to get surface normal at this point
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(mesh);
    let surfaceNormal = new THREE.Vector3(0, 0, 1);
    let hitFaceIndex = -1;
    if (intersects.length > 0) {
        const normalMatrix = new THREE.Matrix3().getNormalMatrix(mesh.matrixWorld);
        surfaceNormal = intersects[0].face.normal.clone().applyMatrix3(normalMatrix).normalize();
        hitFaceIndex = intersects[0].faceIndex;
    }
    
    const brushRadius = gridCellSize * gridBrushCells / 2;
    const vertices = meshData.vertices;
    const faces = meshData.faces;
    
    // Find all FACES within brush radius (selecting actual triangles, not arbitrary grid)
    const selectedFaceIndices = new Set();
    
    for (let faceIdx = 0; faceIdx < faces.length; faceIdx++) {
        const face = faces[faceIdx];
        
        // Calculate face center
        let cx = 0, cy = 0, cz = 0;
        for (const vertIdx of face) {
            cx += vertices[vertIdx * 3];
            cy += vertices[vertIdx * 3 + 1];
            cz += vertices[vertIdx * 3 + 2];
        }
        cx /= face.length;
        cy /= face.length;
        cz /= face.length;
        
        // Check distance from click point to face center
        const dx = cx - localPoint.x;
        const dy = cy - localPoint.y;
        const dz = cz - localPoint.z;
        const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
        
        if (dist <= brushRadius) {
            selectedFaceIndices.add(faceIdx);
        }
    }
    
    if (selectedFaceIndices.size === 0) return;
    
    // Create a unique key for this selection region based on the faces
    // Group connected faces together
    const faceGroups = groupConnectedFaces(selectedFaceIndices, faces);
    
    for (const faceGroup of faceGroups) {
        // Create cell key from the face indices (sorted for consistency)
        const sortedFaces = [...faceGroup].sort((a, b) => a - b);
        const cellKey = `faces_${sortedFaces.slice(0, 5).join('_')}_${sortedFaces.length}`;
        
        // Skip if already selected
        if (selectedCells.has(cellKey)) continue;
        
        // Collect all vertices from these faces
        const cellVertices = [];
        const vertexSet = new Set();
        for (const faceIdx of faceGroup) {
            for (const vertIdx of faces[faceIdx]) {
                if (!vertexSet.has(vertIdx)) {
                    vertexSet.add(vertIdx);
                    cellVertices.push({
                        index: vertIdx,
                        x: vertices[vertIdx * 3],
                        y: vertices[vertIdx * 3 + 1],
                        z: vertices[vertIdx * 3 + 2]
                    });
                }
            }
        }
        
        if (cellVertices.length > 0) {
            // Calculate center
            let avgX = 0, avgY = 0, avgZ = 0;
            for (const v of cellVertices) {
                avgX += v.x;
                avgY += v.y;
                avgZ += v.z;
            }
            avgX /= cellVertices.length;
            avgY /= cellVertices.length;
            avgZ /= cellVertices.length;
            
            const cellBounds = {
                center: { x: avgX, y: avgY, z: avgZ },
                size: gridCellSize
            };
            
            // Create visual highlight from actual faces
            const visualMesh = createFaceSelectionVisual(faceGroup, faces, vertices, surfaceNormal);
            
            selectedCells.set(cellKey, {
                bounds: cellBounds,
                vertices: cellVertices,
                faces: faceGroup,
                visualMesh: visualMesh,
                worldCenter: new THREE.Vector3(avgX + mesh.position.x, avgY + mesh.position.y, avgZ + mesh.position.z)
            });
        }
    }
    
    updateDuplicateButtonVisibility();
    updateSelectionInfo();
}

function groupConnectedFaces(faceIndices, allFaces) {
    // Group faces that share vertices (connected faces)
    const faceArray = [...faceIndices];
    const groups = [];
    const visited = new Set();
    
    for (const startFace of faceArray) {
        if (visited.has(startFace)) continue;
        
        // BFS to find all connected faces
        const group = new Set();
        const queue = [startFace];
        
        while (queue.length > 0) {
            const faceIdx = queue.shift();
            if (visited.has(faceIdx)) continue;
            visited.add(faceIdx);
            group.add(faceIdx);
            
            // Find adjacent faces (share at least one vertex)
            const faceVerts = new Set(allFaces[faceIdx]);
            for (const otherFace of faceArray) {
                if (visited.has(otherFace)) continue;
                // Check if faces share a vertex
                for (const v of allFaces[otherFace]) {
                    if (faceVerts.has(v)) {
                        queue.push(otherFace);
                        break;
                    }
                }
            }
        }
        
        groups.push(group);
    }
    
    return groups;
}

function createFaceSelectionVisual(faceGroup, allFaces, vertices, surfaceNormal) {
    // Create a mesh from the actual selected faces
    const positions = [];
    const indices = [];
    const vertexMap = new Map(); // oldIndex -> newIndex
    
    for (const faceIdx of faceGroup) {
        const face = allFaces[faceIdx];
        const newFaceIndices = [];
        
        for (const vertIdx of face) {
            if (!vertexMap.has(vertIdx)) {
                vertexMap.set(vertIdx, positions.length / 3);
                // Offset slightly along normal to prevent z-fighting
                positions.push(
                    vertices[vertIdx * 3] + mesh.position.x + surfaceNormal.x * 0.02,
                    vertices[vertIdx * 3 + 1] + mesh.position.y + surfaceNormal.y * 0.02,
                    vertices[vertIdx * 3 + 2] + mesh.position.z + surfaceNormal.z * 0.02
                );
            }
            newFaceIndices.push(vertexMap.get(vertIdx));
        }
        
        indices.push(newFaceIndices[0], newFaceIndices[1], newFaceIndices[2]);
    }
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    
    const material = new THREE.MeshBasicMaterial({
        color: 0x00aaff,
        transparent: true,
        opacity: 0.5,
        side: THREE.DoubleSide,
        depthTest: false
    });
    
    const visual = new THREE.Mesh(geometry, material);
    
    // Add wireframe outline
    const edges = new THREE.EdgesGeometry(geometry);
    const lineMaterial = new THREE.LineBasicMaterial({ color: 0x00ffff, linewidth: 2 });
    const wireframe = new THREE.LineSegments(edges, lineMaterial);
    visual.add(wireframe);
    
    scene.add(visual);
    selectionVisuals.push(visual);
    
    return visual;
}

function selectCellAtPoint(point) {
    // Deprecated - now using selectCellsAtPoint
    selectCellsAtPoint(point);
}

function createSelectionVisual(cellBounds, worldPoint) {
    // Legacy function - redirect to 3D version with default normal
    return createSelectionVisual3D(cellBounds, new THREE.Vector3(0, 0, 1));
}

function clearSelection() {
    // Remove all visual meshes
    for (const [key, cellData] of selectedCells) {
        if (cellData.visualMesh) {
            scene.remove(cellData.visualMesh);
            cellData.visualMesh.geometry.dispose();
            cellData.visualMesh.material.dispose();
        }
    }
    selectedCells.clear();
    selectionVisuals = [];
    
    debugLog('Selection cleared', 'info');
    updateDuplicateButtonVisibility();
    updateSelectionInfo();
}

function updateDuplicateButtonVisibility() {
    const dupBtn = document.getElementById('duplicate-btn');
    const elongateBtn = document.getElementById('elongate-btn');
    if (dupBtn) {
        dupBtn.style.display = selectedCells.size > 0 ? 'block' : 'none';
    }
    if (elongateBtn) {
        elongateBtn.style.display = selectedCells.size > 0 ? 'block' : 'none';
    }
}

function updateSelectionInfo() {
    const infoSpan = document.getElementById('selection-info');
    if (infoSpan) {
        if (selectedCells.size > 0) {
            let totalVerts = 0;
            for (const [key, cellData] of selectedCells) {
                totalVerts += cellData.vertices.length;
            }
            infoSpan.textContent = `${selectedCells.size} cells (${totalVerts} vertices)`;
        } else {
            infoSpan.textContent = 'None';
        }
    }
}

// ============================================================
// ELONGATE TOOL - Stretch selection along surface normal
// ============================================================

function startElongate() {
    if (selectedCells.size === 0) {
        debugLog('Nothing selected to elongate', 'warn');
        return;
    }
    
    // Calculate average normal from all selected faces
    const vertices = meshData.vertices;
    const faces = meshData.faces;
    
    let avgNormal = new THREE.Vector3(0, 0, 0);
    let faceCount = 0;
    
    // Collect all selected vertices and calculate average normal
    elongateSelectedVertices = new Set();
    
    for (const [key, cellData] of selectedCells) {
        // Add vertices
        for (const v of cellData.vertices) {
            elongateSelectedVertices.add(v.index);
        }
        
        // Calculate face normals if faces are stored
        if (cellData.faces) {
            for (const faceIdx of cellData.faces) {
                const face = faces[faceIdx];
                const v0 = new THREE.Vector3(vertices[face[0]*3], vertices[face[0]*3+1], vertices[face[0]*3+2]);
                const v1 = new THREE.Vector3(vertices[face[1]*3], vertices[face[1]*3+1], vertices[face[1]*3+2]);
                const v2 = new THREE.Vector3(vertices[face[2]*3], vertices[face[2]*3+1], vertices[face[2]*3+2]);
                
                const edge1 = v1.clone().sub(v0);
                const edge2 = v2.clone().sub(v0);
                const faceNormal = edge1.cross(edge2).normalize();
                
                avgNormal.add(faceNormal);
                faceCount++;
            }
        }
    }
    
    if (faceCount > 0) {
        avgNormal.divideScalar(faceCount).normalize();
    } else {
        // Fallback: use Z-up if no faces
        avgNormal.set(0, 0, 1);
    }
    
    elongateNormal = avgNormal;
    
    // Save original positions
    elongateOriginalPositions = new Map();
    for (const vertIdx of elongateSelectedVertices) {
        elongateOriginalPositions.set(vertIdx, {
            x: vertices[vertIdx * 3],
            y: vertices[vertIdx * 3 + 1],
            z: vertices[vertIdx * 3 + 2]
        });
    }
    
    // Save to history for undo
    saveToHistory();
    
    // Enter elongate mode
    isElongateMode = true;
    isSelectMode = false;
    
    // Update UI
    document.getElementById('elongate-controls').style.display = 'block';
    document.getElementById('elongate-btn').style.display = 'none';
    document.getElementById('duplicate-btn').style.display = 'none';
    
    const selectBtn = document.getElementById('select-mode-btn');
    if (selectBtn) {
        selectBtn.classList.remove('active');
        selectBtn.textContent = '☐ Select Mode';
    }
    
    debugLog(`Elongate mode: ${elongateSelectedVertices.size} vertices, normal: (${avgNormal.x.toFixed(2)}, ${avgNormal.y.toFixed(2)}, ${avgNormal.z.toFixed(2)})`, 'info');
    setStatus('Drag up/down to elongate along surface normal');
}

function updateElongate(deltaY) {
    if (!isElongateMode || !elongateNormal || !elongateOriginalPositions) return;
    
    const vertices = meshData.vertices;
    
    // Scale factor: negative deltaY = elongate outward (mouse up = stretch)
    const elongateAmount = -deltaY * 0.01 * meshScale;
    
    // Move each vertex along the normal direction
    for (const [vertIdx, origPos] of elongateOriginalPositions) {
        vertices[vertIdx * 3] = origPos.x + elongateNormal.x * elongateAmount;
        vertices[vertIdx * 3 + 1] = origPos.y + elongateNormal.y * elongateAmount;
        vertices[vertIdx * 3 + 2] = origPos.z + elongateNormal.z * elongateAmount;
    }
    
    // Update geometry
    mesh.geometry.attributes.position.needsUpdate = true;
    mesh.geometry.computeVertexNormals();
    
    // Update selection visuals
    updateSelectionVisualsPosition();
}

function updateSelectionVisualsPosition() {
    // Update the visual highlights to match the new vertex positions
    const vertices = meshData.vertices;
    
    for (const [key, cellData] of selectedCells) {
        if (cellData.visualMesh && cellData.vertices) {
            const positions = cellData.visualMesh.geometry.attributes.position.array;
            
            // Recalculate center
            let avgX = 0, avgY = 0, avgZ = 0;
            for (const v of cellData.vertices) {
                avgX += vertices[v.index * 3];
                avgY += vertices[v.index * 3 + 1];
                avgZ += vertices[v.index * 3 + 2];
            }
            avgX /= cellData.vertices.length;
            avgY /= cellData.vertices.length;
            avgZ /= cellData.vertices.length;
            
            // Update visual mesh position
            // The visual was created with positions, need to rebuild
            // For simplicity, just update the mesh position based on average movement
            const offset = elongateNormal.clone().multiplyScalar(-elongateStartY * 0.01 * meshScale);
            
            // Update each vertex in the visual
            for (let i = 0; i < positions.length / 3; i++) {
                positions[i * 3] += elongateNormal.x * 0.01;
                positions[i * 3 + 1] += elongateNormal.y * 0.01;
                positions[i * 3 + 2] += elongateNormal.z * 0.01;
            }
            cellData.visualMesh.geometry.attributes.position.needsUpdate = true;
        }
    }
}

function confirmElongate() {
    if (!isElongateMode) return;
    
    debugLog('Elongation applied', 'success');
    
    // Exit elongate mode
    isElongateMode = false;
    elongateNormal = null;
    elongateOriginalPositions = null;
    elongateSelectedVertices = null;
    elongateStartY = 0;
    
    // Clear selection and rebuild visuals
    clearSelection();
    
    // Update UI
    document.getElementById('elongate-controls').style.display = 'none';
    
    setStatus('Elongation applied');
}

function cancelElongate() {
    if (!isElongateMode || !elongateOriginalPositions) return;
    
    // Restore original positions
    const vertices = meshData.vertices;
    for (const [vertIdx, origPos] of elongateOriginalPositions) {
        vertices[vertIdx * 3] = origPos.x;
        vertices[vertIdx * 3 + 1] = origPos.y;
        vertices[vertIdx * 3 + 2] = origPos.z;
    }
    
    // Update geometry
    mesh.geometry.attributes.position.needsUpdate = true;
    mesh.geometry.computeVertexNormals();
    
    debugLog('Elongation cancelled', 'info');
    
    // Exit elongate mode
    isElongateMode = false;
    elongateNormal = null;
    elongateOriginalPositions = null;
    elongateSelectedVertices = null;
    elongateStartY = 0;
    
    // Update UI
    document.getElementById('elongate-controls').style.display = 'none';
    updateDuplicateButtonVisibility();
    
    setStatus('Elongation cancelled');
}

// ============================================================
// STRETCH/OVAL TOOL - Scale mesh along X, Y, Z axes
// ============================================================

let stretchOriginalPositions = null; // Store original positions before stretching
let stretchCenter = null; // Center point for stretching

function updateStretchPreview() {
    if (!mesh || !meshData.vertices) return;
    
    // Store original positions on first stretch interaction
    if (!stretchOriginalPositions) {
        stretchOriginalPositions = new Float32Array(meshData.vertices.length);
        stretchOriginalPositions.set(meshData.vertices);
        
        // Calculate center of mesh
        const vertices = meshData.vertices;
        const vertexCount = vertices.length / 3;
        let cx = 0, cy = 0, cz = 0;
        for (let i = 0; i < vertexCount; i++) {
            cx += vertices[i * 3];
            cy += vertices[i * 3 + 1];
            cz += vertices[i * 3 + 2];
        }
        stretchCenter = {
            x: cx / vertexCount,
            y: cy / vertexCount,
            z: cz / vertexCount
        };
    }
    
    // Get scale values from sliders
    const scaleX = parseFloat(document.getElementById('stretch-x').value) / 100;
    const scaleY = parseFloat(document.getElementById('stretch-y').value) / 100;
    const scaleZ = parseFloat(document.getElementById('stretch-z').value) / 100;
    
    // Update display values
    document.getElementById('stretch-x-value').textContent = `${Math.round(scaleX * 100)}%`;
    document.getElementById('stretch-y-value').textContent = `${Math.round(scaleY * 100)}%`;
    document.getElementById('stretch-z-value').textContent = `${Math.round(scaleZ * 100)}%`;
    
    // Apply stretch to all vertices (scale around center)
    const vertices = meshData.vertices;
    const vertexCount = vertices.length / 3;
    
    for (let i = 0; i < vertexCount; i++) {
        // Get original position relative to center
        const ox = stretchOriginalPositions[i * 3] - stretchCenter.x;
        const oy = stretchOriginalPositions[i * 3 + 1] - stretchCenter.y;
        const oz = stretchOriginalPositions[i * 3 + 2] - stretchCenter.z;
        
        // Scale and add back center
        vertices[i * 3] = ox * scaleX + stretchCenter.x;
        vertices[i * 3 + 1] = oy * scaleY + stretchCenter.y;
        vertices[i * 3 + 2] = oz * scaleZ + stretchCenter.z;
    }
    
    // Update geometry
    mesh.geometry.attributes.position.needsUpdate = true;
    mesh.geometry.computeVertexNormals();
    mesh.geometry.computeBoundingBox();
    
    debugLog(`Stretch preview: X=${scaleX.toFixed(2)} Y=${scaleY.toFixed(2)} Z=${scaleZ.toFixed(2)}`, 'info');
}

function applyStretch() {
    if (!stretchOriginalPositions) {
        debugLog('No stretch to apply', 'warn');
        return;
    }
    
    // Save to history before finalizing
    saveToHistory();
    
    // Clear the stored original positions (makes current stretch permanent)
    stretchOriginalPositions = null;
    stretchCenter = null;
    
    // Reset sliders to 100%
    document.getElementById('stretch-x').value = 100;
    document.getElementById('stretch-y').value = 100;
    document.getElementById('stretch-z').value = 100;
    document.getElementById('stretch-x-value').textContent = '100%';
    document.getElementById('stretch-y-value').textContent = '100%';
    document.getElementById('stretch-z-value').textContent = '100%';
    
    // Update mesh info display
    updateMeshInfo();
    
    debugLog('Stretch applied permanently', 'success');
    setStatus('Stretch applied');
}

function resetStretch() {
    if (!stretchOriginalPositions) {
        debugLog('No stretch to reset', 'info');
        return;
    }
    
    // Restore original positions
    meshData.vertices.set(stretchOriginalPositions);
    
    // Update geometry
    mesh.geometry.attributes.position.needsUpdate = true;
    mesh.geometry.computeVertexNormals();
    mesh.geometry.computeBoundingBox();
    
    // Clear stored positions
    stretchOriginalPositions = null;
    stretchCenter = null;
    
    // Reset sliders to 100%
    document.getElementById('stretch-x').value = 100;
    document.getElementById('stretch-y').value = 100;
    document.getElementById('stretch-z').value = 100;
    document.getElementById('stretch-x-value').textContent = '100%';
    document.getElementById('stretch-y-value').textContent = '100%';
    document.getElementById('stretch-z-value').textContent = '100%';
    
    debugLog('Stretch reset to original', 'info');
    setStatus('Stretch reset');
}

function duplicateSelection() {
    if (selectedCells.size === 0) {
        debugLog('Nothing selected to duplicate', 'warn');
        return;
    }
    
    // Collect all selected vertices
    const allVertices = [];
    const vertexIndexMap = new Map(); // old index -> new index
    
    for (const [key, cellData] of selectedCells) {
        for (const v of cellData.vertices) {
            if (!vertexIndexMap.has(v.index)) {
                vertexIndexMap.set(v.index, allVertices.length);
                allVertices.push({ x: v.x, y: v.y, z: v.z, oldIndex: v.index });
            }
        }
    }
    
    if (allVertices.length === 0) {
        debugLog('No vertices in selection', 'warn');
        return;
    }
    
    // Find faces that use only selected vertices
    const selectedFaces = [];
    for (const face of meshData.faces) {
        const hasAll = face.every(idx => vertexIndexMap.has(idx));
        if (hasAll) {
            selectedFaces.push([
                vertexIndexMap.get(face[0]),
                vertexIndexMap.get(face[1]),
                vertexIndexMap.get(face[2])
            ]);
        }
    }
    
    debugLog(`Creating duplicate: ${allVertices.length} vertices, ${selectedFaces.length} faces`, 'info');
    
    // Calculate center of selection for offset
    let centerX = 0, centerY = 0, centerZ = 0;
    for (const v of allVertices) {
        centerX += v.x;
        centerY += v.y;
        centerZ += v.z;
    }
    centerX /= allVertices.length;
    centerY /= allVertices.length;
    centerZ /= allVertices.length;
    
    // Create geometry for duplicate
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(allVertices.length * 3);
    
    for (let i = 0; i < allVertices.length; i++) {
        positions[i * 3] = allVertices[i].x - centerX;
        positions[i * 3 + 1] = allVertices[i].y - centerY;
        positions[i * 3 + 2] = allVertices[i].z - centerZ;
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    if (selectedFaces.length > 0) {
        const indices = [];
        for (const face of selectedFaces) {
            indices.push(face[0], face[1], face[2]);
        }
        geometry.setIndex(indices);
    }
    
    geometry.computeVertexNormals();
    
    // Create mesh with semi-transparent material
    const material = new THREE.MeshStandardMaterial({
        color: 0x44ff44,
        transparent: true,
        opacity: 0.7,
        side: THREE.DoubleSide
    });
    
    // Remove old duplicate if exists
    if (duplicateMesh) {
        scene.remove(duplicateMesh);
        duplicateMesh.geometry.dispose();
        duplicateMesh.material.dispose();
    }
    
    duplicateMesh = new THREE.Mesh(geometry, material);
    duplicateMesh.position.set(centerX + mesh.position.x, centerY + mesh.position.y, centerZ + mesh.position.z + 1);
    scene.add(duplicateMesh);
    
    // Store vertices and faces for placement
    duplicateVertices = allVertices;
    duplicateFaces = selectedFaces;
    
    // Enter drag mode
    isDraggingDuplicate = true;
    
    debugLog('Duplicate created! Drag freely in 3D space, click to place.', 'success');
    setStatus('Drag in 3D space (scroll to move forward/back), Click to place');
    
    // Update UI
    const placeBtn = document.getElementById('place-duplicate-btn');
    if (placeBtn) placeBtn.style.display = 'block';
    const cancelBtn = document.getElementById('cancel-duplicate-btn');
    if (cancelBtn) cancelBtn.style.display = 'block';
}

function updateDuplicatePosition(point) {
    if (!duplicateMesh || !isDraggingDuplicate) return;
    
    // Position duplicate at cursor - free 3D space
    duplicateMesh.position.copy(point);
}

function placeDuplicate() {
    if (!duplicateMesh || !duplicateVertices || duplicateVertices.length === 0) {
        debugLog('No duplicate to place', 'warn');
        return;
    }
    
    // Get the placement position in local mesh coordinates
    const placementPos = duplicateMesh.position.clone().sub(mesh.position);
    
    // Add the duplicate vertices to the main mesh
    const oldVertexCount = meshData.vertices.length / 3;
    const newVertexCount = oldVertexCount + duplicateVertices.length;
    
    // Create new vertex array
    const newVertices = new Float32Array(newVertexCount * 3);
    newVertices.set(meshData.vertices);
    
    // Add duplicate vertices at placement position
    for (let i = 0; i < duplicateVertices.length; i++) {
        const v = duplicateVertices[i];
        newVertices[(oldVertexCount + i) * 3] = placementPos.x + (v.x - duplicateVertices[0].x);
        newVertices[(oldVertexCount + i) * 3 + 1] = placementPos.y + (v.y - duplicateVertices[0].y);
        newVertices[(oldVertexCount + i) * 3 + 2] = placementPos.z + (v.z - duplicateVertices[0].z);
    }
    
    // Add duplicate faces with offset indices
    const newFaces = [...meshData.faces];
    for (const face of duplicateFaces) {
        newFaces.push([
            face[0] + oldVertexCount,
            face[1] + oldVertexCount,
            face[2] + oldVertexCount
        ]);
    }
    
    // Update mesh data
    meshData.vertices = newVertices;
    meshData.faces = newFaces;
    
    // Rebuild the mesh geometry
    const geometry = mesh.geometry;
    geometry.setAttribute('position', new THREE.BufferAttribute(newVertices, 3));
    
    const indices = [];
    for (const face of newFaces) {
        indices.push(face[0], face[1], face[2]);
    }
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    
    debugLog(`Placed duplicate! Mesh now has ${newVertexCount} vertices`, 'success');
    
    // Clean up
    cancelDuplicate();
    saveToHistory();
    
    // Update info
    document.getElementById('info-vertices').textContent = newVertexCount.toLocaleString();
    document.getElementById('info-faces').textContent = newFaces.length.toLocaleString();
}

function cancelDuplicate() {
    if (duplicateMesh) {
        scene.remove(duplicateMesh);
        duplicateMesh.geometry.dispose();
        duplicateMesh.material.dispose();
        duplicateMesh = null;
    }
    
    duplicateVertices = null;
    duplicateFaces = null;
    isDraggingDuplicate = false;
    
    // Hide buttons
    const placeBtn = document.getElementById('place-duplicate-btn');
    if (placeBtn) placeBtn.style.display = 'none';
    const cancelBtn = document.getElementById('cancel-duplicate-btn');
    if (cancelBtn) cancelBtn.style.display = 'none';
    
    setStatus('Ready');
}

// ============================================================
// CARVE BLOCK TOOL
// ============================================================

function createCarveBlockMesh() {
    // Create a 3D rectangular box for carving preview
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshBasicMaterial({
        color: 0xff4444,
        transparent: true,
        opacity: 0.4,
        side: THREE.DoubleSide
    });
    
    carveBlockMesh = new THREE.Mesh(geometry, material);
    
    // Add wireframe edges
    const edges = new THREE.EdgesGeometry(geometry);
    const lineMaterial = new THREE.LineBasicMaterial({ color: 0xff0000, linewidth: 2 });
    const wireframe = new THREE.LineSegments(edges, lineMaterial);
    carveBlockMesh.add(wireframe);
    
    carveBlockMesh.visible = false;
    scene.add(carveBlockMesh);
    
    updateCarveBlockSize();
}

function toggleCarveMode() {
    isCarveMode = !isCarveMode;
    
    // Turn off other modes
    if (isCarveMode) {
        isSelectMode = false;
        const selectBtn = document.getElementById('select-mode-btn');
        if (selectBtn) {
            selectBtn.classList.remove('active');
            selectBtn.textContent = '☐ Select Mode';
        }
    }
    
    // Update UI
    const carveBtn = document.getElementById('carve-mode-btn');
    if (carveBtn) {
        carveBtn.classList.toggle('active', isCarveMode);
        carveBtn.textContent = isCarveMode ? '✓ Carve Mode ON' : '🔲 Carve Block';
    }
    
    // Show/hide carve block controls
    const carveControls = document.getElementById('carve-controls');
    if (carveControls) {
        carveControls.style.display = isCarveMode ? 'block' : 'none';
    }
    
    // Create carve block if needed
    if (!carveBlockMesh) {
        createCarveBlockMesh();
    }
    
    carveBlockMesh.visible = isCarveMode;
    
    if (isCarveMode) {
        debugLog(`Carve mode ON - ${carveBlockWidth.toFixed(1)}x${carveBlockHeight.toFixed(1)}x${carveBlockDepth.toFixed(1)} block`, 'info');
        setStatus('Carve Mode - Click to carve/push down vertices under block');
    } else {
        debugLog('Carve mode OFF', 'info');
        setStatus('Ready');
    }
}

function updateCarveBlockSize() {
    if (!carveBlockMesh) return;
    
    carveBlockMesh.scale.set(carveBlockWidth, carveBlockHeight, carveBlockDepth);
    
    // Update the wireframe
    if (carveBlockMesh.children.length > 0) {
        carveBlockMesh.remove(carveBlockMesh.children[0]);
    }
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const edges = new THREE.EdgesGeometry(geometry);
    const lineMaterial = new THREE.LineBasicMaterial({ color: 0xff0000, linewidth: 2 });
    const wireframe = new THREE.LineSegments(edges, lineMaterial);
    carveBlockMesh.add(wireframe);
}

function updateCarveWidth(value) {
    carveBlockWidth = parseFloat(value);
    document.getElementById('carve-width-value').textContent = carveBlockWidth.toFixed(1);
    updateCarveBlockSize();
}

function updateCarveHeight(value) {
    carveBlockHeight = parseFloat(value);
    document.getElementById('carve-height-value').textContent = carveBlockHeight.toFixed(1);
    updateCarveBlockSize();
}

function updateCarveDepth(value) {
    carveBlockDepth = parseFloat(value);
    document.getElementById('carve-depth-value').textContent = carveBlockDepth.toFixed(1);
    updateCarveBlockSize();
}

function updateCarveBlockPosition(point, normal) {
    if (!carveBlockMesh || !isCarveMode) return;
    
    // Position block at cursor, centered on surface
    carveBlockMesh.position.copy(point);
    
    // Offset so bottom of block is at surface
    carveBlockMesh.position.add(normal.clone().multiplyScalar(-carveBlockDepth / 2 + 0.1));
    
    // Orient block to align with surface normal
    const up = new THREE.Vector3(0, 0, 1);
    const quaternion = new THREE.Quaternion().setFromUnitVectors(up, normal);
    carveBlockMesh.quaternion.copy(quaternion);
}

function carveAtPoint(point, normal) {
    if (!mesh || !meshData.vertices) return;
    
    saveToHistory();
    
    const vertices = meshData.vertices;
    const vertexCount = vertices.length / 3;
    const localPoint = point.clone().sub(mesh.position);
    
    // Calculate block bounds in local space
    const halfW = carveBlockWidth / 2;
    const halfH = carveBlockHeight / 2;
    
    // Create transformation matrix based on surface normal
    const up = new THREE.Vector3(0, 0, 1);
    const quaternion = new THREE.Quaternion().setFromUnitVectors(up, normal);
    const inverseQuaternion = quaternion.clone().invert();
    
    let modifiedCount = 0;
    const carveAmount = carveBlockDepth * brushStrength;
    
    for (let i = 0; i < vertexCount; i++) {
        const vx = vertices[i * 3];
        const vy = vertices[i * 3 + 1];
        const vz = vertices[i * 3 + 2];
        
        // Get vertex position relative to carve point
        const relPos = new THREE.Vector3(vx - localPoint.x, vy - localPoint.y, vz - localPoint.z);
        
        // Transform to block-local space
        relPos.applyQuaternion(inverseQuaternion);
        
        // Check if within block XY bounds
        if (Math.abs(relPos.x) <= halfW && Math.abs(relPos.y) <= halfH) {
            // Check if within carve depth (from surface downward)
            if (relPos.z >= -carveBlockDepth && relPos.z <= carveBlockDepth * 0.5) {
                // Push vertex down along the normal (carve/indent)
                vertices[i * 3] -= normal.x * carveAmount;
                vertices[i * 3 + 1] -= normal.y * carveAmount;
                vertices[i * 3 + 2] -= normal.z * carveAmount;
                modifiedCount++;
            }
        }
    }
    
    // Update geometry
    if (modifiedCount > 0) {
        mesh.geometry.attributes.position.needsUpdate = true;
        mesh.geometry.computeVertexNormals();
        mesh.geometry.computeBoundingBox();
        debugLog(`✓ Carved ${modifiedCount} vertices (depth: ${carveAmount.toFixed(2)}mm)`, 'success');
    } else {
        debugLog(`✗ No vertices in carve block area`, 'warn');
    }
}

// ============================================================
// SCULPTING TOOLS
// ============================================================

function selectTool(tool) {
    currentTool = tool;
    
    // Turn off select mode when choosing a sculpt tool
    if (isSelectMode && tool !== 'select') {
        isSelectMode = false;
        const selectBtn = document.getElementById('select-mode-btn');
        if (selectBtn) {
            selectBtn.classList.remove('active');
            selectBtn.textContent = '☐ Select Mode';
        }
    }
    
    // Update UI
    document.querySelectorAll('.btn-tool').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tool === tool);
    });
    document.querySelectorAll('.toolbar-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    setStatus(`Tool: ${tool.charAt(0).toUpperCase() + tool.slice(1)}`);
}

function sculptAtPoint(point, normal) {
    if (!mesh || !meshData.vertices) return;
    
    // Use grid mode or circular mode
    if (gridMode) {
        sculptGridMode(point, normal);
    } else {
        sculptCircularMode(point, normal);
    }
}

function sculptGridMode(point, normal) {
    const vertices = meshData.vertices;
    const vertexCount = vertices.length / 3;
    
    // Transform point to mesh local space
    const localPoint = point.clone().sub(mesh.position);
    
    // Get grid-aligned bounds
    const gridBounds = getGridAlignedBounds(localPoint);
    
    debugLog(`Grid brush: ${gridBrushCells}x${gridBrushCells} at (${gridBounds.center.x.toFixed(2)}, ${gridBounds.center.y.toFixed(2)})`, 'info');
    
    // Determine direction
    let direction = 1;
    if (currentTool === 'lower' || (shiftPressed && currentTool === 'raise')) {
        direction = -1;
    }
    if (shiftPressed && currentTool === 'lower') {
        direction = 1;
    }
    
    const sculptNormal = normal.clone().normalize();
    const scaledStrength = brushStrength * gridCellSize * 2; // Scale strength to cell size
    
    // Find vertices within the grid square (using 2D projection on the surface plane)
    // We check X and Y bounds primarily (assuming Z is the "height" direction)
    let modifiedCount = 0;
    const brushVertices = [];
    
    // First pass: collect vertices in the grid square
    for (let i = 0; i < vertexCount; i++) {
        const vx = vertices[i * 3];
        const vy = vertices[i * 3 + 1];
        const vz = vertices[i * 3 + 2];
        
        // Check if vertex is within the grid square bounds (2D check on dominant axes)
        // Project to the plane perpendicular to the normal
        const toVertex = new THREE.Vector3(vx - localPoint.x, vy - localPoint.y, vz - localPoint.z);
        const distAlongNormal = toVertex.dot(sculptNormal);
        
        // Get position on the surface plane
        const onPlane = new THREE.Vector3(
            vx - sculptNormal.x * distAlongNormal,
            vy - sculptNormal.y * distAlongNormal,
            vz - sculptNormal.z * distAlongNormal
        );
        
        // Check if within grid square (using the projected center)
        const halfSize = gridBounds.size / 2;
        const dx = Math.abs(onPlane.x - gridBounds.center.x);
        const dy = Math.abs(onPlane.y - gridBounds.center.y);
        
        // Within square bounds AND close to the surface (within a cell depth)
        if (dx <= halfSize && dy <= halfSize && Math.abs(distAlongNormal) < gridCellSize * 2) {
            brushVertices.push({ index: i, x: vx, y: vy, z: vz, distAlongNormal });
        }
    }
    
    // Calculate flatten height if needed
    let flattenHeight = 0;
    if (currentTool === 'flatten' && brushVertices.length > 0) {
        for (const v of brushVertices) {
            flattenHeight += v.distAlongNormal;
        }
        flattenHeight /= brushVertices.length;
    }
    
    // Calculate smooth offsets if needed
    const smoothOffsets = new Map();
    if (currentTool === 'smooth' && brushVertices.length > 0) {
        // Use a larger search radius to find more neighbors
        const searchRadius = gridCellSize * 3;
        debugLog(`Smooth: searching with radius ${searchRadius.toFixed(3)}, ${brushVertices.length} vertices in brush`, 'info');
        
        for (const v of brushVertices) {
            let avgX = 0, avgY = 0, avgZ = 0, count = 0;
            for (const n of brushVertices) {
                if (v.index === n.index) continue;
                const dx = n.x - v.x;
                const dy = n.y - v.y;
                const dz = n.z - v.z;
                const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
                if (dist < searchRadius) {
                    avgX += n.x; avgY += n.y; avgZ += n.z;
                    count++;
                }
            }
            if (count > 0) {
                smoothOffsets.set(v.index, [
                    (avgX / count) - v.x,
                    (avgY / count) - v.y,
                    (avgZ / count) - v.z
                ]);
            }
        }
        debugLog(`Smooth: computed ${smoothOffsets.size} offsets from ${brushVertices.length} vertices`, 'info');
    }
    
    // Apply sculpting to vertices in the grid square
    for (const v of brushVertices) {
        const i = v.index;
        
        // Uniform strength within the square (no falloff - that's the point of grid mode!)
        const strength = scaledStrength;
        
        switch (currentTool) {
            case 'raise':
            case 'lower':
                vertices[i * 3] += sculptNormal.x * strength * direction;
                vertices[i * 3 + 1] += sculptNormal.y * strength * direction;
                vertices[i * 3 + 2] += sculptNormal.z * strength * direction;
                break;
                
            case 'flatten':
                const heightDiff = flattenHeight - v.distAlongNormal;
                vertices[i * 3] += sculptNormal.x * heightDiff * strength * 0.5;
                vertices[i * 3 + 1] += sculptNormal.y * heightDiff * strength * 0.5;
                vertices[i * 3 + 2] += sculptNormal.z * heightDiff * strength * 0.5;
                break;
                
            case 'smooth':
                const offset = smoothOffsets.get(i);
                if (offset) {
                    // Increased strength for visible effect
                    vertices[i * 3] += offset[0] * strength * 1.0;
                    vertices[i * 3 + 1] += offset[1] * strength * 1.0;
                    vertices[i * 3 + 2] += offset[2] * strength * 1.0;
                }
                break;
                
            case 'pinch':
                const toCenter = new THREE.Vector3(
                    gridBounds.center.x - v.x,
                    gridBounds.center.y - v.y,
                    0 // Don't pinch in Z
                );
                toCenter.normalize().multiplyScalar(strength * direction * 0.2);
                vertices[i * 3] += toCenter.x;
                vertices[i * 3 + 1] += toCenter.y;
                break;
        }
        
        modifiedCount++;
    }
    
    // Update geometry
    if (modifiedCount > 0) {
        mesh.geometry.attributes.position.needsUpdate = true;
        mesh.geometry.computeVertexNormals();
        mesh.geometry.computeBoundingBox();
        debugLog(`✓ Grid sculpted ${modifiedCount} vertices in ${gridBrushCells}x${gridBrushCells} cells`, 'success');
    } else {
        debugLog(`✗ No vertices in grid area`, 'warn');
    }
}

function sculptCircularMode(point, normal) {
    const vertices = meshData.vertices;
    const vertexCount = vertices.length / 3;
    
    // Transform point to mesh local space (since mesh is offset)
    const localPoint = point.clone().sub(mesh.position);
    
    debugLog(`Circular brush at (${localPoint.x.toFixed(2)}, ${localPoint.y.toFixed(2)}, ${localPoint.z.toFixed(2)})`, 'info');
    
    // Determine direction based on tool and shift key
    let direction = 1;
    if (currentTool === 'lower' || (shiftPressed && currentTool === 'raise')) {
        direction = -1;
    }
    if (shiftPressed && currentTool === 'lower') {
        direction = 1;
    }
    
    // Get the sculpting normal (transform to local space)
    const sculptNormal = normal.clone().normalize();
    
    // Scale strength based on mesh size - STRONG effect (0.5 = 50% of brush strength * mesh scale)
    // For a 18mm mesh with strength 0.3, this gives: 0.3 * 18 * 0.5 = 2.7mm displacement
    const scaledStrength = brushStrength * meshScale * 0.5;
    debugLog(`Sculpt strength: ${scaledStrength.toFixed(3)}mm, Normal: (${sculptNormal.x.toFixed(2)}, ${sculptNormal.y.toFixed(2)}, ${sculptNormal.z.toFixed(2)})`, 'info');
    
    // For flatten, calculate average height in brush area
    let flattenHeight = 0;
    let flattenCount = 0;
    
    if (currentTool === 'flatten') {
        for (let i = 0; i < vertexCount; i++) {
            const vx = vertices[i * 3];
            const vy = vertices[i * 3 + 1];
            const vz = vertices[i * 3 + 2];
            
            const dx = vx - localPoint.x;
            const dy = vy - localPoint.y;
            const dz = vz - localPoint.z;
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
            
            if (dist < brushSize) {
                // Project vertex onto normal direction
                const vertexPos = new THREE.Vector3(vx, vy, vz);
                const toVertex = vertexPos.clone().sub(localPoint);
                const heightAlongNormal = toVertex.dot(sculptNormal);
                flattenHeight += heightAlongNormal;
                flattenCount++;
            }
        }
        if (flattenCount > 0) {
            flattenHeight /= flattenCount;
        }
    }
    
    // For smooth, we need neighbor information - but ONLY for vertices in brush radius
    // First, find vertices in brush radius, then find their neighbors
    const smoothOffsets = new Map(); // vertex index -> [dx, dy, dz]
    
    if (currentTool === 'smooth') {
        // Step 1: Find all vertices in the brush area
        const brushVertices = [];
        for (let i = 0; i < vertexCount; i++) {
            const vx = vertices[i * 3];
            const vy = vertices[i * 3 + 1];
            const vz = vertices[i * 3 + 2];
            
            const dx = vx - localPoint.x;
            const dy = vy - localPoint.y;
            const dz = vz - localPoint.z;
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
            
            if (dist < brushSize) {
                brushVertices.push({ index: i, x: vx, y: vy, z: vz, dist });
            }
        }
        
        // Step 2: For each brush vertex, find nearby neighbors (only within brush area)
        const searchRadius = brushSize * 0.5; // Increased from 0.3
        debugLog(`Smooth (circular): ${brushVertices.length} vertices, search radius ${searchRadius.toFixed(3)}`, 'info');
        for (const v of brushVertices) {
            let avgX = 0, avgY = 0, avgZ = 0;
            let neighborCount = 0;
            
            // Only search other brush vertices (much smaller set!)
            for (const n of brushVertices) {
                if (v.index === n.index) continue;
                
                const dx = n.x - v.x;
                const dy = n.y - v.y;
                const dz = n.z - v.z;
                const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
                
                if (dist < searchRadius) {
                    avgX += n.x;
                    avgY += n.y;
                    avgZ += n.z;
                    neighborCount++;
                }
            }
            
            if (neighborCount > 0) {
                smoothOffsets.set(v.index, [
                    (avgX / neighborCount) - v.x,
                    (avgY / neighborCount) - v.y,
                    (avgZ / neighborCount) - v.z
                ]);
            }
        }
        
        debugLog(`Smooth: Found ${brushVertices.length} vertices in brush, ${smoothOffsets.size} with neighbors`, 'info');
    }
    
    // Apply sculpting to vertices within brush radius
    let modifiedCount = 0;
    let minDist = Infinity;
    let maxDist = 0;
    
    for (let i = 0; i < vertexCount; i++) {
        const vx = vertices[i * 3];
        const vy = vertices[i * 3 + 1];
        const vz = vertices[i * 3 + 2];
        
        // Distance from brush center (use local point)
        const dx = vx - localPoint.x;
        const dy = vy - localPoint.y;
        const dz = vz - localPoint.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        
        // Track min/max distances for debugging
        if (dist < minDist) minDist = dist;
        if (dist > maxDist) maxDist = dist;
        
        if (dist < brushSize) {
            // Calculate falloff (1 at center, 0 at edge)
            const normalizedDist = dist / brushSize;
            const falloff = Math.pow(1 - normalizedDist, brushFalloff * 3 + 0.5);
            
            // Use scaled strength
            const strength = scaledStrength * falloff;
            
            switch (currentTool) {
                case 'raise':
                case 'lower':
                    // Move along surface normal
                    vertices[i * 3] += sculptNormal.x * strength * direction;
                    vertices[i * 3 + 1] += sculptNormal.y * strength * direction;
                    vertices[i * 3 + 2] += sculptNormal.z * strength * direction;
                    break;
                    
                case 'flatten':
                    // Move toward average height
                    const vertexPos = new THREE.Vector3(vx, vy, vz);
                    const toVertex = vertexPos.clone().sub(localPoint);
                    const currentHeight = toVertex.dot(sculptNormal);
                    const heightDiff = flattenHeight - currentHeight;
                    
                    vertices[i * 3] += sculptNormal.x * heightDiff * strength * 2;
                    vertices[i * 3 + 1] += sculptNormal.y * heightDiff * strength * 2;
                    vertices[i * 3 + 2] += sculptNormal.z * heightDiff * strength * 2;
                    break;
                    
                case 'smooth':
                    // Move toward average of neighbors (using Map lookup)
                    const offset = smoothOffsets.get(i);
                    if (offset) {
                        // Increased strength multiplier for visible effect
                        vertices[i * 3] += offset[0] * strength * 2;
                        vertices[i * 3 + 1] += offset[1] * strength * 2;
                        vertices[i * 3 + 2] += offset[2] * strength * 2;
                    }
                    break;
                    
                case 'pinch':
                    // Move toward brush center
                    const pinchStrength = strength * direction;
                    vertices[i * 3] -= dx * pinchStrength;
                    vertices[i * 3 + 1] -= dy * pinchStrength;
                    vertices[i * 3 + 2] -= dz * pinchStrength;
                    break;
            }
            
            modifiedCount++;
        }
    }
    
    // Update geometry
    if (modifiedCount > 0) {
        mesh.geometry.attributes.position.needsUpdate = true;
        mesh.geometry.computeVertexNormals();
        mesh.geometry.computeBoundingBox();
        
        debugLog(`✓ Sculpted ${modifiedCount} vertices`, 'success');
    } else {
        debugLog(`✗ No vertices in brush range! Min dist: ${minDist.toFixed(2)}, Brush: ${brushSize.toFixed(2)}`, 'error');
    }
}

// ============================================================
// BRUSH SETTINGS
// ============================================================

function updateBrushSize(value) {
    brushSize = parseFloat(value);
    document.getElementById('brush-size-value').textContent = brushSize.toFixed(1);
    
    // Update brush indicator
    if (brushIndicator) {
        brushIndicator.scale.set(brushSize, brushSize, brushSize);
    }
}

function updateBrushStrength(value) {
    brushStrength = parseFloat(value);
    document.getElementById('brush-strength-value').textContent = brushStrength.toFixed(2);
}

function updateBrushFalloff(value) {
    brushFalloff = parseFloat(value);
    document.getElementById('brush-falloff-value').textContent = brushFalloff.toFixed(2);
}

// ============================================================
// MOUSE/KEYBOARD EVENTS
// ============================================================

function onMouseDown(event) {
    debugLog(`Mouse down - button: ${event.button}, mesh: ${!!mesh}`, 'info');
    if (event.button === 0 && mesh) { // Left click
        isSculpting = true;
        debugLog('Starting sculpt session', 'info');
        saveToHistory(); // Save state before sculpting
        performSculpt(event);
        setStatus(`Sculpting with ${currentTool}... (${brushSize.toFixed(2)} brush)`);
    }
}

function onMouseMove(event) {
    // Update mouse position
    const canvas = document.getElementById('canvas');
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    // Update brush indicator position
    updateBrushIndicator();
    
    // Sculpt if dragging
    if (isSculpting && mesh) {
        performSculpt(event);
    }
    
    // Update coordinates display
    raycaster.setFromCamera(mouse, camera);
    if (mesh) {
        const intersects = raycaster.intersectObject(mesh);
        if (intersects.length > 0) {
            const point = intersects[0].point;
            document.getElementById('status-coords').textContent = 
                `X: ${point.x.toFixed(2)} Y: ${point.y.toFixed(2)} Z: ${point.z.toFixed(2)}`;
        }
    }
}

function onMouseUp(event) {
    if (event.button === 0) {
        if (isSculpting) {
            setStatus('Ready');
        }
        isSculpting = false;
    }
}

function onWheel(event) {
    // Move duplicate forward/backward when dragging
    if (isDraggingDuplicate && duplicateMesh) {
        event.preventDefault();
        const cameraDirection = new THREE.Vector3();
        camera.getWorldDirection(cameraDirection);
        const moveAmount = event.deltaY > 0 ? -0.5 : 0.5;
        duplicateMesh.position.add(cameraDirection.multiplyScalar(moveAmount));
        return;
    }
    
    // Adjust brush size with Ctrl+scroll
    if (event.ctrlKey) {
        event.preventDefault();
        const delta = event.deltaY > 0 ? -0.2 : 0.2;
        const newSize = Math.max(0.5, Math.min(10, brushSize + delta));
        updateBrushSize(newSize);
        document.getElementById('brush-size').value = newSize;
    }
}

function onKeyDown(event) {
    if (event.key === 'Shift') {
        shiftPressed = true;
    }
    
    // Brush size shortcuts
    if (event.key === '[') {
        const step = meshScale * 0.02;
        const newSize = Math.max(meshScale * 0.01, brushSize - step);
        updateBrushSize(newSize);
        document.getElementById('brush-size').value = newSize;
    }
    if (event.key === ']') {
        const step = meshScale * 0.02;
        const newSize = Math.min(meshScale * 0.5, brushSize + step);
        updateBrushSize(newSize);
        document.getElementById('brush-size').value = newSize;
    }
    
    // Tool shortcuts
    if (event.key === '1') selectTool('raise');
    if (event.key === '2') selectTool('lower');
    if (event.key === '3') selectTool('flatten');
    if (event.key === '4') selectTool('smooth');
    if (event.key === '5') selectTool('pinch');
    
    // Undo/Redo
    if (event.ctrlKey && event.key === 'z') {
        event.preventDefault();
        undo();
    }
    if (event.ctrlKey && event.key === 'y') {
        event.preventDefault();
        redo();
    }
}

function onKeyUp(event) {
    if (event.key === 'Shift') {
        shiftPressed = false;
    }
}

function performSculpt() {
    if (!mesh) {
        debugLog('No mesh loaded!', 'error');
        return;
    }
    
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(mesh);
    
    if (intersects.length > 0) {
        const hit = intersects[0];
        
        // Log intersection details
        debugLog(`HIT at (${hit.point.x.toFixed(2)}, ${hit.point.y.toFixed(2)}, ${hit.point.z.toFixed(2)})`, 'success');
        
        // Get face normal and transform to world space
        const normalMatrix = new THREE.Matrix3().getNormalMatrix(mesh.matrixWorld);
        const worldNormal = hit.face.normal.clone().applyMatrix3(normalMatrix).normalize();
        sculptAtPoint(hit.point, worldNormal);
    } else {
        debugLog('No raycast hit on mesh', 'warn');
    }
}

function updateBrushIndicator() {
    if (!mesh) return;
    
    // Don't show brush when dragging duplicate
    if (isDraggingDuplicate) {
        brushIndicator && (brushIndicator.visible = false);
        gridSquareIndicator && (gridSquareIndicator.visible = false);
        return;
    }
    
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(mesh);
    
    if (intersects.length > 0) {
        const hit = intersects[0];
        
        // Get surface normal
        const normalMatrix = new THREE.Matrix3().getNormalMatrix(mesh.matrixWorld);
        const worldNormal = hit.face.normal.clone().applyMatrix3(normalMatrix).normalize();
        
        // In select mode or grid mode, show the grid square indicator
        if ((isSelectMode || gridMode) && gridSquareIndicator) {
            // Show grid square indicator (green in select mode, normal color otherwise)
            brushIndicator && (brushIndicator.visible = false);
            gridSquareIndicator.visible = true;
            
            // Change color based on mode
            if (gridSquareIndicator.material) {
                gridSquareIndicator.material.color.setHex(isSelectMode ? 0x00aaff : 0x00ff88);
            }
            
            // Get grid-aligned position
            const localPoint = hit.point.clone().sub(mesh.position);
            const gridBounds = getGridAlignedBounds(localPoint);
            
            // Position at grid center
            gridSquareIndicator.position.set(
                gridBounds.center.x + mesh.position.x,
                gridBounds.center.y + mesh.position.y,
                gridBounds.center.z + mesh.position.z
            );
            gridSquareIndicator.position.add(worldNormal.clone().multiplyScalar(0.02));
            
            // Orient to face along the surface normal
            gridSquareIndicator.lookAt(gridSquareIndicator.position.clone().add(worldNormal));
            
            // Scale to grid size
            const size = gridBounds.size / 2;
            gridSquareIndicator.scale.set(size, size, size);
            
        } else if (brushIndicator) {
            // Show circular brush indicator
            gridSquareIndicator && (gridSquareIndicator.visible = false);
            brushIndicator.visible = true;
            
            // Position slightly above surface
            brushIndicator.position.copy(hit.point).add(worldNormal.clone().multiplyScalar(0.01));
            
            // Orient to face along the surface normal
            brushIndicator.lookAt(hit.point.clone().add(worldNormal));
            
            // Scale to brush size
            brushIndicator.scale.set(brushSize, brushSize, brushSize);
        }
    } else {
        brushIndicator && (brushIndicator.visible = false);
        gridSquareIndicator && (gridSquareIndicator.visible = false);
    }
}

// ============================================================
// VIEW CONTROLS
// ============================================================

function resetCamera() {
    if (!mesh) {
        camera.position.set(30, 20, 30);
        controls.target.set(0, 0, 0);
    } else {
        mesh.geometry.computeBoundingBox();
        const box = mesh.geometry.boundingBox;
        const size = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        
        camera.position.set(maxDim * 2, maxDim * 1.5, maxDim * 2);
        controls.target.set(0, 0, 0);
    }
    controls.update();
}

function toggleWireframe() {
    if (!mesh) return;
    wireframeMode = !wireframeMode;
    mesh.material.wireframe = wireframeMode;
    setStatus(`Wireframe: ${wireframeMode ? 'ON' : 'OFF'}`);
}

function toggleNormals() {
    if (!mesh) return;
    
    showNormalsMode = !showNormalsMode;
    
    if (normalsHelper) {
        scene.remove(normalsHelper);
        normalsHelper = null;
    }
    
    if (showNormalsMode) {
        // Create vertex normals helper
        const normalLength = brushSize * 0.5;
        normalsHelper = new THREE.VertexNormalsHelper(mesh, normalLength, 0x00ff00);
        scene.add(normalsHelper);
    }
    
    setStatus(`Normals: ${showNormalsMode ? 'ON' : 'OFF'}`);
}

function onWindowResize() {
    const canvas = document.getElementById('canvas');
    const container = document.getElementById('canvas-container');
    
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

// ============================================================
// HISTORY (UNDO/REDO)
// ============================================================

function saveToHistory() {
    if (!meshData.vertices) return;
    
    // Remove any redo states
    history = history.slice(0, historyIndex + 1);
    
    // Save current state
    history.push(meshData.vertices.slice());
    historyIndex = history.length - 1;
    
    // Limit history size
    if (history.length > MAX_HISTORY) {
        history.shift();
        historyIndex--;
    }
}

function undo() {
    if (historyIndex > 0) {
        historyIndex--;
        restoreFromHistory();
        setStatus('Undo');
    }
}

function redo() {
    if (historyIndex < history.length - 1) {
        historyIndex++;
        restoreFromHistory();
        setStatus('Redo');
    }
}

function restoreFromHistory() {
    if (!mesh || historyIndex < 0) return;
    
    meshData.vertices = history[historyIndex].slice();
    mesh.geometry.attributes.position.array = meshData.vertices;
    mesh.geometry.attributes.position.needsUpdate = true;
    mesh.geometry.computeVertexNormals();
}

// ============================================================
// EXPORT
// ============================================================

async function exportMesh(format = 'obj') {
    if (!mesh || !meshData.vertices) {
        setStatus('No mesh to export');
        return;
    }
    
    showLoading(`Exporting as ${format.toUpperCase()}...`);
    
    // Convert Float32Array vertices back to array of [x,y,z]
    const vertices = [];
    for (let i = 0; i < meshData.vertices.length; i += 3) {
        vertices.push([
            meshData.vertices[i],
            meshData.vertices[i + 1],
            meshData.vertices[i + 2]
        ]);
    }
    
    try {
        const response = await fetch('/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                vertices: vertices,
                faces: meshData.faces,
                format: format,
                filename: 'sculpted_mesh'
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Trigger download
            window.location.href = data.download_url;
            setStatus(`Exported: ${data.filename}`);
        } else {
            throw new Error(data.error || 'Export failed');
        }
    } catch (error) {
        console.error('Export error:', error);
        setStatus('Export failed: ' + error.message);
    }
    
    hideLoading();
}

// ============================================================
// UI HELPERS
// ============================================================

function updateMeshInfo(info) {
    document.getElementById('info-vertices').textContent = info.vertex_count.toLocaleString();
    document.getElementById('info-faces').textContent = info.face_count.toLocaleString();
    document.getElementById('info-dimensions').textContent = 
        `${info.dimensions.width.toFixed(1)} × ${info.dimensions.height.toFixed(1)} × ${info.dimensions.depth.toFixed(1)}`;
}

function setStatus(text) {
    document.getElementById('status-text').textContent = text;
}

function showLoading(text = 'Loading...') {
    document.getElementById('loading-text').textContent = text;
    document.getElementById('loading').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

// ============================================================
// RENDER LOOP
// ============================================================

function animate() {
    requestAnimationFrame(animate);
    
    controls.update();
    
    // Update normals helper if active
    if (normalsHelper) {
        normalsHelper.update();
    }
    
    renderer.render(scene, camera);
}

// ============================================================
// START
// ============================================================

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', init);
