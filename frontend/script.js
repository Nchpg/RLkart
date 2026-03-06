const canvas = document.getElementById('trackCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const generateBtn = document.getElementById('generateBtn');
const simulateBtn = document.getElementById('simulateBtn');
const statusDiv = document.getElementById('status');
const errorDiv = document.getElementById('error-msg');

// --- Configuration ---
const SCALE = 10; 
const OFFSET_X = canvas.width / 2;
const OFFSET_Y = canvas.height / 2;
const POINT_RADIUS = 6;

let controlPoints = [];
let draggedPoint = null;
let simulationResult = null;
let currentFrame = 0;
let isAnimating = false;
let isTrackValid = true;

// --- Catmull-Rom Preview Logic ---
function getCatmullRomPoint(p0, p1, p2, p3, t, tension = 0.5) {
    const t2 = t * t;
    const t3 = t2 * t;
    const f1 = -tension * t3 + 2 * tension * t2 - tension * t;
    const f2 = (2 - tension) * t3 + (tension - 3) * t2 + 1;
    const f3 = (tension - 2) * t3 + (3 - 2 * tension) * t2 + tension * t;
    const f4 = tension * t3 - tension * t2;
    return {
        x: p0.x * f1 + p1.x * f2 + p2.x * f3 + p3.x * f4,
        y: p0.y * f1 + p1.y * f2 + p2.y * f3 + p3.y * f4
    };
}

function generatePreviewCenterline(points, samples = 20) {
    if (points.length < 3) return [];
    const centerline = [];
    const n = points.length;
    for (let i = 0; i < n; i++) {
        const p0 = points[(i - 1 + n) % n];
        const p1 = points[i];
        const p2 = points[(i + 1) % n];
        const p3 = points[(i + 2) % n];
        for (let s = 0; s < samples; s++) {
            centerline.push(getCatmullRomPoint(p0, p1, p2, p3, s / samples));
        }
    }
    return centerline;
}

// --- Interaction ---
function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    let x = (e.clientX - rect.left - OFFSET_X) / SCALE;
    let y = -(e.clientY - rect.top - OFFSET_Y) / SCALE;

    // Clamp to canvas boundaries
    const margin = 1;
    const xLimit = (canvas.width / 2) / SCALE - margin;
    const yLimit = (canvas.height / 2) / SCALE - margin;

    x = Math.max(-xLimit, Math.min(xLimit, x));
    y = Math.max(-yLimit, Math.min(yLimit, y));

    return { x, y };
}

canvas.addEventListener('mousedown', (e) => {
    if (isAnimating) return;
    const pos = getMousePos(e);
    
    const hitIndex = controlPoints.findIndex(p => {
        const dx = p.x - pos.x;
        const dy = p.y - pos.y;
        return Math.sqrt(dx*dx + dy*dy) < (POINT_RADIUS * 2 / SCALE);
    });

    if (e.button === 0) { // Left Click
        if (hitIndex !== -1) {
            draggedPoint = controlPoints[hitIndex];
        } else {
            // Add new point at mouse position
            controlPoints.push({ x: pos.x, y: pos.y });
            simulationResult = null;
            validateTrack();
            draw();
        }
    } else if (e.button === 2) { // Right Click
        if (hitIndex !== -1) {
            controlPoints.splice(hitIndex, 1);
            simulationResult = null;
            validateTrack();
            draw();
        }
    }
});

canvas.addEventListener('contextmenu', (e) => e.preventDefault());

window.addEventListener('mousemove', (e) => {
    if (draggedPoint) {
        const pos = getMousePos(e);
        draggedPoint.x = pos.x;
        draggedPoint.y = pos.y;
        simulationResult = null;
        validateTrack();
        draw();
    }
});

window.addEventListener('mouseup', () => { draggedPoint = null; });

function segmentsIntersect(p1, p2, p3, p4) {
    function ccw(A, B, C) {
        return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
    }
    return (ccw(p1, p3, p4) !== ccw(p2, p3, p4)) && (ccw(p1, p2, p3) !== ccw(p1, p2, p4));
}

function getAngle(p1, p2, p3) {
    const v1 = { x: p1.x - p2.x, y: p1.y - p2.y };
    const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };
    const dot = v1.x * v2.x + v1.y * v2.y;
    const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
    const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
    if (mag1 === 0 || mag2 === 0) return 180;
    return Math.acos(Math.max(-1, Math.min(1, dot / (mag1 * mag2)))) * 180 / Math.PI;
}

function validateTrack() {
    errorDiv.innerText = "";
    isTrackValid = true;
    if (controlPoints.length < 3) return;

    // Check for sharp angles
    const minAngle = 15; 
    for (let i = 0; i < controlPoints.length; i++) {
        const p1 = controlPoints[(i - 1 + controlPoints.length) % controlPoints.length];
        const p2 = controlPoints[i];
        const p3 = controlPoints[(i + 1) % controlPoints.length];
        if (getAngle(p1, p2, p3) < minAngle) {
            errorDiv.innerText = "Angle too sharp! Widen the turn.";
            isTrackValid = false;
            break;
        }
    }

    if (!isTrackValid) { updateButtons(); return; }

    const preview = generatePreviewCenterline(controlPoints, 10);
    const n = preview.length;
    for (let i = 0; i < n; i++) {
        const p1 = preview[i];
        const p2 = preview[(i + 1) % n];
        for (let j = i + 2; j < n; j++) {
            if ((j + 1) % n === i) continue;
            const p3 = preview[j];
            const p4 = preview[(j + 1) % n];
            if (segmentsIntersect(p1, p2, p3, p4)) {
                errorDiv.innerText = "Warning: The track crosses itself!";
                isTrackValid = false;
                break;
            }
        }
        if (!isTrackValid) break;
    }
    updateButtons();
}

function updateButtons() {
    simulateBtn.disabled = !isTrackValid || isAnimating || controlPoints.length < 3;
}

function toCanvasX(x) { return x * SCALE + OFFSET_X; }
function toCanvasY(y) { return -y * SCALE + OFFSET_Y; }

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Grid & Axis
    ctx.strokeStyle = '#222'; ctx.lineWidth = 1;
    for(let i = 0; i < canvas.width; i += 50) { ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, canvas.height); ctx.stroke(); }
    for(let i = 0; i < canvas.height; i += 50) { ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(canvas.width, i); ctx.stroke(); }
    
    // Preview Spline
    const preview = generatePreviewCenterline(controlPoints);
    ctx.strokeStyle = isTrackValid ? '#444' : '#622';
    ctx.lineWidth = 2;
    ctx.beginPath();
    preview.forEach((p, i) => {
        if(i === 0) ctx.moveTo(toCanvasX(p.x), toCanvasY(p.y));
        else ctx.lineTo(toCanvasX(p.x), toCanvasY(p.y));
    });
    if (preview.length > 0) ctx.closePath();
    ctx.stroke();

    // Simulation Result
    if (simulationResult) {
        const { track, trajectory } = simulationResult;
        ctx.lineWidth = 2; ctx.strokeStyle = isTrackValid ? '#fff' : '#ff4444';
        [track.left, track.right].forEach(border => {
            if (border && border.length > 0) {
                ctx.beginPath();
                border.forEach((p, i) => {
                    if(i === 0) ctx.moveTo(toCanvasX(p[0]), toCanvasY(p[1]));
                    else ctx.lineTo(toCanvasX(p[0]), toCanvasY(p[1]));
                });
                ctx.closePath(); ctx.stroke();
            }
        });

        // Trajectory
        if (trajectory && trajectory.length > 0) {
            ctx.strokeStyle = '#0f0'; ctx.lineWidth = 2;
            ctx.beginPath();
            const limit = Math.min(currentFrame, trajectory.length);
            for (let i = 0; i < limit; i++) {
                const p = trajectory[i];
                if(i === 0) ctx.moveTo(toCanvasX(p.x), toCanvasY(p.y));
                else ctx.lineTo(toCanvasX(p.x), toCanvasY(p.y));
            }
            ctx.stroke();
            const carIdx = Math.min(currentFrame, trajectory.length - 1);
            const car = trajectory[carIdx];
            if (car) drawCar(car.x, car.y, car.yaw);
        }
    }

    // Control Points
    controlPoints.forEach(p => {
        ctx.fillStyle = (p === draggedPoint) ? '#0ff' : '#f44';
        ctx.beginPath(); ctx.arc(toCanvasX(p.x), toCanvasY(p.y), POINT_RADIUS, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke();
    });
}

function drawCar(x, y, yaw) {
    const cx = toCanvasX(x); const cy = toCanvasY(y);
    const w = 1.0 * SCALE; const h = 0.6 * SCALE; 
    ctx.save();
    ctx.translate(cx, cy); ctx.rotate(-yaw); 
    ctx.fillStyle = '#aaa'; ctx.fillRect(-w/2, -h/2, w, h);
    ctx.fillStyle = '#f00'; ctx.fillRect(w/2 - 2, -h/2, 4, h);
    ctx.restore();
}

clearBtn.addEventListener('click', () => {
    controlPoints = []; simulationResult = null; currentFrame = 0;
    isAnimating = false; isTrackValid = true;
    statusDiv.innerText = "Click to add points."; errorDiv.innerText = "";
    updateButtons(); draw();
});

generateBtn.addEventListener('click', async () => {
    statusDiv.innerText = "Generating randomly...";
    try {
        const response = await fetch('/generate');
        if (!response.ok) throw new Error(await response.text());
        controlPoints = await response.json();
        simulationResult = null;
        validateTrack();
        draw();
        statusDiv.innerText = "Track generated! Adjust it or simulate.";
    } catch (e) {
        statusDiv.innerText = "Error: " + e.message;
    }
});

simulateBtn.addEventListener('click', async () => {
    statusDiv.innerText = "Simulation in progress...";
    simulateBtn.disabled = true;
    isTrackValid = true;
    try {
        const response = await fetch('/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ control_points: controlPoints, width: 4.0 })
        });
        if (!response.ok) throw new Error(await response.text());
        simulationResult = await response.json();

        if (!simulationResult.track.left || simulationResult.track.left.length < 3 || 
            !simulationResult.track.right || simulationResult.track.right.length < 3) {
            errorDiv.innerText = "Invalid: Turn too tight.";
            isTrackValid = false; draw(); simulateBtn.disabled = false; return;
        }

        currentFrame = 0; isAnimating = true; animate();
        statusDiv.innerText = simulationResult.success ? "Success!" : "Failed.";
    } catch (e) {
        statusDiv.innerText = "Error: " + e.message; simulateBtn.disabled = false;
    }
});

function animate() {
    if (!isAnimating) return;
    draw(); currentFrame++;
    if (simulationResult && currentFrame < simulationResult.trajectory.length) {
        requestAnimationFrame(animate);
    } else {
        isAnimating = false; simulateBtn.disabled = false; draw();
    }
}

draw();
