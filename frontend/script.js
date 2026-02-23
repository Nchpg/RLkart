const canvas = document.getElementById('trackCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const simulateBtn = document.getElementById('simulateBtn');
const statusDiv = document.getElementById('status');
const errorDiv = document.getElementById('error-msg');
const trackWidthInput = document.getElementById('trackWidth');

let controlPoints = [];
let simulationResult = null;
let currentFrame = 0;
let isAnimating = false;
let isTrackValid = true;

const SCALE = 10; 
const OFFSET_X = canvas.width / 2;
const OFFSET_Y = canvas.height / 2;

function segmentsIntersect(p1, p2, p3, p4) {
    function ccw(A, B, C) {
        return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
    }
    return (ccw(p1, p3, p4) !== ccw(p2, p3, p4)) && (ccw(p1, p2, p3) !== ccw(p1, p2, p4));
}

function checkSelfIntersection(points) {
    if (points.length < 4) return false;
    const n = points.length;
    for (let i = 0; i < n; i++) {
        const p1 = points[i];
        const p2 = points[(i + 1) % n];
        for (let j = i + 2; j < n; j++) {
            if ((j + 1) % n === i) continue;
            const p3 = points[j];
            const p4 = points[(j + 1) % n];
            if (segmentsIntersect(p1, p2, p3, p4)) return true;
        }
    }
    return false;
}

function checkBorderOverlap(track) {
    if (!track.left || !track.right || track.left.length < 2 || track.right.length < 2) return false;
    const left = track.left.map(p => ({x: p[0], y: p[1]}));
    const right = track.right.map(p => ({x: p[0], y: p[1]}));
    
    // Sampling for performance if track is very long
    const step = Math.max(1, Math.floor(left.length / 200)); 
    for (let i = 0; i < left.length - step; i += step) {
        const l1 = left[i];
        const l2 = left[i + step];
        for (let j = 0; j < right.length - step; j += step) {
            const r1 = right[j];
            const r2 = right[j + step];
            if (segmentsIntersect(l1, l2, r1, r2)) return true;
        }
    }
    return false;
}

function validateControlPoints() {
    errorDiv.innerText = "";
    isTrackValid = true;

    if (controlPoints.length >= 3) {
        if (checkSelfIntersection(controlPoints)) {
            errorDiv.innerText = "Attention: Le circuit se croise lui-même !";
            isTrackValid = false;
        }
    }
    
    for (let i = 0; i < controlPoints.length; i++) {
        for (let j = i + 1; j < controlPoints.length; j++) {
            const dx = controlPoints[i].x - controlPoints[j].x;
            const dy = controlPoints[i].y - controlPoints[j].y;
            const dist = Math.sqrt(dx*dx + dy*dy);
            if (dist < 1.0) {
                errorDiv.innerText = "Points trop proches.";
                isTrackValid = false;
            }
        }
    }
    updateButtons();
}

canvas.addEventListener('click', (e) => {
    if (isAnimating) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left - OFFSET_X) / SCALE;
    const y = -(e.clientY - rect.top - OFFSET_Y) / SCALE; 
    
    controlPoints.push({ x, y });
    validateControlPoints();
    draw();
});

function updateButtons() {
    simulateBtn.disabled = controlPoints.length < 3 || !isTrackValid;
}

function toCanvasX(x) { return x * SCALE + OFFSET_X; }
function toCanvasY(y) { return -y * SCALE + OFFSET_Y; }

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Grid & Axis
    ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
    for(let i = 0; i < canvas.width; i += 50) { ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, canvas.height); ctx.stroke(); }
    for(let i = 0; i < canvas.height; i += 50) { ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(canvas.width, i); ctx.stroke(); }
    ctx.strokeStyle = '#555';
    ctx.beginPath(); ctx.moveTo(0, OFFSET_Y); ctx.lineTo(canvas.width, OFFSET_Y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(OFFSET_X, 0); ctx.lineTo(OFFSET_X, canvas.height); ctx.stroke();

    if (simulationResult) {
        const { track, trajectory } = simulationResult;
        
        // Draw Borders
        ctx.lineWidth = 2;
        ctx.strokeStyle = isTrackValid ? '#fff' : '#ff4444';
        
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

        // Centerline
        ctx.strokeStyle = '#444'; ctx.setLineDash([5, 5]);
        ctx.beginPath();
        track.centerline.forEach((p, i) => {
            if(i === 0) ctx.moveTo(toCanvasX(p[0]), toCanvasY(p[1]));
            else ctx.lineTo(toCanvasX(p[0]), toCanvasY(p[1]));
        });
        ctx.closePath(); ctx.stroke();
        ctx.setLineDash([]);

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

            // Car
            const carIdx = Math.min(currentFrame, trajectory.length - 1);
            const car = trajectory[carIdx];
            if (car) drawCar(car.x, car.y, car.yaw);
        }
    }

    // Control Points
    ctx.fillStyle = isTrackValid ? '#f00' : '#ff8800';
    controlPoints.forEach(p => {
        ctx.beginPath(); ctx.arc(toCanvasX(p.x), toCanvasY(p.y), 4, 0, Math.PI * 2); ctx.fill();
    });

    if (controlPoints.length > 1) {
        ctx.strokeStyle = isTrackValid ? 'rgba(255, 0, 0, 0.3)' : 'rgba(255, 136, 0, 0.5)';
        ctx.beginPath();
        ctx.moveTo(toCanvasX(controlPoints[0].x), toCanvasY(controlPoints[0].y));
        controlPoints.forEach(p => ctx.lineTo(toCanvasX(p.x), toCanvasY(p.y)));
        ctx.lineTo(toCanvasX(controlPoints[0].x), toCanvasY(controlPoints[0].y));
        ctx.stroke();
    }
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
    statusDiv.innerText = "Tout effacé."; errorDiv.innerText = "";
    updateButtons(); draw();
});

simulateBtn.addEventListener('click', async () => {
    statusDiv.innerText = "Simulation en cours...";
    simulateBtn.disabled = true;
    isTrackValid = true; // Reset validity for new attempt
    
    try {
        const response = await fetch('http://localhost:8000/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                control_points: controlPoints,
                width: parseFloat(trackWidthInput.value)
            })
        });
        
        if (!response.ok) throw new Error(await response.text());
        
        simulationResult = await response.json();
        console.log("Simulation result received:", simulationResult);

        if (checkBorderOverlap(simulationResult.track)) {
            errorDiv.innerText = "Invalide: Les bordures se chevauchent.";
            isTrackValid = false;
            draw();
            simulateBtn.disabled = false;
            return;
        }

        currentFrame = 0;
        isAnimating = true;
        animate();
        statusDiv.innerText = simulationResult.success ? "Succès!" : "Échec.";
    } catch (e) {
        statusDiv.innerText = "Erreur: " + e.message;
        simulateBtn.disabled = false;
    }
});

function animate() {
    if (!isAnimating) return;
    draw();
    currentFrame++;
    if (simulationResult && currentFrame < simulationResult.trajectory.length) {
        requestAnimationFrame(animate);
    } else {
        isAnimating = false;
        simulateBtn.disabled = false;
        draw(); // Final draw to ensure everything stays visible
    }
}

draw();
