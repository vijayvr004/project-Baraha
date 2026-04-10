(function () {
    const canvas = document.getElementById('drawCanvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clearBtn');
    const predictBtn = document.getElementById('predictBtn');
    const canvasHint = document.querySelector('.canvas-hint');

    const resultIdle = document.getElementById('resultIdle');
    const resultLoading = document.getElementById('resultLoading');
    const resultOutput = document.getElementById('resultOutput');
    const resultError = document.getElementById('resultError');
    const predictedChar = document.getElementById('predictedChar');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceValue = document.getElementById('confidenceValue');
    const alternativesList = document.getElementById('alternativesList');
    const errorMessage = document.getElementById('errorMessage');
    const errorSub = document.getElementById('errorSub');

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    let hasDrawn = false;

    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#000000';

    function showState(state) {
        resultIdle.classList.remove('active');
        resultLoading.classList.remove('active');
        resultOutput.classList.remove('active');
        resultError.classList.remove('active');

        if (state === 'idle') resultIdle.classList.add('active');
        else if (state === 'loading') resultLoading.classList.add('active');
        else if (state === 'output') resultOutput.classList.add('active');
        else if (state === 'error') resultError.classList.add('active');
    }

    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        if (e.touches) {
            const touch = e.touches[0];
            return {
                x: (touch.clientX - rect.left) * scaleX,
                y: (touch.clientY - rect.top) * scaleY
            };
        }
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    function startDraw(e) {
        e.preventDefault();
        isDrawing = true;
        const pos = getPos(e);
        lastX = pos.x;
        lastY = pos.y;

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, ctx.lineWidth / 2, 0, Math.PI * 2);
        ctx.fillStyle = '#000000';
        ctx.fill();

        canvas.classList.add('drawing');

        if (!hasDrawn) {
            hasDrawn = true;
            canvasHint.classList.add('hidden');
        }
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();

        const pos = getPos(e);

        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();

        lastX = pos.x;
        lastY = pos.y;
    }

    function stopDraw(e) {
        if (!isDrawing) return;
        isDrawing = false;
        canvas.classList.remove('drawing');
    }

    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDraw);
    canvas.addEventListener('mouseleave', stopDraw);

    canvas.addEventListener('touchstart', startDraw, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    canvas.addEventListener('touchend', stopDraw);
    canvas.addEventListener('touchcancel', stopDraw);

    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        hasDrawn = false;
        canvasHint.classList.remove('hidden');
        showState('idle');
    }

    clearBtn.addEventListener('click', clearCanvas);

    predictBtn.addEventListener('click', async function () {
        if (!hasDrawn) {
            errorMessage.textContent = 'Please draw a character first.';
            errorSub.textContent = '';
            showState('error');
            return;
        }

        showState('loading');
        predictBtn.disabled = true;
        clearBtn.disabled = true;

        try {
            const imageData = canvas.toDataURL('image/png');

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });

            const data = await response.json();

            if (!response.ok) {
                const msg = data.error || 'Prediction failed.';
                errorMessage.textContent = msg;

                if (data.model_missing) {
                    errorSub.textContent = 'Run: python model_trainer.py in the terminal to train the model first.';
                } else {
                    errorSub.textContent = '';
                }
                showState('error');
                return;
            }

            predictedChar.textContent = data.character;
            confidenceBar.style.width = data.confidence + '%';
            confidenceValue.textContent = data.confidence.toFixed(1) + '%';

            alternativesList.innerHTML = '';
            if (data.alternatives && data.alternatives.length > 0) {
                data.alternatives.forEach(alt => {
                    const item = document.createElement('div');
                    item.className = 'alt-item';

                    const charEl = document.createElement('span');
                    charEl.className = 'alt-char';
                    charEl.textContent = alt.character;

                    const confEl = document.createElement('span');
                    confEl.className = 'alt-conf';
                    confEl.textContent = (alt.confidence * 100).toFixed(1) + '%';

                    item.appendChild(charEl);
                    item.appendChild(confEl);
                    alternativesList.appendChild(item);
                });
            }

            showState('output');

        } catch (err) {
            errorMessage.textContent = 'Network error. Could not reach the server.';
            errorSub.textContent = '';
            showState('error');
        } finally {
            predictBtn.disabled = false;
            clearBtn.disabled = false;
        }
    });

    showState('idle');
})();
