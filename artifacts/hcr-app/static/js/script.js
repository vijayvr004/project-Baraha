(function () {
    'use strict';

    // ═══════════════════════════════════════════════════════════════
    // TAB SWITCHING
    // ═══════════════════════════════════════════════════════════════
    const tabDraw     = document.getElementById('tabDraw');
    const tabUpload   = document.getElementById('tabUpload');
    const panelDraw   = document.getElementById('panelDraw');
    const panelUpload = document.getElementById('panelUpload');

    function switchTab(activeTab, activePanel) {
        [tabDraw, tabUpload].forEach(t => {
            t.classList.remove('active');
            t.setAttribute('aria-selected', 'false');
        });
        [panelDraw, panelUpload].forEach(p => p.classList.remove('active'));
        activeTab.classList.add('active');
        activeTab.setAttribute('aria-selected', 'true');
        activePanel.classList.add('active');
    }

    tabDraw.addEventListener('click',   () => switchTab(tabDraw,   panelDraw));
    tabUpload.addEventListener('click', () => switchTab(tabUpload, panelUpload));


    // ═══════════════════════════════════════════════════════════════
    // MODEL STATUS POLLING — polls /model-status until TrOCR ready
    // ═══════════════════════════════════════════════════════════════
    const modelBanner       = document.getElementById('modelBanner');
    const modelBannerInner  = document.getElementById('modelBannerInner');
    const modelBannerTitle  = document.getElementById('modelBannerTitle');
    const modelBannerSub    = document.getElementById('modelBannerSub');
    const modelSpinner      = document.getElementById('modelSpinner');
    const engineBadge       = document.getElementById('engineBadge');
    const engineBadgeLabel  = document.getElementById('engineBadgeLabel');
    const idleEngineNote    = document.getElementById('idleEngineNote');
    const statEngine        = document.getElementById('statEngine');
    const statModel         = document.getElementById('statModel');

    let trocr_ready  = false;
    let pollInterval = null;

    function applyModelStatus(data) {
        const status = data.trocr_status;

        if (status === 'ready') {
            // ── Engine is ready — show actual engine name ───────────────────
            trocr_ready = true;
            clearInterval(pollInterval);

            const engineName  = data.engine_name  || 'OCR Engine';
            const modelName   = data.trocr_model  || '—';
            const isGemini    = data.gemini_key_set;
            const isOllama    = engineName.toLowerCase().includes('ollama');

            // Banner → success
            modelBannerInner.classList.add('banner-success');
            modelSpinner.style.display = 'none';
            modelBannerTitle.textContent = '✓ ' + engineName + ' — Ready';
            if (isGemini) {
                modelBannerSub.textContent = 'Model: gemini-1.5-flash · Highest accuracy (95%+) · Cloud-powered';
            } else if (isOllama) {
                modelBannerSub.textContent = 'Model: ' + modelName + ' · Running 100% locally · No API key needed';
            } else {
                modelBannerSub.textContent = 'Model: ' + modelName + ' · Running locally';
            }

            // Engine badge → green
            engineBadge.className  = 'engine-badge badge-ready';
            engineBadgeLabel.textContent = isGemini ? '✦ Gemini Vision' : engineName;

            // Idle note
            if (idleEngineNote) {
                idleEngineNote.textContent = isGemini
                    ? 'Powered by Gemini Vision — 95%+ accuracy on cursive handwriting'
                    : 'Powered by ' + engineName;
            }

            // Stats panel
            statEngine.textContent = engineName;
            statModel.textContent  = isGemini ? 'gemini-1.5-flash' : modelName;

            // Auto-hide banner after 4 s
            setTimeout(() => {
                modelBanner.style.maxHeight = '0';
                modelBanner.style.opacity   = '0';
                modelBanner.style.marginBottom = '0';
            }, 4000);

        } else if (status === 'loading') {
            // ── Still loading ───────────────────────────────────────────────
            engineBadge.className  = 'engine-badge badge-loading';
            engineBadgeLabel.textContent = 'Loading model…';
            if (idleEngineNote) idleEngineNote.textContent = 'OCR model loading — upload will work once ready';

        } else if (status === 'error') {
            // ── Failed, fallback to Tesseract ──────────────────────────────
            trocr_ready = false;
            clearInterval(pollInterval);

            modelBannerInner.classList.add('banner-warning');
            modelSpinner.style.display = 'none';
            modelBannerTitle.textContent = '⚠ Primary engine failed to load';
            modelBannerSub.textContent   = data.trocr_error || 'Falling back to Tesseract OCR';

            engineBadge.className  = 'engine-badge badge-warning';
            engineBadgeLabel.textContent = data.tesseract_available ? 'Tesseract (fallback)' : 'No engine';

            const fbEngine = data.tesseract_available ? 'Tesseract OCR v5 (fallback)' : 'None available';
            statEngine.textContent = fbEngine;
            statModel.textContent  = data.tesseract_available ? 'Tesseract v5' : '—';
            if (idleEngineNote) idleEngineNote.textContent = data.tesseract_available
                ? 'Using Tesseract as fallback'
                : 'No OCR engine available';
        }
    }

    async function pollModelStatus() {
        try {
            const res  = await fetch('/model-status');
            const data = await res.json();
            applyModelStatus(data);
        } catch (e) {
            // server not ready yet — keep polling
        }
    }

    // Poll immediately, then every 3 seconds
    pollModelStatus();
    pollInterval = setInterval(pollModelStatus, 3000);


    // ═══════════════════════════════════════════════════════════════
    // TAB 1 — DRAW CHARACTER (canvas + CNN predict)
    // ═══════════════════════════════════════════════════════════════
    const canvas     = document.getElementById('drawCanvas');
    const ctx        = canvas.getContext('2d');
    const clearBtn   = document.getElementById('clearBtn');
    const predictBtn = document.getElementById('predictBtn');
    const canvasHint = document.querySelector('.canvas-hint');

    const resultIdle    = document.getElementById('resultIdle');
    const resultLoading = document.getElementById('resultLoading');
    const resultOutput  = document.getElementById('resultOutput');
    const resultError   = document.getElementById('resultError');
    const predictedChar = document.getElementById('predictedChar');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceValue   = document.getElementById('confidenceValue');
    const alternativesList  = document.getElementById('alternativesList');
    const errorMessage  = document.getElementById('errorMessage');
    const errorSub      = document.getElementById('errorSub');

    const saraControlsChar = document.getElementById('saraControlsChar');
    const saraPlayChar     = document.getElementById('saraPlayChar');
    const saraStopChar     = document.getElementById('saraStopChar');
    const saraLabelChar    = document.getElementById('saraLabelChar');

    let isDrawing = false;
    let lastX = 0, lastY = 0;
    let hasDrawn = false;

    ctx.lineWidth   = 15;
    ctx.lineCap     = 'round';
    ctx.lineJoin    = 'round';
    ctx.strokeStyle = '#000000';

    function showState(state) {
        [resultIdle, resultLoading, resultOutput, resultError]
            .forEach(el => el.classList.remove('active'));
        if (state === 'idle')    resultIdle.classList.add('active');
        else if (state === 'loading') resultLoading.classList.add('active');
        else if (state === 'output')  resultOutput.classList.add('active');
        else if (state === 'error')   resultError.classList.add('active');
    }

    function getPos(e) {
        const rect   = canvas.getBoundingClientRect();
        const scaleX = canvas.width  / rect.width;
        const scaleY = canvas.height / rect.height;
        if (e.touches) {
            const t = e.touches[0];
            return { x: (t.clientX - rect.left) * scaleX, y: (t.clientY - rect.top) * scaleY };
        }
        return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY };
    }

    function startDraw(e) {
        e.preventDefault();
        isDrawing = true;
        const pos = getPos(e);
        lastX = pos.x; lastY = pos.y;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, ctx.lineWidth / 2, 0, Math.PI * 2);
        ctx.fillStyle = '#000000';
        ctx.fill();
        canvas.classList.add('drawing');
        if (!hasDrawn) { hasDrawn = true; canvasHint.classList.add('hidden'); }
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const pos = getPos(e);
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        lastX = pos.x; lastY = pos.y;
    }

    function stopDraw() { isDrawing = false; canvas.classList.remove('drawing'); }

    canvas.addEventListener('mousedown',  startDraw);
    canvas.addEventListener('mousemove',  draw);
    canvas.addEventListener('mouseup',    stopDraw);
    canvas.addEventListener('mouseleave', stopDraw);
    canvas.addEventListener('touchstart', startDraw, { passive: false });
    canvas.addEventListener('touchmove',  draw,      { passive: false });
    canvas.addEventListener('touchend',   stopDraw);
    canvas.addEventListener('touchcancel',stopDraw);

    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        hasDrawn = false;
        canvasHint.classList.remove('hidden');
        showState('idle');
        saraControlsChar.style.display = 'none';
        if (typeof Sara !== 'undefined') Sara.stop();
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
        clearBtn.disabled   = true;

        try {
            const imageData = canvas.toDataURL('image/png');
            const response  = await fetch('/predict', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ image: imageData })
            });
            const data = await response.json();

            if (!response.ok) {
                errorMessage.textContent = data.error || 'Prediction failed.';
                errorSub.textContent = data.model_missing
                    ? 'CNN model has DLL issues on this machine. The Draw tab requires TensorFlow.'
                    : '';
                showState('error');
                return;
            }

            predictedChar.textContent = data.character;
            confidenceBar.style.width  = data.confidence + '%';
            confidenceValue.textContent = data.confidence.toFixed(1) + '%';

            alternativesList.innerHTML = '';
            (data.alternatives || []).forEach(alt => {
                const item   = document.createElement('div');
                item.className = 'alt-item';
                const charEl = document.createElement('span');
                charEl.className = 'alt-char'; charEl.textContent = alt.character;
                const confEl = document.createElement('span');
                confEl.className = 'alt-conf'; confEl.textContent = (alt.confidence * 100).toFixed(1) + '%';
                item.appendChild(charEl);
                item.appendChild(confEl);
                alternativesList.appendChild(item);
            });

            if (typeof Sara !== 'undefined' && Sara.isSupported) {
                saraControlsChar.style.display = 'flex';
                saraPlayChar.onclick = () => {
                    Sara.onStartCallback = () => {
                        saraPlayChar.classList.add('speaking');
                        saraStopChar.classList.add('active');
                        saraLabelChar.classList.add('active');
                    };
                    Sara.onEndCallback = () => {
                        saraPlayChar.classList.remove('speaking');
                        saraStopChar.classList.remove('active');
                        saraLabelChar.classList.remove('active');
                    };
                    Sara.speak(`The predicted character is ${data.character}`);
                };
                saraStopChar.onclick = () => Sara.stop();
            } else if (typeof Sara !== 'undefined') {
                saraControlsChar.style.display = 'flex';
                saraControlsChar.innerHTML = '<span class="sara-label active" style="color:var(--text-muted);">Audio not supported in this browser.</span>';
            }

            showState('output');
        } catch (err) {
            errorMessage.textContent = 'Network error. Could not reach the server.';
            errorSub.textContent = '';
            showState('error');
        } finally {
            predictBtn.disabled = false;
            clearBtn.disabled   = false;
        }
    });

    showState('idle');


    // ═══════════════════════════════════════════════════════════════
    // TAB 2 — UPLOAD DOCUMENT (TrOCR primary / Tesseract fallback)
    // ═══════════════════════════════════════════════════════════════
    const dropZone       = document.getElementById('dropZone');
    const fileInput      = document.getElementById('fileInput');
    const previewBox     = document.getElementById('previewBox');
    const previewImg     = document.getElementById('previewImg');
    const previewMeta    = document.getElementById('previewMeta');
    const clearUploadBtn = document.getElementById('clearUploadBtn');
    const ocrLoadingMsg  = document.getElementById('ocrLoadingMsg');

    const ocrIdle        = document.getElementById('ocrIdle');
    const ocrOutput      = document.getElementById('ocrOutput');
    const ocrError       = document.getElementById('ocrError');
    const ocrErrorMsg    = document.getElementById('ocrErrorMessage');
    const ocrPagesContainer = document.getElementById('ocrPagesContainer');
    const ocrWordCount   = document.getElementById('ocrWordCount');
    const ocrBlockCount  = document.getElementById('ocrBlockCount');

    const copyBtn         = document.getElementById('copyBtn');
    const downloadTxtBtn  = document.getElementById('downloadTxtBtn');
    const downloadPdfBtn  = document.getElementById('downloadPdfBtn');
    const downloadDocxBtn = document.getElementById('downloadDocxBtn');

    const saraControlsDoc = document.getElementById('saraControlsDoc');
    const saraPlayDoc     = document.getElementById('saraPlayDoc');
    const saraStopDoc     = document.getElementById('saraStopDoc');
    const saraLabelDoc    = document.getElementById('saraLabelDoc');

    function showOcrState(state) {
        [ocrIdle, ocrOutput, ocrError].forEach(el => el.classList.remove('active'));
        if (state === 'idle')   ocrIdle.classList.add('active');
        else if (state === 'output') ocrOutput.classList.add('active');
        else if (state === 'error')  ocrError.classList.add('active');
    }

    function getAllOcrText() {
        return Array.from(ocrPagesContainer.querySelectorAll('textarea'))
                    .map(ta => ta.value)
                    .join('\n\n');
    }

    function updateWordCount() {
        const text  = getAllOcrText().trim();
        const words = text ? text.split(/\s+/).filter(Boolean).length : 0;
        ocrWordCount.textContent = words + (words === 1 ? ' word' : ' words');
    }

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', e => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', e => {
        if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove('dragover');
    });
    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });
    fileInput.addEventListener('change', () => {
        if (fileInput.files[0]) handleFile(fileInput.files[0]);
        fileInput.value = '';
    });

    clearUploadBtn.addEventListener('click', () => {
        previewImg.src = '';
        previewBox.classList.remove('visible');
        showOcrState('idle');
        ocrBlockCount.textContent = '—';
        saraControlsDoc.style.display = 'none';
        if (typeof Sara !== 'undefined') Sara.stop();
    });

    function handleFile(file) {
        const allowed = ['application/pdf', 'image/png', 'image/jpeg', 'image/webp', 'image/bmp', 'image/tiff'];
        if (!allowed.includes(file.type)) {
            showOcrState('error');
            ocrErrorMsg.textContent = `Unsupported file type: ${file.type || 'unknown'}. Please use PDF, PNG, JPG, WEBP, BMP, or TIFF.`;
            return;
        }
        const reader = new FileReader();
        reader.onload = e => {
            if (file.type === 'application/pdf') {
                // Generic PDF icon SVG
                previewImg.src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="%23374151"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8" fill="none" stroke="%239ca3af" stroke-width="2"/><text x="7" y="16" font-family="sans-serif" font-size="5" fill="%239ca3af" font-weight="bold">PDF</text></svg>';
                previewImg.style.objectFit = 'contain';
                previewImg.style.padding = '20px';
                previewImg.style.background = '#111827';
            } else {
                previewImg.src = e.target.result;
                previewImg.style.objectFit = '';
                previewImg.style.padding = '';
                previewImg.style.background = '';
            }
            previewBox.classList.add('visible');
            previewMeta.textContent = `${file.name} — ${formatBytes(file.size)}`;
        };
        reader.readAsDataURL(file);
        runOcr(file);
    }

    async function runOcr(file) {
        dropZone.classList.add('loading');
        showOcrState('idle');
        ocrBlockCount.textContent = '—';

        // Update loading message based on active engine
        if (ocrLoadingMsg) {
            ocrLoadingMsg.textContent = 'Processing document... please wait (this may take a minute for PDFs)';
        }

        const formData = new FormData();
        formData.append('image', file);
        
        const engineSelect = document.getElementById('engineSelect');
        if (engineSelect) {
            formData.append('engine_pref', engineSelect.value);
        }

        try {
            const response = await fetch('/ocr', { method: 'POST', body: formData });
            const data     = await response.json();

            if (!response.ok) {
                showOcrState('error');
                if (data.model_loading) {
                    ocrErrorMsg.textContent = 'TrOCR is still loading. Wait a moment and try again.';
                } else if (data.tesseract_missing) {
                    ocrErrorMsg.innerHTML =
                        'No OCR engine available. ' +
                        '<a href="https://github.com/UB-Mannheim/tesseract/wiki" target="_blank" ' +
                        'style="color:var(--accent);">Install Tesseract</a> as a fallback, ' +
                        'or wait for TrOCR to finish loading.';
                } else {
                    ocrErrorMsg.textContent = data.error || 'OCR failed. Please try again.';
                }
                return;
            }

            ocrPagesContainer.innerHTML = '';
            
            if (data.pages && data.pages.length > 0) {
                data.pages.forEach(page => {
                    const wrapper = document.createElement('div');
                    wrapper.className = 'ocr-page-wrapper';
                    
                    if (data.pages.length > 1) {
                        const header = document.createElement('div');
                        header.className = 'ocr-page-header';
                        header.textContent = `Page ${page.page_num}`;
                        wrapper.appendChild(header);
                    }
                    
                    const ta = document.createElement('textarea');
                    ta.className = 'ocr-textarea';
                    ta.value = page.text || '';
                    ta.spellcheck = true;
                    ta.addEventListener('input', updateWordCount);
                    
                    wrapper.appendChild(ta);
                    ocrPagesContainer.appendChild(wrapper);
                });
            } else {
                const ta = document.createElement('textarea');
                ta.className = 'ocr-textarea';
                ta.placeholder = data.message || 'No text extracted.';
                ta.addEventListener('input', updateWordCount);
                ocrPagesContainer.appendChild(ta);
            }
            updateWordCount();

            // Update stats panel with engine info from response
            const engineName = data.engine || 'Unknown';
            statEngine.textContent = engineName;
            // Show correct model name per engine
            if (data.engine_key === 'gemini')        statModel.textContent = 'gemini-1.5-flash';
            else if (data.engine_key === 'ollama')   statModel.textContent = 'llava:7b / qwen2.5vl:7b';
            else if (data.engine_key === 'easyocr')  statModel.textContent = 'EasyOCR 1.7';
            else if (data.engine_key === 'trocr')    statModel.textContent = 'trocr-large-handwritten';
            else                                     statModel.textContent = 'Tesseract v5';

            const lineCount = data.line_count || 0;
            ocrBlockCount.textContent = lineCount + (lineCount === 1 ? ' line' : ' lines');

            if (typeof Sara !== 'undefined' && Sara.isSupported && getAllOcrText().trim().length > 0) {
                saraControlsDoc.style.display = 'flex';
                saraPlayDoc.onclick = () => {
                    Sara.onStartCallback = () => {
                        saraPlayDoc.classList.add('speaking');
                        saraStopDoc.classList.add('active');
                        saraLabelDoc.classList.add('active');
                    };
                    Sara.onEndCallback = () => {
                        saraPlayDoc.classList.remove('speaking');
                        saraStopDoc.classList.remove('active');
                        saraLabelDoc.classList.remove('active');
                    };
                    Sara.speak(getAllOcrText());
                };
                saraStopDoc.onclick = () => Sara.stop();
            } else if (typeof Sara !== 'undefined' && !Sara.isSupported) {
                saraControlsDoc.style.display = 'flex';
                saraControlsDoc.innerHTML = '<span class="sara-label active" style="color:var(--text-muted);">Audio not supported in this browser.</span>';
            }

            showOcrState('output');
        } catch (err) {
            showOcrState('error');
            ocrErrorMsg.textContent = 'Network error. Make sure the server is running.';
        } finally {
            dropZone.classList.remove('loading');
        }
    }

    // ── Copy to clipboard ─────────────────────────────────────────
    copyBtn.addEventListener('click', async () => {
        const text = getAllOcrText();
        if (!text) return;
        try {
            await navigator.clipboard.writeText(text);
            copyBtn.classList.add('copy-success');
            copyBtn.textContent = '✓ Copied!';
            setTimeout(() => {
                copyBtn.classList.remove('copy-success');
                copyBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy`;
            }, 2000);
        } catch {
            alert('Copy failed. Please manually select and copy the text.');
        }
    });

    // ── Download .txt ─────────────────────────────────────────────
    downloadTxtBtn.addEventListener('click', () => {
        const text = getAllOcrText();
        if (!text) return;
        const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
        triggerDownload(blob, 'extracted_text.txt');
    });

    // ── Download .docx ────────────────────────────────────────────
    downloadDocxBtn.addEventListener('click', async () => {
        const text = getAllOcrText();
        if (!text) return;

        downloadDocxBtn.disabled    = true;
        downloadDocxBtn.textContent = 'Generating…';

        try {
            const response = await fetch('/download/docx', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ text })
            });
            if (!response.ok) {
                const err = await response.json();
                alert(err.error || 'DOCX generation failed.');
                return;
            }
            const blob = await response.blob();
            triggerDownload(blob, 'extracted_text.docx');
        } catch {
            alert('Network error. Could not reach the server.');
        } finally {
            downloadDocxBtn.disabled = false;
            downloadDocxBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg> .docx`;
        }
    });

    // ── Download .pdf ─────────────────────────────────────────────
    downloadPdfBtn.addEventListener('click', () => {
        const text = getAllOcrText();
        if (!text) return;
        
        try {
            const { jsPDF } = window.jspdf;
            const doc = new jsPDF();
            
            doc.setFont("helvetica", "normal");
            doc.setFontSize(12);
            
            const margin = 15;
            const pageWidth = doc.internal.pageSize.getWidth();
            const pageHeight = doc.internal.pageSize.getHeight();
            const textLines = doc.splitTextToSize(text, pageWidth - margin * 2);
            
            let cursorY = margin + 5;
            
            textLines.forEach(line => {
                if (cursorY > pageHeight - margin) {
                    doc.addPage();
                    cursorY = margin + 5;
                }
                doc.text(line, margin, cursorY);
                cursorY += 6; // approximate line height
            });
            
            doc.save('extracted_text.pdf');
        } catch (e) {
            console.error(e);
            alert('Failed to generate PDF. Make sure jsPDF is loaded.');
        }
    });

    // ── Helpers ───────────────────────────────────────────────────
    function triggerDownload(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a   = document.createElement('a');
        a.href     = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 5000);
    }

    function formatBytes(bytes) {
        if (bytes < 1024)    return bytes + ' B';
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1048576).toFixed(1) + ' MB';
    }

    // ═══════════════════════════════════════════════════════════════
    // SARA — WEB SPEECH API AUDIO PLAYBACK
    // ═══════════════════════════════════════════════════════════════
    const Sara = {
        synth: window.speechSynthesis,
        voice: null,
        utterance: null,
        isPlaying: false,
        isSupported: !!window.speechSynthesis,
        onStartCallback: null,
        onEndCallback: null,

        init() {
            if (!this.isSupported) return;
            this.loadVoices();
            if (this.synth.onvoiceschanged !== undefined) {
                this.synth.onvoiceschanged = () => this.loadVoices();
            }
        },

        loadVoices() {
            const voices = this.synth.getVoices();
            if (!voices.length) return;

            const names = ["samantha", "zira", "google uk english female", "aria", "female"];
            for (let name of names) {
                const found = voices.find(v => v.name.toLowerCase().includes(name) && v.lang.startsWith('en'));
                if (found) { this.voice = found; return; }
            }
            
            const fallbackFemale = voices.find(v => v.lang.startsWith('en') && (v.name.toLowerCase().includes('female') || v.gender === 'female'));
            if (fallbackFemale) { this.voice = fallbackFemale; return; }
            
            this.voice = voices.find(v => v.lang.startsWith('en')) || voices[0];
        },

        speak(text) {
            if (!this.isSupported || !text) return;
            this.stop();

            this.utterance = new SpeechSynthesisUtterance(text);
            if (this.voice) this.utterance.voice = this.voice;
            this.utterance.rate = 0.88;
            this.utterance.pitch = 1.08;
            this.utterance.volume = 1.0;

            this.utterance.onstart = () => {
                this.isPlaying = true;
                if (this.onStartCallback) this.onStartCallback();
            };

            this.utterance.onend = () => {
                this.isPlaying = false;
                if (this.onEndCallback) this.onEndCallback();
            };

            this.utterance.onerror = (e) => {
                console.error("SpeechSynthesis Error:", e);
                this.isPlaying = false;
                if (this.onEndCallback) this.onEndCallback();
            };

            this.synth.speak(this.utterance);
        },

        stop() {
            if (!this.isSupported) return;
            this.synth.cancel();
            this.isPlaying = false;
            if (this.onEndCallback) this.onEndCallback();
        }
    };

    Sara.init();

    // Init
    showOcrState('idle');

})();
