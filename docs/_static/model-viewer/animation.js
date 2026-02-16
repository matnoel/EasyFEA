const modelSelector = document.getElementById('model-selector');
const viewer = document.getElementById('model-viewer');
const toggleBtn = document.getElementById('autoplay-toggle');
const slider = document.getElementById('animation-slider');
let autoplayOn = true;
let sliderValue = 0;

function setAnimationState(playing) {
    autoplayOn = playing;
    if (playing) {
        viewer.setAttribute('autoplay', '');
        viewer.removeAttribute('animation-controls');
        viewer.play();
        slider.style.display = 'none';
        toggleBtn.textContent = '⏸';
    } else {
        viewer.removeAttribute('autoplay');
        viewer.setAttribute('animation-controls', '');
        viewer.pause();
        slider.style.display = 'block';
        toggleBtn.textContent = '▶';
    }
    // save slider value
    if (viewer.duration) {
        sliderValue = viewer.currentTime / viewer.duration;
        slider.value = sliderValue;
    }
}

// ---------- Update current time at model load ----------
viewer.addEventListener('load', () => {
    if (viewer.availableAnimations && viewer.availableAnimations.length > 0) {
        // Animation available
        toggleBtn.style.display = 'block';
        // Restore position regardless of play/pause state
        if (viewer.duration) {
            viewer.currentTime = sliderValue * viewer.duration;
            slider.value = sliderValue;
        }
        setAnimationState(autoplayOn);
    } else {
        // No animation - hide controls
        toggleBtn.style.display = 'none';
        slider.style.display = 'none';
    }
});

// ---------- Save position before model change ----------
modelSelector.addEventListener('change', () => {
    if (viewer.duration) {
        sliderValue = viewer.currentTime / viewer.duration;
    }
});

// ---------- Autoplay toggle ----------
toggleBtn.addEventListener('click', () => {
    autoplayOn = !autoplayOn;
    setAnimationState(autoplayOn);
});

// ---------- Slider controls animation ----------
slider.addEventListener('input', (e) => {
    sliderValue = parseFloat(e.target.value);
    if (viewer.duration) {
        viewer.currentTime = sliderValue * viewer.duration;
    }
});