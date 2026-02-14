const modelSelector = document.getElementById('model-selector');
const viewer = document.getElementById('model-viewer');
const toggleBtn = document.getElementById('autoplay-toggle');
const slider = document.getElementById('animation-slider');

let autoplayOn = true;
let sliderValue = 0;

// ---------- Autoplay toggle ----------
toggleBtn.addEventListener('click', () => {
    autoplayOn = !autoplayOn;

    if (autoplayOn) {
        viewer.setAttribute('autoplay', '');
        viewer.removeAttribute('animation-controls');
        viewer.play();

        slider.style.display = 'none';
        toggleBtn.textContent = '▶';
    } else {
        viewer.removeAttribute('autoplay');
        viewer.setAttribute('animation-controls', '');
        viewer.pause();

        slider.style.display = 'block';
        toggleBtn.textContent = '⏸';
    }

    // save slider value
    if (viewer.duration) {
        sliderValue = viewer.currentTime / viewer.duration;
        slider.value = sliderValue;
    }
});

// ---------- Slider controls animation ----------
slider.addEventListener('input', (e) => {
    sliderValue = parseFloat(e.target.value);
    if (viewer.duration) {
        viewer.currentTime = sliderValue * viewer.duration;
    }
});

// ---------- Update slider at model load ----------
viewer.addEventListener('load', () => {
    if (viewer.duration) {
        viewer.currentTime = sliderValue * viewer.duration;
    }
});