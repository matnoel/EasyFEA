const modelSelector = document.getElementById('model-selector');
const viewer = document.getElementById('model-viewer');
const toggleBtn = document.getElementById('autoplay-toggle');
const slider = document.getElementById('animation-slider');

let playing = true;
let sliderValue = 0;

// ---------- Control time ----------

function setCurrentTime()
{
    if (viewer.duration) {
        slider.value = sliderValue;
        viewer.currentTime = sliderValue * viewer.duration;
    }
}

function saveCurrentTime()
{
    if (viewer.duration) {
        sliderValue =  viewer.currentTime / viewer.duration;
    }
}

// ---------- Animation controls ----------
function setAnimationControls()
{
    if (viewer.availableAnimations && viewer.availableAnimations.length > 0) {
        // Animation available
        if (playing)
        {
            viewer.setAttribute('autoplay', '');
            viewer.removeAttribute('animation-controls');
            viewer.play();

            slider.style.display = 'none';
            toggleBtn.textContent = '⏸';
        }
        else
        {
            viewer.removeAttribute('autoplay');
            viewer.setAttribute('animation-controls', '');
            viewer.pause();
            
            slider.style.display = 'block';
            toggleBtn.textContent = '▶';
        }
        setCurrentTime();
    } else {
        // No animation - hides controls
        toggleBtn.style.display = 'none';
        slider.style.display = 'none';
    }
}

viewer.addEventListener('load', () => {
    setAnimationControls();
});

// ---------- Playing toggle ----------
toggleBtn.addEventListener('click', () => {
    saveCurrentTime();
    playing = !playing;
    setAnimationControls();
});

// ---------- Slider control ----------
slider.addEventListener('input', (e) => {
    sliderValue = parseFloat(e.target.value);
    if (viewer.duration) {
        viewer.currentTime = sliderValue * viewer.duration;
    }
});

if (modelSelector) {
    modelSelector.addEventListener('change', () => {
        saveCurrentTime();
    });
}