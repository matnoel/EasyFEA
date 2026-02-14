document.addEventListener('DOMContentLoaded', function() {
    const modelSelector = document.getElementById('model-selector');
    const modelViewer = document.getElementById('model-viewer');
    const colorbarImg = document.getElementById('colorbar');

    modelSelector.addEventListener('change', function(event) {
        const selectedOption = event.target.options[event.target.selectedIndex];

        modelViewer.src = selectedOption.value;
        colorbarImg.src = selectedOption.dataset.colorbar;
    });
});