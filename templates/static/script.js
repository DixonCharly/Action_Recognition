// script.js

const actionLabel = document.getElementById('action-label');
const actionDescriptionText = document.getElementById('action-description-text');
const stopBtn = document.getElementById('stop-btn');

stopBtn.addEventListener('click', () => {
    actionLabel.textContent = 'Action recognition stopped.';
});


