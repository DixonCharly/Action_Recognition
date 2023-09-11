// Add an event listener to a button or form submission
const showPatternButton = document.getElementById('show-pattern-button');
showPatternButton.addEventListener('click', () => {
    // Get the selected time range (start and end times)
    const startTime = document.getElementById('start-time').value;
    const endTime = document.getElementById('end-time').value;

    // Make an AJAX request to the /get_activity_data route
    fetch(`/get_activity_data?start_time=${startTime}&end_time=${endTime}`)
        .then(response => response.json())
        .then(data => {
            // Handle the response data and update the webpage
            const patternContainer = document.getElementById('pattern-container');

            // Example: Display the pattern as a list
            patternContainer.innerHTML = '<h2>Activity Pattern:</h2>';
            data.pattern.forEach(entry => {
                patternContainer.innerHTML += `<p>${entry.action}: ${entry.count}</p>`;
            });
        })
        .catch(error => {
            console.error('Error:', error);
        });
});
