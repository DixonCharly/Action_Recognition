// Wrap your code in a DOMContentLoaded event listener to ensure the document is fully loaded
document.addEventListener('DOMContentLoaded', function () {
    // Get references to the elements you want to work with
    const showPatternButton = document.getElementById('show-pattern-button');
    const startTimeInput = document.getElementById('start-time');
    const endTimeInput = document.getElementById('end-time');
    const patternContainer = document.getElementById('pattern-container');

    // Function to create and display a chart
    function createChart(data) {
        const chartContainer = document.getElementById('myChart');

        if (!chartContainer) {
            console.error('Canvas element not found');
            return;
        }

        // Extract data for the chart (replace with your data)
        const labels = data.pattern.map(entry => entry.action);
        const counts = data.pattern.map(entry => entry.count);

        // Create a chart using Chart.js
        const chart = new Chart(chartContainer, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Activity Counts',
                    data: counts,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)', // Customize the chart colors
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // Add an event listener to the "Show Pattern" button
    showPatternButton.addEventListener('click', function () {
        // Get the selected start and end times
        const startTime = startTimeInput.value;
        const endTime = endTimeInput.value;

        // Make an AJAX request to fetch data for the specified time range
        fetch(`/get_activity_data?start_time=${startTime}&end_time=${endTime}`)
            .then(response => response.json())
            .then(data => {
                // Handle the response data and update the pattern container
                patternContainer.innerHTML = ''; // Clear previous content

                // Check if data.pattern is an array and not empty
                if (Array.isArray(data.pattern) && data.pattern.length > 0) {
                    // Display the pattern as a list
                    const ul = document.createElement('ul');
                    data.pattern.forEach(entry => {
                        const li = document.createElement('li');
                        li.textContent = `${entry.action}: ${entry.count}`;
                        ul.appendChild(li);
                    });
                    patternContainer.appendChild(ul);

                    // Create and display the chart
                    createChart(data);
                } else {
                    // Display a message if there is no data
                    patternContainer.textContent = 'No data available for the selected time range.';
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
    });
});
