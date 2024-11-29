let chart;
const logsDiv = document.getElementById('trainingLogs');

document.addEventListener('DOMContentLoaded', function() {
    const ctx = document.getElementById('trainingChart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Loss',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
                yAxisID: 'y'
            }, {
                label: 'Accuracy',
                data: [],
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1,
                yAxisID: 'y1'
            }]
        },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Loss'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Accuracy (%)'
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });

    document.getElementById('startTraining').addEventListener('click', startTraining);
});

function addLog(epoch, batch, loss, accuracy) {
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    logEntry.innerHTML = `Epoch ${epoch}, Batch ${batch}: Loss = ${loss.toFixed(4)}, Accuracy = ${accuracy.toFixed(2)}%`;
    
    if (logsDiv.children.length % 2 === 0) {
        logEntry.style.backgroundColor = '#f8f9fa';
    }
    
    logsDiv.appendChild(logEntry);
    logsDiv.scrollTop = logsDiv.scrollHeight;
}

function startTraining() {
    const button = document.getElementById('startTraining');
    button.disabled = true;
    button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Training...';

    // Clear previous data
    chart.data.labels = [];
    chart.data.datasets[0].data = [];
    chart.data.datasets[1].data = [];
    chart.update();
    
    if (logsDiv) {
        logsDiv.innerHTML = '';
    }

    fetch('/train')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            updateChart(data);
            button.disabled = false;
            button.innerHTML = 'Start Training';
        })
        .catch(error => {
            console.error('Error during training:', error);
            button.disabled = false;
            button.innerHTML = 'Start Training';
            alert('An error occurred during training. Please check the console for details.');
        });
}

function updateChart(data) {
    if (!data || data.length === 0) {
        console.error('No data received from training');
        return;
    }

    const labels = data.map(item => `Epoch ${item.epoch}, Batch ${item.batch}`);
    const losses = data.map(item => item.loss);
    const accuracies = data.map(item => item.accuracy);

    chart.data.labels = labels;
    chart.data.datasets[0].data = losses;
    chart.data.datasets[1].data = accuracies;
    chart.update();

    // Add logs for each data point
    data.forEach(item => {
        addLog(item.epoch, item.batch, item.loss, item.accuracy);
    });
} 