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
    
    // Alternate background colors for better readability
    if (logsDiv.children.length % 2 === 0) {
        logEntry.style.backgroundColor = '#f8f9fa';
    }
    
    logsDiv.appendChild(logEntry);
    logsDiv.scrollTop = logsDiv.scrollHeight;
}

function clearLogs() {
    logsDiv.innerHTML = '';
}

function startTraining() {
    document.getElementById('startTraining').disabled = true;
    clearLogs();
    
    fetch('/train')
        .then(response => response.json())
        .then(data => {
            updateChart(data);
            document.getElementById('startTraining').disabled = false;
        });
}

function updateChart(data) {
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