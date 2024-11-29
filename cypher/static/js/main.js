let chart;

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
                tension: 0.1
            }, {
                label: 'Accuracy',
                data: [],
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    document.getElementById('startTraining').addEventListener('click', startTraining);
});

function startTraining() {
    document.getElementById('startTraining').disabled = true;
    
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
} 