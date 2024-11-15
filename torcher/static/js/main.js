let trainingChart = null;
let eventSource = null;

// Add this variable at the top of your file to control how many points to show
const MAX_DATA_POINTS = 100;  // Adjust this number as needed

function initializeChart() {
    const ctx = document.getElementById('trainingChart').getContext('2d');
    trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training Loss',
                    data: [],
                    borderColor: '#7ed0e1',
                    yAxisID: 'y-loss',
                    pointRadius: 0,  // Remove points for better performance
                    borderWidth: 1,
                },
                {
                    label: 'Accuracy (%)',
                    data: [],
                    borderColor: '#ed98f9',
                    yAxisID: 'y-accuracy',
                    pointRadius: 0,  // Remove points for better performance
                    borderWidth: 1,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,  // Allow chart to fill container
            animation: {
                duration: 0  // Disable animations for better performance
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            },
            plugins: {
                legend: {
                    position: 'top',
                },
            },
            scales: {
                x: {
                    ticks: {
                        maxTicksLimit: 10,  // Limit number of x-axis labels
                        maxRotation: 0,
                        minRotation: 0
                    }
                },
                'y-loss': {
                    type: 'linear',
                    position: 'left',
                    grid: {
                        color: 'rgba(126, 208, 225, 0.1)'
                    },
                    ticks: {
                        maxTicksLimit: 6  // Limit number of y-axis labels
                    }
                },
                'y-accuracy': {
                    type: 'linear',
                    position: 'right',
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxTicksLimit: 6  // Limit number of y-axis labels
                    }
                }
            }
        }
    });
}

function setupEventSource() {
    if (eventSource) {
        eventSource.close();
    }
    
    eventSource = new EventSource('/stream');
    
    eventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            if (data.error) {
                console.error('Server error:', data.error);
                eventSource.close();
                return;
            }
            updateUI(data);
        } catch (error) {
            console.error('Error parsing event data:', error);
        }
    };
    
    eventSource.onerror = function(error) {
        console.error('EventSource failed:', error);
        if (eventSource.readyState === EventSource.CLOSED) {
            console.log('EventSource closed');
        }
    };
    
    eventSource.onopen = function() {
        console.log('EventSource connected');
    };
    
    return eventSource;
}

function updateUI(data) {
    // Update progress
    const totalSteps = data.total_epochs * data.total_batches;
    const currentStep = (data.current_epoch - 1) * data.total_batches + data.current_batch;
    const progressPercentage = (currentStep / totalSteps) * 100;
    
    document.getElementById('epoch-text').textContent = 
        `Epoch: ${data.current_epoch}/${data.total_epochs}`;
    document.getElementById('batch-text').textContent = 
        `Batch: ${data.current_batch}/${data.total_batches}`;
    document.getElementById('overall-progress-fill').style.width = 
        `${progressPercentage}%`;
    
    // Update image and prediction
    const currentImage = document.getElementById('current-image');
    const predictionText = document.getElementById('prediction-text');
    
    if (data.image && currentImage && predictionText) {
        // Set image source and make it visible
        currentImage.src = `data:image/jpeg;base64,${data.image}`;
        currentImage.style.display = 'block';  // Make sure image is visible
        
        // Update prediction text
        if (data.prediction !== undefined && data.confidence !== undefined) {
            predictionText.textContent = 
                `Prediction: ${data.prediction} (${data.confidence.toFixed(2)}%)`;
        }
        
        // Log to console for debugging
        console.log('Image updated:', currentImage.src.slice(0, 50) + '...');
        console.log('Prediction:', data.prediction, 'Confidence:', data.confidence);
    }
    
    // Update chart with sliding window
    if (trainingChart) {
        const label = `E${data.current_epoch}B${data.current_batch}`;
        
        // Add new data points
        trainingChart.data.labels.push(label);
        trainingChart.data.datasets[0].data.push(data.loss);
        trainingChart.data.datasets[1].data.push(data.accuracy);
        
        // Keep only the last MAX_DATA_POINTS points
        if (trainingChart.data.labels.length > MAX_DATA_POINTS) {
            // Remove oldest data point
            trainingChart.data.labels.shift();
            trainingChart.data.datasets[0].data.shift();
            trainingChart.data.datasets[1].data.shift();
        }
        
        // Update chart with animation
        trainingChart.update('none');  // 'none' disables animation for smoother updates
    }
}

document.addEventListener('DOMContentLoaded', function() {
    initializeChart();
    
    const form = document.getElementById('training-form');
    const modelType = document.getElementById('modelType');
    const cnnOptions = document.getElementById('cnn-options');

    if (modelType && cnnOptions) {
        modelType.addEventListener('change', function() {
            cnnOptions.style.display = this.value === 'cnn' ? 'block' : 'none';
        });
    }

    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Get form elements with null checks
            const optimizerElement = document.getElementById('optimizer');
            const batchSizeElement = document.getElementById('batchSize');
            const epochsElement = document.getElementById('epochs');
            const datasetElement = document.getElementById('dataset');
            const kernelsElement = document.getElementById('kernels1');

            // Validate required elements exist
            if (!modelType || !optimizerElement || !batchSizeElement || 
                !epochsElement || !datasetElement) {
                console.error('Required form elements are missing');
                return;
            }

            // Reset UI
            if (trainingChart) {
                trainingChart.data.labels = [];
                trainingChart.data.datasets.forEach(dataset => dataset.data = []);
                trainingChart.update();
            }
            
            const epochText = document.getElementById('epoch-text');
            const batchText = document.getElementById('batch-text');
            const progressFill = document.getElementById('overall-progress-fill');
            
            if (epochText) epochText.textContent = 'Epoch: 0/0';
            if (batchText) batchText.textContent = 'Batch: 0/0';
            if (progressFill) progressFill.style.width = '0%';
            
            const currentImage = document.getElementById('current-image');
            const predictionText = document.getElementById('prediction-text');
            if (currentImage) {
                currentImage.src = '';
                currentImage.style.display = 'none';
            }
            if (predictionText) predictionText.textContent = '';

            // Prepare form data
            const formData = {
                modelType: modelType.value,
                optimizer: optimizerElement.value,
                batchSize: batchSizeElement.value,
                epochs: epochsElement.value,
                dataset: datasetElement.value,
            };

            // Add kernels for CNN if applicable
            if (modelType.value === 'cnn' && kernelsElement) {
                formData.kernels1 = kernelsElement.value;
            }

            try {
                const response = await fetch('/train', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });

                if (response.ok) {
                    setupEventSource();
                } else {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Failed to start training');
                }
            } catch (error) {
                console.error('Error:', error);
                const logsDiv = document.getElementById('logs');
                if (logsDiv) {
                    logsDiv.innerHTML += `<div class="error">Error: ${error.message}</div>`;
                }
            }
        });
    } else {
        console.error('Training form not found');
    }
}); 