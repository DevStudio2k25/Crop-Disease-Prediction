// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const predictBtn = document.getElementById('predictBtn');
const resetBtn = document.getElementById('resetBtn');
const loading = document.getElementById('loading');
const resultSection = document.getElementById('resultSection');

// Event Listeners
uploadBox.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
predictBtn.addEventListener('click', predictDisease);
resetBtn.addEventListener('click', resetUpload);

// Drag and drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#764ba2';
    uploadBox.style.background = '#f0f2ff';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '#f8f9ff';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '#f8f9ff';

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Handle file selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// Handle file
function handleFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (JPG, JPEG, or PNG)');
        return;
    }

    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size should not exceed 16MB');
        return;
    }

    // Display preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadBox.style.display = 'none';
        previewSection.style.display = 'block';
        resultSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Predict disease
async function predictDisease() {
    const file = fileInput.files[0];
    if (!file) return;

    // Show loading
    loading.style.display = 'block';
    previewSection.style.display = 'none';
    resultSection.style.display = 'none';

    // Prepare form data
    const formData = new FormData();
    formData.append('file', file);

    try {
        // Send request
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            alert('Error: ' + (data.error || 'Prediction failed'));
            resetUpload();
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during prediction. Please try again.');
        resetUpload();
    } finally {
        loading.style.display = 'none';
    }
}

// Display results
function displayResults(data) {
    // Show sections
    previewSection.style.display = 'block';
    resultSection.style.display = 'block';

    // Update disease information
    document.getElementById('diseaseName').textContent = data.disease_info.name;
    document.getElementById('confidenceBadge').textContent = data.confidence + '% Confidence';
    document.getElementById('diseaseDescription').textContent = data.disease_info.description;
    document.getElementById('diseaseSeverity').textContent = 'Severity: ' + data.disease_info.severity;
    document.getElementById('diseaseRecommendation').textContent = data.disease_info.recommendation;

    // Update result card color
    const resultCard = document.querySelector('.result-card');
    resultCard.style.borderLeft = `5px solid ${data.disease_info.color}`;

    // Display probability bars
    displayProbabilityBars(data.all_predictions);

    // Scroll to results
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Display probability bars
function displayProbabilityBars(predictions) {
    const container = document.getElementById('probabilityBars');
    container.innerHTML = '';

    // Sort predictions by probability
    const sortedPredictions = Object.entries(predictions).sort((a, b) => b[1] - a[1]);

    sortedPredictions.forEach(([className, probability]) => {
        const probItem = document.createElement('div');
        probItem.className = 'prob-item';

        // Format class name
        const formattedName = className.split('_').map(word =>
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');

        probItem.innerHTML = `
            <div class="prob-label">
                <span>${formattedName}</span>
                <span>${probability.toFixed(2)}%</span>
            </div>
            <div class="prob-bar">
                <div class="prob-fill" style="width: ${probability}%"></div>
            </div>
        `;

        container.appendChild(probItem);
    });
}

// Reset upload
function resetUpload() {
    fileInput.value = '';
    uploadBox.style.display = 'block';
    previewSection.style.display = 'none';
    resultSection.style.display = 'none';
    loading.style.display = 'none';
    imagePreview.src = '';
}

// Add animation on page load
window.addEventListener('load', () => {
    document.querySelector('.hero-section').style.animation = 'fadeIn 0.5s';
    document.querySelector('.upload-section').style.animation = 'fadeIn 0.5s 0.2s both';
    document.querySelector('.features-section').style.animation = 'fadeIn 0.5s 0.4s both';
});
