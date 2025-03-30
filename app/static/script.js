async function handleSubmit() {
    const features = [
        parseFloat(document.getElementById('age').value),
        parseFloat(document.getElementById('gender').value),
        parseFloat(document.getElementById('chestPainType').value),
        parseFloat(document.getElementById('bloodPressure').value),
        parseFloat(document.getElementById('cholesterol').value),
        parseFloat(document.getElementById('fbs').value),
        parseFloat(document.getElementById('restecg').value),
        parseFloat(document.getElementById('thalach').value),
        parseFloat(document.getElementById('exang').value),
        parseFloat(document.getElementById('oldpeak').value),
        parseFloat(document.getElementById('slope').value),
        parseFloat(document.getElementById('ca').value),
        parseFloat(document.getElementById('thal').value)
    ];

    const loadingBar = document.getElementById('loadingBar');
    const progressBar = document.getElementById('progressBar');
    
    // Show the loading bar and start the animation
    loadingBar.style.display = 'block';
    progressBar.style.width = '10%';

    try {
        // Simulate loading bar progress
        const updateProgress = setInterval(() => {
            if (parseInt(progressBar.style.width) < 80) {
                progressBar.style.width = parseInt(progressBar.style.width) + 10 + '%';
            }
        }, 400);

        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ features })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        
        // Stop the progress bar increment
        clearInterval(updateProgress);
        progressBar.style.width = '100%';

        // Store the prediction result in localStorage
        localStorage.setItem('risk', data.risk);

        // Redirect to result page after a brief delay
        setTimeout(() => {
            window.location.href = 'result.html';
        }, 500);
        
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while predicting the risk.');
    } finally {
        // Reset loading bar after completion or error
        setTimeout(() => {
            loadingBar.style.display = 'none';
            progressBar.style.width = '0';
        }, 1000);
    }
}
