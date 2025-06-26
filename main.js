const video = document.getElementById('video');
const predictionDiv = document.getElementById('prediction');

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        video.play();
    })
    .catch(err => {
        console.error('Error accessing webcam: ', err);
    });

function sendFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg');

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: dataUrl })
    })
    .then(response => response.json())
    .then(data => {
        predictionDiv.textContent = `Prediction: ${data.prediction}`;
    })
    .catch(err => {
        console.error('Error sending frame: ', err);
    });
}

setInterval(sendFrame, 1000);  // Send frame every second
