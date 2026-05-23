# Diabetic Retinopathy Web App

A deep learning-powered web application for automated **Diabetic Retinopathy (DR) detection** from retinal fundus images. The application helps identify signs of diabetic retinopathy using a trained CNN model and provides visual explanations using Grad-CAM heatmaps for better interpretability.

## Live Demo

Deployment Link:
[Diabetic Retinopathy Web App Live Demo](https://diabetic-retinopathy-by-deepta.streamlit.app/?utm_source=chatgpt.com)

## Features

* Upload retinal fundus images for prediction
* Detect diabetic retinopathy severity using deep learning
* Grad-CAM visualization for explainability
* Interactive and user-friendly web interface
* Fast inference with optimized model deployment
* Medical image preprocessing pipeline

## About Diabetic Retinopathy

Diabetic Retinopathy is a diabetes-related eye disease caused by damage to retinal blood vessels and can lead to vision loss if untreated. Early detection using AI-based systems can significantly improve screening efficiency and accessibility.

## Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Deep Learning:** TensorFlow / Keras
* **Image Processing:** OpenCV, NumPy
* **Visualization:** Grad-CAM
* **Model Architecture:** CNN-based classifier

## Project Structure

```bash id="dc0a3l"
Diabetic-Retinopathy-Webapp/
│── app.py                  # Main Streamlit application
│── model/                  # Trained model files
│── utils/                  # Helper functions
│── assets/                 # Images and UI assets
│── requirements.txt        # Python dependencies
│── README.md               # Project documentation
```

## Installation

Clone the repository:

```bash id="l4oc9h"
git clone https://github.com/Deepta-Das/Diabetic-Retinopathy-Webapp.git
cd Diabetic-Retinopathy-Webapp
```

Create a virtual environment (recommended):

```bash id="u5b9y6"
python -m venv venv
```

Activate the environment:

### Windows

```bash id="d5nzn8"
venv\Scripts\activate
```

### Linux / macOS

```bash id="jlwmn0"
source venv/bin/activate
```

Install dependencies:

```bash id="57v8nj"
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit server:

```bash id="jlwmn0"
streamlit run app.py
```

The app will run locally at:

```bash id="4p6glt"
http://localhost:8501
```

## How It Works

1. Upload a retinal fundus image
2. The image is preprocessed and passed to the trained CNN model
3. The model predicts the DR severity level
4. Grad-CAM generates a heatmap highlighting important retinal regions
5. Results are displayed in the web interface

## Model Explainability

The application uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to improve interpretability by showing which retinal regions influenced the model’s prediction. Explainable AI techniques improve trust and usability in medical AI systems.

## Possible DR Severity Classes

Depending on the trained model and dataset:

* No DR
* Mild DR
* Moderate DR
* Severe DR
* Proliferative DR

## Future Improvements

* Multi-class classification improvements
* Cloud deployment support
* Mobile-friendly interface
* Integration with real-time screening systems
* Support for additional retinal diseases

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch

```bash id="qjn61c"
git checkout -b feature-name
```

3. Commit changes

```bash id="fw9b4r"
git commit -m "Added new feature"
```

4. Push to branch

```bash id="prvk6x"
git push origin feature-name
```

5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Author

Developed by [Deepta Das](https://github.com/Deepta-Das?utm_source=chatgpt.com)

## Support

If you found this project useful, consider starring the repository:

