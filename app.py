import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
import cv2

from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# =========================================================
# CONFIGURATION
# =========================================================

IMG_SIZE = 224
NUM_CLASSES = 5

MODEL_NAME = "efficientnet_b3"

MODEL_PATH = "best_model.pth"

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

CLASS_DESCRIPTIONS = {

    "No DR":
    "No signs of diabetic retinopathy detected.",

    "Mild DR":
    "Early stage diabetic retinopathy detected.",

    "Moderate DR":
    "Moderate retinal blood vessel damage detected.",

    "Severe DR":
    "Severe diabetic retinopathy detected.",

    "Proliferative DR":
    "Advanced diabetic retinopathy detected."
}

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="centered"
)

# =========================================================
# MODEL DEFINITION
# =========================================================

class RetinopathyModel(nn.Module):

    def __init__(self, model_name, num_classes):

        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes
        )

    def forward(self, x):

        return self.model(x)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = RetinopathyModel(
        MODEL_NAME,
        NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    return model, device

# =========================================================
# IMAGE TRANSFORMS
# =========================================================

def get_transforms():

    return A.Compose([

        A.Resize(
            IMG_SIZE,
            IMG_SIZE
        ),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2()

    ])

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(image, model, device):

    transform = get_transforms()

    image_np = np.array(image)

    transformed = transform(image=image_np)

    tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, pred_class = torch.max(
            probabilities,
            dim=1
        )

    pred_idx = pred_class.item()

    confidence = confidence.item() * 100

    return (
        pred_idx,
        confidence,
        probabilities.cpu().numpy()[0]
    )

# =========================================================
# GRAD-CAM FUNCTION
# =========================================================

def generate_gradcam(image, model, device):

    transform = get_transforms()

    # Original image
    image_np = np.array(image)

    # Resize image for model
    resized_image = cv2.resize(
        image_np,
        (IMG_SIZE, IMG_SIZE)
    )

    transformed = transform(
        image=resized_image
    )

    input_tensor = transformed["image"].unsqueeze(0).to(device)

    # Normalize resized image
    rgb_img = resized_image.astype(np.float32) / 255.0

    # Target layer
    target_layers = [model.model.blocks[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    # Resize CAM properly
    grayscale_cam = cv2.resize(
        grayscale_cam,
        (IMG_SIZE, IMG_SIZE)
    )

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    return visualization

# =========================================================
# MAIN APP
# =========================================================

def main():

    st.title("👁️ Diabetic Retinopathy Detection")

    st.write(
        "Upload a retinal fundus image and the AI model will predict the stage of diabetic retinopathy."
    )

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    try:

        model, device = load_model()

    except Exception as e:

        st.error(
            f"Error loading model: {e}"
        )

        st.stop()

    # -----------------------------------------------------
    # FILE UPLOAD
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload Retinal Image",
        type=["jpg", "jpeg", "png"]
    )

    # -----------------------------------------------------
    # IF IMAGE IS UPLOADED
    # -----------------------------------------------------

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Retinal Image",
            use_container_width=True
        )

        # -------------------------------------------------
        # PREDICT BUTTON
        # -------------------------------------------------

        if st.button("Predict"):

            with st.spinner(
                "Analyzing image..."
            ):

                pred_idx, confidence, probabilities = predict_image(
                    image,
                    model,
                    device
                )

            predicted_class = CLASS_NAMES[pred_idx]

            # -------------------------------------------------
            # RESULTS
            # -------------------------------------------------

            st.success(
                f"Prediction: {predicted_class}"
            )

            st.info(
                f"Confidence: {confidence:.2f}%"
            )

            st.write(
                CLASS_DESCRIPTIONS[predicted_class]
            )

            # -------------------------------------------------
            # PROBABILITY BARS
            # -------------------------------------------------

            st.subheader(
                "Class Probabilities"
            )

            for i, prob in enumerate(probabilities):

                st.write(
                    f"{CLASS_NAMES[i]} : {prob * 100:.2f}%"
                )

                st.progress(float(prob))

            # -------------------------------------------------
            # GRAD-CAM
            # -------------------------------------------------

            # -------------------------------------------------
            # AI ATTENTION MAP
            # -------------------------------------------------
            
            st.subheader(
                "🩺 AI Attention Map"
            )
            
            st.markdown(
                """
                The highlighted regions below show the retinal areas the AI model focused on 
                most strongly while making its prediction. 
                
                - 🔴 Red/Yellow regions → High attention
                - 🔵 Blue/Dark regions → Low attention
                
                This helps visualize which retinal features influenced the diagnosis.
                """
            )
            
            try:
            
                gradcam_image = generate_gradcam(
                    image,
                    model,
                    device
                )
            
                st.image(
                    gradcam_image,
                    caption="Model Focus Visualization",
                    use_container_width=True
                )
            
            except Exception as e:
            
                st.error(
                    f"Visualization Error: {e}"
                )
            
# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    main()
