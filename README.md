---
title: Brain Tumor Segmentation
emoji: 🧠
colorFrom: blue
colorTo: pink
sdk: streamlit
sdk_version: 1.48.1
app_file: app.py
pinned: false
license: mit
---

## Sample Data

You can download sample NIfTI files for two patients to test the model from this Google Drive link:

[Sample Data (Google Drive)](https://drive.google.com/drive/folders/19LzKOcoIrWQhwY91e_kn644AcQi4tl8z?usp=sharing)

## Sample Data

You can download sample NIfTI files for two patients to test the model from this Google Drive link:

[Sample Data (Google Drive)](https://drive.google.com/drive/folders/19LzKOcoIrWQhwY91e_kn644AcQi4tl8z?usp=sharing)

---
title: Brain Tumor Segmentation
emoji: 🧠
colorFrom: blue
colorTo: pink
sdk: streamlit
sdk_version: 1.48.1
app_file: app.py
pinned: false
license: mit
---

# Brain Tumor Segmentation App

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-Online-brightgreen" alt="Streamlit">
  <a href="https://huggingface.co/spaces/saketh-005/brain-tumor-segmentation"><img src="https://img.shields.io/badge/HuggingFace-Spaces-yellow" alt="Hugging Face Spaces"></a>
</p>

This project is a web application for brain tumor segmentation from 3D/4D NIfTI MRI scans using a 3D U-Net model, built with PyTorch and Streamlit. You can run it locally or deploy it on [Hugging Face Spaces](https://huggingface.co/spaces).

## Features
- Upload four 3D NIfTI brain scans (T1, T1ce, T2, FLAIR)
- Automatic preprocessing and patch-based inference
- Visualizes the predicted tumor mask overlayed on the MRI

## Quick Start (Hugging Face Spaces)

1. **Upload your trained model file** (`unet3d_model.pth`) to the Space's root directory or use a download link in the code.
2. Click "Run" or "Duplicate Space" to use your own model.
3. Use the web interface to upload your NIfTI files and view results.

## Local Usage
1. Clone this repository:
   ```sh
   git clone https://github.com/saketh-005/brain-tumor-segmentation.git
   cd brain-tumor-segmentation
   ```
2. (Recommended) Create and activate a Python virtual environment:
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Download the trained model file (`unet3d_model.pth`) and place it in this directory. (Due to file size, it is not included in the repo. Please contact the author or use your own trained model.)
5. Run the app:
   ```sh
   streamlit run app.py
   ```
6. Open your browser to [http://localhost:8501](http://localhost:8501) and use the app.

## File Structure
- `app.py` - Main Streamlit app
- `unet_model.py` - 3D U-Net model definition
- `utils.py` - Preprocessing, postprocessing, and visualization utilities
- `requirements.txt` - Python dependencies
- `unet3d_model.pth` - Trained model weights (**not included**)

## Notes
- The model file (`unet3d_model.pth`) must be trained and exported separately.
- For large files, use cloud storage and provide a download link in this README or in your Hugging Face Space.
- For best results, ensure all input NIfTI files have the same dimensions and orientation.

## License
MIT License

## Author
[Saketh Jangala](https://github.com/saketh-005)
