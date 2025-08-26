# --- Patch-based Inference Helper ---
def run_patch_inference(model, tensor, patch_depth=32):
    """
    Run model inference on 3D tensor in patches along the depth axis.
    Args:
        model: The 3D segmentation model.
        tensor: Input tensor of shape [1, 4, D, H, W].
        patch_depth: Depth of each patch.
    Returns:
        Output tensor stitched together.
    """
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
    _, c, d, h, w = tensor.shape
    output = []
    for start in range(0, d, patch_depth):
        end = min(start + patch_depth, d)
        patch = tensor[:, :, start:end, :, :]
        with torch.no_grad():
            patch_out = model(patch.to(device))
        output.append(patch_out.cpu())
    # Concatenate along the depth axis
    return torch.cat(output, dim=2)
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import os
import io
import tempfile

from utils import preprocess_nifti, postprocess_mask, visualize_prediction, combine_nifti_files

# --- Page Configuration ---
st.set_page_config(
    page_title="Brain Tumor Segmentation App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title and Description ---
st.title("Brain Tumor Segmentation")
st.write("Upload the four 3D NIfTI brain scans (.nii or .nii.gz) for each modality to get a segmentation mask of the tumor.")
st.markdown("---")

# --- Model Architecture ---
# A single block in the U-Net architecture.
class DoubleConv(nn.Module):
    """(convolution => GroupNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 3D convolutional layers, GroupNorm for stable training, and ReLU activation.
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=out_channels // 2, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=out_channels // 2, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# The downsampling part of the U-Net.
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)

# The upsampling part of the U-Net.
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Use bilinear upsampling and then a convolution layer
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to match the size of x2 for concatenation
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffZ // 2, diffZ - diffZ // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffX // 2, diffX - diffX // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# The final output convolutional layer.
class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Out, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# The complete 3D U-Net model.
class UNet3d(nn.Module):
    def __init__(self, n_channels=4, n_classes=3):
        super().__init__()
        # The number of classes is 3 (tumor core, edema, enhancing tumor).
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Contracting path
        self.conv = DoubleConv(n_channels, 16)
        self.enc1 = Down(16, 32)
        self.enc2 = Down(32, 64)
        self.enc3 = Down(64, 128)
        self.enc4 = Down(128, 256)

        # Expansive path
        self.dec1 = Up(256 + 128, 128)
        self.dec2 = Up(128 + 64, 64)
        self.dec3 = Up(64 + 32, 32)
        self.dec4 = Up(32 + 16, 16)
        
        self.out = Out(16, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        
        x = self.dec1(x5, x4)
        x = self.dec2(x, x3)
        x = self.dec3(x, x2)
        x = self.dec4(x, x1)
        
        logits = self.out(x)
        return logits
        
# --- Model Loading ---
@st.cache_resource
def load_model(model_path):
    """Loads the trained PyTorch model from a .pth file."""
    try:
        # FIX: Directly load the model object, which is what was saved.
        # The weights_only=False argument is needed for custom classes.
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        model.eval()
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Main App Logic ---
model_file_path = "unet3d_model.pth"
if not os.path.exists(model_file_path):
    st.warning("Model file 'unet3d_model.pth' not found. Please ensure it is in the same directory.")
    model = None
else:
    model = load_model(model_file_path)

st.sidebar.header("Upload NIfTI Files")
t1_file = st.sidebar.file_uploader("Choose a T1 scan (.nii or .nii.gz)", type=["nii", "gz"], key="t1")
t1ce_file = st.sidebar.file_uploader("Choose a T1ce scan (.nii or .nii.gz)", type=["nii", "gz"], key="t1ce")
t2_file = st.sidebar.file_uploader("Choose a T2 scan (.nii or .nii.gz)", type=["nii", "gz"], key="t2")
flair_file = st.sidebar.file_uploader("Choose a FLAIR scan (.nii or .nii.gz)", type=["nii", "gz"], key="flair")

if t1_file and t1ce_file and t2_file and flair_file and model is not None:
    st.info("All files uploaded successfully. Processing...")
    
    # temp_combined_file_path is now defined at the start of the block
    temp_combined_file_path = None
    
    with st.spinner("Combining NIfTI files and making prediction..."):
        try:
            # Create temporary files for each uploaded file
            with tempfile.NamedTemporaryFile(suffix=f"_{t1_file.name}") as t1_temp, \
                 tempfile.NamedTemporaryFile(suffix=f"_{t1ce_file.name}") as t1ce_temp, \
                 tempfile.NamedTemporaryFile(suffix=f"_{t2_file.name}") as t2_temp, \
                 tempfile.NamedTemporaryFile(suffix=f"_{flair_file.name}") as flair_temp:

                t1_temp.write(t1_file.getvalue())
                t1ce_temp.write(t1ce_file.getvalue())
                t2_temp.write(t2_file.getvalue())
                flair_temp.write(flair_file.getvalue())

                # Pass the temporary file paths to the combine function
                combined_nifti_img = combine_nifti_files(t1_temp.name, t1ce_temp.name, t2_temp.name, flair_temp.name)

                original_data = combined_nifti_img.get_fdata()
                
                # Preprocess the combined image
                # We need to save the combined NIfTI object to a file for nibabel to load it properly
                temp_combined_file_path = "combined_4d.nii.gz"
                nib.save(combined_nifti_img, temp_combined_file_path)

                _, processed_tensor = preprocess_nifti(temp_combined_file_path)

                if original_data is not None and processed_tensor is not None:
                    st.success("Preprocessing complete!")

                    # --- Patch-based Model Prediction ---
                    st.info("Running patch-based model inference...")
                    try:
                        prediction_tensor = run_patch_inference(model, processed_tensor, patch_depth=32)
                        st.success("Prediction complete!")
                    except Exception as e:
                        st.error(f"Error during patch-based inference: {e}")
                        raise

                    # Post-process the prediction to get a mask, resizing back to original size
                    predicted_mask = postprocess_mask(prediction_tensor, original_data.shape)

                    if predicted_mask is not None:
                        st.header("Results")
                        # Ensure mask is int and shape matches for visualization
                        max_slices = original_data.shape[2]
                        slice_index = st.slider("Select an axial slice to view", 0, max_slices - 1, max_slices // 2)
                        fig = visualize_prediction(original_data, predicted_mask.astype(int), slice_index=slice_index)
                        st.pyplot(fig)
                    else:
                        st.error("Could not post-process the model's prediction.")
                
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.error("Please ensure the uploaded files are valid NIfTI files with the same dimensions.")
        finally:
            # Clean up temporary files
            if os.path.exists(temp_combined_file_path):
                os.remove(temp_combined_file_path)
            
# --- Footer ---
st.markdown("---")
st.markdown("Developed with PyTorch and Streamlit.")
