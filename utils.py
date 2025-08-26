import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import io
import tempfile
from skimage.transform import resize

def preprocess_nifti(nifti_file):
    """
    Loads a NIfTI file, preprocesses it, and returns a PyTorch tensor.
    
    Args:
        nifti_file (str or io.BytesIO): Path to the NIfTI file or a file-like object.
    
    Returns:
        tuple: A tuple containing the original image data and a preprocessed tensor.
    """
    try:
        nifti_img = nib.load(nifti_file)
        img_data = nifti_img.get_fdata()
        
        if len(img_data.shape) != 4 or img_data.shape[-1] != 4:
            st.error("The uploaded NIfTI file must be a 4D image with 4 channels.")
            return None, None
            
        for i in range(img_data.shape[-1]):
            channel_data = img_data[..., i]
            if np.max(channel_data) > 0:
                img_data[..., i] = channel_data / np.max(channel_data)
        
        img_data = np.transpose(img_data, (3, 2, 0, 1))
        
        tensor_data = torch.from_numpy(img_data).float()
        tensor_data = torch.unsqueeze(tensor_data, 0)
        
        return nifti_img.get_fdata(), tensor_data
        
    except Exception as e:
        st.error(f"Error during NIfTI preprocessing: {e}")
        return None, None

def postprocess_mask(prediction_tensor, original_shape):
    """
    Converts model output (tensor) into a visualizable mask (numpy array)
    and resizes it to the original image dimensions.
    """
    try:
        probabilities = F.softmax(prediction_tensor, dim=1)
        
        mask = torch.argmax(probabilities, dim=1)
        
        mask = mask.detach().cpu().numpy()
        mask = np.squeeze(mask)

        mask = np.transpose(mask, (1, 2, 0))

        resized_mask = resize(mask, original_shape[:3], order=0, preserve_range=True, anti_aliasing=False)
        
        return resized_mask
    except Exception as e:
        st.error(f"Error during mask post-processing: {e}")
        return None

def visualize_prediction(original_image, predicted_mask, slice_index=75):
    """
    Creates a 2-panel visualization of the original image and the predicted mask.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    # Show FLAIR channel for the original image
    axes[0].imshow(np.rot90(original_image[:, :, slice_index, 3]), cmap='bone')
    axes[0].set_title('Original Image (FLAIR)', fontsize=16)
    axes[0].axis('off')

    axes[1].imshow(np.rot90(original_image[:, :, slice_index, 3]), cmap='bone')
    # Overlay the predicted mask directly
    mask_slice = np.rot90(predicted_mask[:, :, slice_index])
    axes[1].imshow(np.ma.masked_where(mask_slice == 0, mask_slice), cmap='jet', alpha=0.5)
    axes[1].set_title('Predicted Tumor Mask', fontsize=16)
    axes[1].axis('off')
    return fig

def combine_nifti_files(t1_file_path, t1ce_file_path, t2_file_path, flair_file_path):
    """
    Combines four 3D NIfTI files from given paths into a single 4D NIfTI file object.
    
    Args:
        t1_file_path, t1ce_file_path, t2_file_path, flair_file_path (str): Paths to the temporary NIfTI files.
        
    Returns:
        nib.Nifti1Image: A 4D NIfTI image object.
    """
    try:
        # Load the four 3D NIfTI files from file paths
        t1_img = nib.load(t1_file_path)
        t1ce_img = nib.load(t1ce_file_path)
        t2_img = nib.load(t2_file_path)
        flair_img = nib.load(flair_file_path)

        # Get the image data as NumPy arrays
        t1_data = t1_img.get_fdata()
        t1ce_data = t1ce_img.get_fdata()
        t2_data = t2_img.get_fdata()
        flair_data = flair_img.get_fdata()

        # Ensure all files have the same shape
        if not (t1_data.shape == t1ce_data.shape == t2_data.shape == flair_data.shape):
            st.error("Error: Input NIfTI files do not have matching dimensions.")
            return None

        # Stack the 3D arrays along a new (4th) dimension to create a 4D array
        combined_data = np.stack([t1_data, t1ce_data, t2_data, flair_data], axis=-1)

        # Create a new 4D NIfTI image object
        combined_img = nib.Nifti1Image(combined_data, t1_img.affine)

        return combined_img
    except Exception as e:
        st.error(f"Error combining NIfTI files: {e}")
        return None
