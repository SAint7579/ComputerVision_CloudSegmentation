import glob
import rasterio
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.segmentation import slic
from scipy.ndimage import center_of_mass
from fast_slic import Slic
import pandas as pd

## For MDS
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
# mds = MDS(n_components=1, random_state=0, normalized_stress='auto')
pca = PCA(n_components=1)


def convert_to_xy(image,patch,n_segments=300):
    agg, segments = fast_image_to_slic(image, patch, n_segments=n_segments, compactness=10)

    ## Getting the X and y arrays
    X_array = agg[['R', 'G', 'B', 'x', 'y', 'num_pixels']].values
    y_array = np.array(agg['labels'])

    ## Normalizing the X_array columwise
    X_array[:,0] = X_array[:,0]/255
    X_array[:,1] = X_array[:,1]/255
    X_array[:,2] = X_array[:,2]/255
    X_array[:,3] = X_array[:,3]/512
    X_array[:,4] = X_array[:,4]/512
    X_array[:,5] = X_array[:,5]/1000

    ## Ordering
    ordering = pca.fit_transform(X_array[:,3:5]).reshape(-1)
    X_array = X_array[ordering.argsort()]
    y_array = y_array[ordering.argsort()]


    ## Pad the X_array with -1 and y_array with 0 upto 300
    X_array = np.pad(X_array,((0,n_segments-X_array.shape[0]),(0,0)),mode='constant',constant_values=-1)
    y_array = np.pad(y_array,(0,n_segments-y_array.shape[0]),mode='constant',constant_values=0).reshape(-1,1)

    return X_array, y_array, ordering.argsort(), segments


def unpad_and_map(X,y,original_ordering,slic_map):
    '''
    Reverse the padding and ordering to get the original image

    Args:
        X (numpy.ndarray): Input X array.
        y (numpy.ndarray): Input y array.
        original_ordering (numpy.ndarray): Ordering of the superpixels.
        slic_map (numpy.ndarray): SLIC map of the image.

    Returns:
        map (numpy.ndarray): Segmentation map of the image with the original ordering.
    '''
    label = y[[X[:,0] != -1]]
    label = label[original_ordering.argsort()]
    map = reverse_segmentation(slic_map,label)
    return map


def image_to_dataframe(image_array):
    """
    Convert an image array to a dataframe with columns for x, y, R, G, and B values.

    Args:
        image_array (numpy.ndarray): Input RGB image array.

    Returns:
        dataframe (pd.DataFrame): Dataframe with columns for x, y, R, G, and B values.
    """
    height, width, _ = image_array.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    
    pixel_data = image_array.reshape(-1, 3)
    x_coords_flat = x_coords.ravel()
    y_coords_flat = y_coords.ravel()
    
    data = {
        'x': x_coords_flat,
        'y': y_coords_flat,
        'R': pixel_data[:, 0],
        'G': pixel_data[:, 1],
        'B': pixel_data[:, 2]
    }
    
    dataframe = pd.DataFrame(data)
    return dataframe

def fast_image_to_slic(image_array, binary_array, n_segments=300, compactness=10):

    """
    Converts an image and it's binary mask to a superpixel representation using SLIC. Takes in total 70 milliseconds

    Args:
        image_array (numpy.ndarray): Input RGB image array.
        binary_array (numpy.ndarray): Binary array of the same dimensions as the image.
        n_segments (int, optional): Number of segments in SLIC. Default is 100.
        compactness (float, optional): Compactness parameter for SLIC. Default is 10.

    Returns:
        aggregated (pd.DataFramw): All aspects of the superpixels
        slic_frame (numpy.ndarray): SLIC Output
    """
    slic = Slic(num_components=n_segments, compactness=compactness)
    slic_frame = slic.iterate(image_array)
    # slic_frame = slic(image_array, n_segments=n_segments, compactness=compactness, sigma=sigma)

    ## Creating dataframe
    image_frame = image_to_dataframe(image_array)
    image_frame['labels'] = binary_array.reshape(-1)
    image_frame['slic'] = slic_frame.reshape(-1)

    ## Aggregating
    aggregated = image_frame.groupby('slic').mean()
    aggregated['num_pixels'] = image_frame.groupby('slic').count()['x']
    aggregated['labels'] = (aggregated['labels']>0.5).astype(int)

    return aggregated,slic_frame


def reverse_segmentation(slic_object,labels):
    segmented_binary_array=np.zeros((512,512))
    for segment_id in np.unique(slic_object):
        mask = slic_object == segment_id
        if np.any(mask):
            segmented_binary_array[mask]=labels[segment_id]
        
    return segmented_binary_array



def get_patch(path_to_folders_images = "Natural_False_Color/", path_to_folders_labels = "Entire_scene_gts/"):
    # For images
    true_dataset = []

    tif_files = glob.glob(path_to_folders_images + "*.TIF")

    # Iterate through the .tif files and read them using rasterio
    for tif_file in tif_files:
        temp=rasterio.open(tif_file)
        red_band = temp.read(1)
        green_band = temp.read(2)
        blue_band = temp.read(3)
        temp.close()
        true = np.stack((red_band, green_band, blue_band), axis=-1)
        true_dataset.append(true)
        

    # For labels
    label_dataset = []

    tif_files = glob.glob(path_to_folders_labels + "*.TIF")

    # Iterate through the .tif files and read them using rasterio
    for tif_file in tif_files:
        temp=rasterio.open(tif_file)
        label = temp.read(1)
        temp.close()
        label_dataset.append(label)
        
            
            
    # making patches
    patch_size = 512
    num_rows = 16
    num_cols = 15

    true_patches = []
    label_patches = []

    for i in range(len(true_dataset)):
        for row in range(num_rows):
            for col in range(num_cols):
                start_row = row * patch_size
                end_row = start_row + patch_size
                start_col = col * patch_size
                end_col = start_col + patch_size
                patch = true_dataset[i][start_row:end_row, start_col:end_col]
                if any(dim == 0 for dim in patch.shape):
                    continue
                patch = cv2.resize(patch, (512, 512))
                label_patch = label_dataset[i][start_row:end_row, start_col:end_col]
                label_patch = cv2.resize(label_patch, (512, 512))
                if calculate_black_pixel_percentage(patch) > 1  or calculate_black_pixel_percentage(label_patch) > 98 or np.sum(label_patch)==262144:
                    continue
                true_patches.append(patch)
                label_patches.append(label_patch)

    true_patches = np.array(true_patches)
    label_patches = np.array(label_patches)

    return true_patches, label_patches




## APPENDIX
def calculate_black_pixel_percentage(image_array):
    """
    Calculate the percentage of black pixels (intensity 0) in the input image array.

    Args:
        image_array (numpy.ndarray): Grayscale or binary image array.

    Returns:
        percentage (float): Percentage of black pixels in the image.
    """
    if image_array.size == 0:
        return 0.0
    black_pixels = np.count_nonzero(image_array == 0)
    total_pixels = image_array.size
    percentage = (black_pixels / total_pixels) * 100
    return percentage

def convert_image_array_to_slic_with_properties(image_array, binary_array, n_segments=100, compactness=10, sigma=1):
    """
    Convert an image array to a segmented RGB image using SLIC (Simple Linear Iterative Clustering) algorithm,
    and calculate properties for each superpixel including color, centroid x, centroid y, and number of pixels.

    Args:
        image_array (numpy.ndarray): Input RGB image array.
        binary_array (numpy.ndarray): Binary array of the same dimensions as the image.
        n_segments (int, optional): Number of segments in SLIC. Default is 100.
        compactness (float, optional): Compactness parameter for SLIC. Default is 10.
        sigma (float, optional): Sigma parameter for SLIC. Default is 1.

    Returns:
        segmented_image_rgb (numpy.ndarray): Segmented RGB image.
        properties (numpy.ndarray): Array of dictionaries containing properties for each superpixel.
        labels (numpy.ndarray): Array of labels for each superpixel.
    """
    segments = slic(image_array, n_segments=n_segments, compactness=compactness, sigma=sigma)
    segmented_image_rgb = np.zeros_like(image_array)
    segmented_binary_array=np.zeros_like(binary_array)
    properties = []  # List to store properties
    labels=[]
    
    for segment_id in np.unique(segments):
        mask = segments == segment_id
        segment_rgb = image_array[mask]

        segmented_pixels = binary_array[mask]
        
        
        if np.any(mask):
            num_pixels = np.sum(mask)
            centroid = center_of_mass(mask)
            centroid_x, centroid_y = centroid
            mean_color = np.mean(segment_rgb, axis=0)
            
            mean_color_binary_array=np.bincount(segmented_pixels).argmax()
            

            properties.append({
                'superpixel_num' : segment_id,
                'color': mean_color,  # RGB color values
                'centroid_x': centroid_x,  # x coordinate of centroid
                'centroid_y': centroid_y,  # y coordinate of centroid
                'num_pixels': num_pixels, # number of pixels in superpixel
                'label' : mean_color_binary_array
            })
            labels.append(mean_color_binary_array)

            segmented_image_rgb[mask] = mean_color
            segmented_binary_array[mask]=mean_color_binary_array

    properties = np.array(properties)  # Convert properties list to numpy array
    return segmented_image_rgb, properties, labels, segmented_binary_array, segments