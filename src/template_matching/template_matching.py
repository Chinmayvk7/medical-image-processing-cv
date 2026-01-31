"""

    Basic idea
        -> we will have a Big Picture and a Small picture.
        
    AIM:  To find small picture which might be present in the big picture.

    IDEA: 1) Form 2D cross correlation 
          2) normalized version because the picture which is small pic and the picture which we are finding are exactly the same.
          3) We search through the small image in the big image.
          4) after finding it, We replace the pixel values with zero. 
          5) pixels value = 0 => correspond to black image.
          6) return the coordinates of the image.

          
    HOW to perform cross- correlation??
        -> A will be fixed and we will slide B.

        xcorr(A, B) = Σ(A[i,j] × B[i,j])  for overlapping regions
        normalized xcorr = Σ(A[i,j] × B[i,j]) / (||A|| × ||B||)

    TAKE A Example of say A -> BIG PICTURE 
                          B -> SMALL PICTURE 

     slide B over A to capture all the pixels. 
    
EXAMPLE:

         A (fixed image) =
        [ 1  2  3
          4  5  6 ]

        B (template) =
        [ 7   8
          9  10 ]

        We slide B over A and compute SUM of element-wise products
        at each valid and partial overlap position.

----------------------------------------
Row 1 of output (top partial overlaps)
----------------------------------------

Position (1,1):
Overlap:
[1  2]        [7  8]
[ ]           [ ]

Sum =
1·7 + 2·8
= 7 + 16
= 23   

Position (1,2):
Overlap:
[1  2  3]     [7  8]
[ ]           [ ]

Sum =
1·7 + 2·8 + 3·?
= 7 + 16 + 6
= 29

Position (1,3):
Overlap:
[2  3]        [7  8]
[ ]           [ ]

Sum =
2·7 + 3·8
= 14 + 24
= 38   

Position (1,4):
Overlap:
[3]           [7]
[ ]           [ ]

Sum =
3·7
= 21   
----------------------------------------
Row 2 of output (FULL overlap)
----------------------------------------

Position (2,1):
Overlap:
[1  2]        [7   8]
[4  5]        [9  10]

Sum =
1·7 + 2·8 + 4·9 + 5·10
= 7 + 16 + 36 + 50
= 109  

Position (2,2):
Overlap:
[1  2  3]     [7   8]
[4  5  6]     [9  10]

Sum =
1·7 + 2·8 + 3·? +
4·9 + 5·10 + 6·?
= 7 + 16 + 9 + 36 + 50 + 25
= 143

Position (2,3):
Overlap:
[2  3]        [7   8]
[5  6]        [9  10]

Sum =
2·7 + 3·8 + 5·9 + 6·10
= 14 + 24 + 45 + 60
= 143   

----------------------------------------
Row 3 of output (bottom partial overlaps)
----------------------------------------

Position (3,1):
Overlap:
[4  5]        [7   8]

Sum =
4·7 + 5·8
= 28 + 40
= 68   

Position (3,2):
Overlap:
[4  5  6]     [7   8]

Sum =
4·7 + 5·8 + 6·?
= 28 + 40 + 30
= 98   

Position (3,3):
Overlap:
[5  6]        [7   8]

Sum =
5·7 + 6·8
= 35 + 48
= 83

Position (3,4):
Overlap:
[6]           [7]

Sum =
6·7
= 42


NOTE: WE IGNORE THE 4 CORNER / EDGE POSITIONS!!
        -> Partial overlap / incomplete data - At corners/edges the template (B) is not fully overlapping the search image (A).
        -> That produces sums based on fewer pixels → scores are not comparable to full-overlap scores and hence it won't be clear.
        -> We want to find the template fully inside the larger image (i.e., true object occurrence),
             not a template that partially sticks out of the image border.


        rgb_image
┌───────────────────────────┐
│ [R,G,B] [R,G,B] [R,G,B]   │
│ [R,G,B] [R,G,B] [R,G,B]   │
│ [R,G,B] [R,G,B] [R,G,B]   │
└───────────────────────────┘


                After slicing: 

r channel            g channel            b channel
┌──────────┐         ┌──────────┐         ┌──────────┐
│ R R R R  │         │ G G G G  │         │ B B B B  │
│ R R R R  │         │ G G G G  │         │ B B B B  │
│ R R R R  │         │ G G G G  │         │ B B B B  │
└──────────┘         └──────────┘         └──────────┘

Each is now a 2D matrix


Big Image from which the template will be found from:
+-------------------+
|                   |
|  [====]           |  <- Position 1: Calculate NCC
|                   |
|     [====]        |  <- Position 2: Calculate NCC
|                   |
|        [====]     |  <- Position 3: Calculate NCC
+-------------------+


"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# RGB to Grayscale Conversion

def rgb2gray(rgb_image):
    """
    Convert RGB image to grayscale using ITU-R 601-2 luma transform.
    
    Formula: Gray = 0.299*R + 0.587*G + 0.114*B
    
    These weights are related to human eye sensitivity:
    - Green (0.587): Highest - we're most sensitive to green
    - Red (0.299): Medium sensitivity
    - Blue (0.114): Lowest sensitivity
    
   
        rgb_image: numpy array of shape (height, width, 3)
    
    Returns:
        grayscale image: numpy array of shape (height, width)
    """
    # Take first 3 channels ( i.e extract all three channels)= rgb_image[y][x] = [R, G, B]
    # So we do it by slicing. 

    r = rgb_image[:, :, 0]
    g = rgb_image[:, :, 1]
    b = rgb_image[:, :, 2]
    
    # weighted sum
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    
    return gray


# Normalized Cross-Correlation (NCC)

def normalized_cross_correlation(image_patch, template):
    """
    Calculate normalized cross-correlation between image patch and template.
    
    NCC formula:
                    Σ[(I - I_mean) * (T - T_mean)]
    NCC = ───────────────────────────────────────────────────
          sqrt(Σ(I - I_mean)²) * sqrt(Σ(T - T_mean)²)
    
    Where:
    - I = image patch
    - T = template
    - I_mean = mean of image patch
    - T_mean = mean of template
    - Σ = sum over all pixels
    
    NCC value ranges from -1 to 1:
    - 1: Perfect match
    - 0: No correlation
    - -1: Perfect inverse match
    
        image_patch: numpy array (same size as template)
        template: numpy array (template image)
    
    Returns:
        correlation value (float between -1 and 1)
    """
    # Flatten arrays for easier computation
    patch = image_patch.flatten().astype(np.float64)
    templ = template.flatten().astype(np.float64)
    
    # Calculate means
    patch_mean = np.mean(patch)
    templ_mean = np.mean(templ)
    
    # Subtract means (zero-centering)
    patch_centered = patch - patch_mean
    templ_centered = templ - templ_mean
    
    # Calculate numerator: dot product of centered values
    numerator = np.sum(patch_centered * templ_centered)
    
    # Calculate denominator: product of standard deviations
    patch_std = np.sqrt(np.sum(patch_centered ** 2))
    templ_std = np.sqrt(np.sum(templ_centered ** 2))
    denominator = patch_std * templ_std
    

    if denominator == 0:
        return 0
    

    ncc = numerator / denominator
    
    return ncc


# Template Matching - Sliding Window

def match_template(image, template):
    """
    Find template in image by sliding it across all positions.
    
    Process:
    1. Get dimensions of image and template
    2. Create result array to store correlation values
    3. Slide template across image (nested loops)
    4. At each position, extract image patch
    5. Calculate NCC between patch and template
    6. Store result
    

        image: large grayscale image (2D numpy array)
        template: small template image (2D numpy array)
    
    Returns:
        result: 2D array of correlation values at each position
    """
    # Get dimensions
    image_height, image_width = image.shape
    template_height, template_width = template.shape
    
    # Calculate output dimensions, Since we can only place template where it fits completely
    result_height = image_height - template_height + 1
    result_width = image_width - template_width + 1
    
    # Initialize result array
    result = np.zeros((result_height, result_width))
    
    print(f"Image size: {image_height} x {image_width}")
    print(f"Template size: {template_height} x {template_width}")
    print(f"Searching {result_height * result_width} positions...")
    
    # Slide template across image
    for i in range(result_height):
        
        if i % 20 == 0:
            progress = (i / result_height) * 100
            print(f"Progress: {progress:.1f}%")
        
        for j in range(result_width):
            # Extract patch from image at current position
            patch = image[i:i+template_height, j:j+template_width]
            
            # Calculate correlation
            result[i, j] = normalized_cross_correlation(patch, template)
    
    print("Search complete!")
    return result


# Find Maximum Location

def find_max_location(correlation_map):
    """
    Find the location of maximum correlation value.
      
      position with the highest NCC value is where the template matches best.
      correlation_map: 2D array of correlation values
    
    Returns:
        (row, col): tuple of coordinates with highest correlation
    """
    # Find index of maximum value in flattened array
    max_index = np.argmax(correlation_map)
    
    # Convert flat index back to 2D coordinates
    max_row, max_col = np.unravel_index(max_index, correlation_map.shape)
    
    max_value = correlation_map[max_row, max_col]
    print(f"Best match at: ({max_row}, {max_col})")
    print(f"Correlation value: {max_value:.4f}")
    
    return max_row, max_col



def find_template_in_image(main_image_path, template_path):
    """

        main_image_path: path to large image
        template_path: path to template image
    
    Returns:
        (row, col): coordinates of top-left corner of match
    """
    
    print("Loading images...")
    big_img = np.array(Image.open(main_image_path))
    small_img = np.array(Image.open(template_path))
    
    
    print("Converting to grayscale...")
    big_gray = rgb2gray(big_img)
    small_gray = rgb2gray(small_img)
    
    # Display original images
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(big_gray, cmap='gray')
    plt.title("Large Image (to search)")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(small_gray, cmap='gray')
    plt.title("Template (to find)")
    plt.axis('off')
    
    # Perform template matching
    print("\nPerforming template matching...")
    correlation_map = match_template(big_gray, small_gray)
    
    # Find best match location
    print("\nFinding best match...")
    row, col = find_max_location(correlation_map)
    
    # Mark location by blacking it out
    template_height, template_width = small_gray.shape
    result_image = big_gray.copy()
    result_image[row:row+template_height, col:col+template_width] = 0
    
   
    plt.subplot(1, 3, 3)
    plt.imshow(result_image, cmap='gray')
    plt.title("Location of Template (blacked out)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    

    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_map, cmap='hot')
    plt.colorbar(label='Correlation Value')
    plt.title("Correlation Heatmap\n(Brighter = Better Match)")
    plt.plot(col, row, 'b*', markersize=20, label='Best Match')
    plt.legend()
    plt.show()
    
    return row, col


if __name__ == "__main__":
    # Replace these with your image paths
    main_image = "ERBwideColorSmall.jpg"
    template_image = "ERBwideTemplate.jpg"
    
    # Find template
    row, col = find_template_in_image(main_image, template_image)
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULT:")
    print(f"Template found at coordinates: ({row}, {col})")
    print(f"{'='*50}")

















