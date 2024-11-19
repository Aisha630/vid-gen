import os
import lpips

def calculate_lpips_distance(dir0, dir1, use_gpu=False):
    """
    Calculate LPIPS distance between images in two directories and save results to a file.
    
    Parameters:
    - dir0 (str): Path to the first directory of images.
    - dir1 (str): Path to the second directory of images.
    - out_file (str): Path to the output text file for saving distances.
    - version (str): Version of the LPIPS model to use ('0.1' or '0.2').
    - use_gpu (bool): Flag to use GPU if available.
    """
    # Initialize the LPIPS model
    loss_fn = lpips.LPIPS(net='vgg', version='0.1')
    if use_gpu:
        loss_fn.cuda()


    # List all files in dir0
    files = os.listdir(dir0)

    total_sum = 0
    total_count = 0
    for file in files:
        # Check if the corresponding file exists in dir1
        if os.path.exists(os.path.join(dir1, file)):
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(dir0, file)))  # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(dir1, file)))

            # Move to GPU if required
            if use_gpu:
                img0 = img0.cuda()
                img1 = img1.cuda()

            # Compute LPIPS distance
            dist01 = loss_fn.forward(img0, img1)
            # print(f'{file}: {dist01.item():.3f}')
            total_sum = total_sum + dist01.item()
            total_count = total_count + 1

    average = total_sum/total_count
    print(f'Average LPIPs loss: {average}')
