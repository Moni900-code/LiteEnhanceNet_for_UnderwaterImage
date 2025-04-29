def calculate_metrics_ssim_psnr(generated_image_path, ground_truth_image_path, resize_size=(256, 256)):
    """
    Calculate SSIM and PSNR metrics between generated and ground truth images.
    """
    if not os.path.exists(generated_image_path):
        print(f"Error: Generated images path does not exist: {generated_image_path}")
        return np.array([]), np.array([])

    if not os.path.exists(ground_truth_image_path):
        print(f"Error: Ground truth images path does not exist: {ground_truth_image_path}")
        return np.array([]), np.array([])

    # Map of lowercase filenames to original filenames in ground truth folder
    gt_files_map = {f.lower(): f for f in os.listdir(ground_truth_image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    generated_image_list = [f for f in os.listdir(generated_image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    error_list_ssim, error_list_psnr = [], []

    for gen_img in generated_image_list:
        gen_img_lower = gen_img.lower()
        gt_img = gt_files_map.get(gen_img_lower)

        if not gt_img:
            print(f"GT image not found for {gen_img}, skip")
            continue

        generated_image_file = os.path.join(generated_image_path, gen_img)
        ground_truth_image_file = os.path.join(ground_truth_image_path, gt_img)

        generated_image = cv2.imread(generated_image_file)
        ground_truth_image = cv2.imread(ground_truth_image_file)

        if generated_image is None:
            print(f"Error: Failed to load generated image: {generated_image_file}")
            continue
        if ground_truth_image is None:
            print(f"Error: Failed to load ground truth image: {ground_truth_image_file}")
            continue

        try:
            generated_image = cv2.resize(generated_image, resize_size)
            ground_truth_image = cv2.resize(ground_truth_image, resize_size)
        except cv2.error as e:
            print(f"Error resizing images for {gen_img}: {e}")
            continue

        try:
            error_ssim, _ = structural_similarity(
                generated_image,
                ground_truth_image,
                full=True,
                multichannel=True,
                channel_axis=2
            )
            error_list_ssim.append(error_ssim)
        except ValueError as e:
            print(f"Error calculating SSIM for {gen_img}: {e}")
            continue

        generated_image_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
        ground_truth_image_gray = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)

        try:
            error_psnr = peak_signal_noise_ratio(generated_image_gray, ground_truth_image_gray)
            error_list_psnr.append(error_psnr)
        except ValueError as e:
            print(f"Error calculating PSNR for {gen_img}: {e}")
            continue

    if not error_list_ssim or not error_list_psnr:
        print("Warning: No valid images were processed for SSIM/PSNR calculation.")
        return np.array([]), np.array([])

    return np.array(error_list_ssim), np.array(error_list_psnr)


def calculate_UIQM(image_path, resize_size=(256, 256)):
    """
    Calculate UIQM metric for images in the specified directory.
    
    Args:
        image_path (str): Path to directory containing images.
        resize_size (tuple): Target size for resizing images (width, height).
    
    Returns:
        np.array: Array of UIQM scores.
    """
    # Validate directory path
    if not os.path.exists(image_path):
        print(f"Error: Image path does not exist: {image_path}")
        return np.array([])

    image_list = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    uiqms = []

    for img in image_list:
        image_file = os.path.join(image_path, img)

        # Read image
        image = cv2.imread(image_file)
        if image is None:
            print(f"Error: Failed to load image: {image_file}")
            continue

        # Resize image
        try:
            image = cv2.resize(image, resize_size)
        except cv2.error as e:
            print(f"Error resizing image for {img}: {e}")
            continue

        # Calculate UIQM
        try:
            uiqm_score = getUIQM(image)
            uiqms.append(uiqm_score)
        except Exception as e:
            print(f"Error calculating UIQM for {img}: {e}")
            continue

    if not uiqms:
        print("Warning: No valid images were processed for UIQM calculation.")
        return np.array([])

    return np.array(uiqms)
