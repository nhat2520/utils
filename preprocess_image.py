import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
class ImageProcessor:
    """
    Một lớp để thực hiện các thao tác tiền xử lý ảnh.

    Thuộc tính:
        image (np.ndarray): Ảnh đang được xử lý.
        original_image (np.ndarray): Ảnh gốc để có thể reset.
    """

    def __init__(self, image_path=None, image_array=None):
        """
        Khởi tạo đối tượng ImageProcessor.

        Args:
            image_path (str, optional): Đường dẫn đến file ảnh.
            image_array (np.ndarray, optional): Mảng NumPy đại diện cho ảnh.
                                                Cần cung cấp image_path hoặc image_array.
        """
        if image_path is not None:
            image = cv2.imread(image_path)
            self.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.original_image is None:
                raise ValueError(f"Không thể đọc ảnh từ: {image_path}")
        elif image_array is not None:
            self.original_image = image_array.copy()
        else:
            raise ValueError("Cần cung cấp image_path hoặc image_array.")
        self.image = self.original_image.copy()

    def _is_grayscale(self):
        """Kiểm tra xem ảnh hiện tại có phải là ảnh xám không."""
        return len(self.image.shape) == 2 or self.image.shape[2] == 1

    def reset(self):
        """Reset ảnh về trạng thái ban đầu."""
        self.image = self.original_image.copy()
        print("Ảnh đã được reset về trạng thái gốc.")
        return self

    def get_image(self):
        """Trả về ảnh đã xử lý hiện tại."""
        return self.image

    # --- Xử lý tóc bằng DullRazor ---
 
    def remove_hair_dullrazor(self, kernel_size=11, inpaint_radius=5, threshold_val=10):
        """
        Loại bỏ tóc trên ảnh bằng thuật toán DullRazor phiên bản đơn giản và hiệu quả.
        Sử dụng Black-hat morphology để phát hiện tóc và inpainting để loại bỏ tóc.

        Args:
            kernel_size (int): Kích thước kernel cho phép toán morphology. Tăng giá trị để phát hiện tóc dài hơn.  
            inpaint_radius (int): Bán kính cho inpainting. Tăng giá trị để lấp đầy vùng tóc rộng hơn.
            threshold_val (int): Ngưỡng cho việc phát hiện tóc. Giá trị càng thấp, phát hiện càng nhạy.
        """
        print("Đang loại bỏ tóc (DullRazor Đơn Giản)...")
        
        # Lưu ảnh gốc cho bước inpainting
        source_image = self.image.copy()
        
        # Chuyển sang ảnh xám nếu cần
        if len(self.image.shape) == 3 and self.image.shape[2] == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        elif len(self.image.shape) == 3 and self.image.shape[2] == 1:
            gray_image = self.image[:,:,0]
        else:
            gray_image = self.image.copy()

        # 1. Phát hiện tóc bằng black-hat morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
        
        # 2. Áp dụng ngưỡng để tạo mask tóc
        _, hair_mask = cv2.threshold(blackhat, threshold_val, 255, cv2.THRESH_BINARY)
        
        # 3. Làm sạch mask bằng morphology
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, 
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        
        # 4. Loại bỏ tóc bằng inpainting
        self.image = cv2.inpaint(source_image, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)
        
        # Đếm số vùng tóc để thông báo
        contours, _ = cv2.findContours(hair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Loại bỏ tóc (DullRazor Đơn Giản) hoàn tất. {len(contours)} vùng tóc được xử lý.")
        return self

    def clahe(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Áp dụng Contrast Limited Adaptive Histogram Equalization (CLAHE).

        Args:
            clip_limit (float): Ngưỡng giới hạn tương phản.
            tile_grid_size (tuple): Kích thước của các vùng (tiles).
        """
        clahe_processor = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        if self._is_grayscale():
            self.image = clahe_processor.apply(self.image)
        else:
            img_lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2Lab)
            l_channel, a_channel, b_channel = cv2.split(img_lab)
            l_channel_clahe = clahe_processor.apply(l_channel)
            img_lab_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
            self.image = cv2.cvtColor(img_lab_clahe, cv2.COLOR_Lab2BGR)
        print(f"Đã áp dụng CLAHE (clip_limit={clip_limit}, tile_grid_size={tile_grid_size}).")
        return self

    def gamma_correction(self, gamma=1.0):
        """
        Áp dụng hiệu chỉnh Gamma.
        Gamma < 1: Làm sáng vùng tối.
        Gamma > 1: Làm tối vùng sáng.

        Args:
            gamma (float): Giá trị gamma.
        """
        if gamma <= 0:
            print("Giá trị gamma phải lớn hơn 0.")
            return self
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        self.image = cv2.LUT(self.image, table)
        print(f"Đã áp dụng Gamma Correction (gamma={gamma}).")
        return self

    # --- 2. Giảm Nhiễu ---
    def gaussian_blur(self, kernel_size=(5, 5), sigma_x=0):
        """
        Áp dụng làm mờ Gauss.

        Args:
            kernel_size (tuple): Kích thước kernel (phải là số lẻ).
            sigma_x (float): Độ lệch chuẩn theo trục X.
        """
        if not (kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1):
            print("Kích thước kernel phải là số lẻ.")
            return self
        self.image = cv2.GaussianBlur(self.image, kernel_size, sigma_x)
        print(f"Đã áp dụng Gaussian Blur (kernel_size={kernel_size}, sigma_x={sigma_x}).")
        return self

    def median_filter(self, kernel_size=5):
        """
        Áp dụng bộ lọc Trung vị.

        Args:
            kernel_size (int): Kích thước kernel (phải là số lẻ).
        """
        if not (kernel_size % 2 == 1):
            print("Kích thước kernel phải là số lẻ.")
            return self
        self.image = cv2.medianBlur(self.image, kernel_size)
        print(f"Đã áp dụng Median Filter (kernel_size={kernel_size}).")
        return self

    def bilateral_filter(self, d=9, sigma_color=75, sigma_space=75):
        """
        Áp dụng bộ lọc Song phương.

        Args:
            d (int): Đường kính của vùng lân cận pixel.
            sigma_color (float): Độ lệch chuẩn trong không gian màu.
            sigma_space (float): Độ lệch chuẩn trong không gian tọa độ.
        """
        self.image = cv2.bilateralFilter(self.image, d, sigma_color, sigma_space)
        print(f"Đã áp dụng Bilateral Filter (d={d}, sigma_color={sigma_color}, sigma_space={sigma_space}).")
        return self

    def non_local_means_denoising(self, h=10, h_color=10, template_window_size=7, search_window_size=21):
        """
        Áp dụng giảm nhiễu Non-Local Means.

        Args:
            h (float): Tham số điều chỉnh cường độ lọc cho kênh độ sáng.
            h_color (float): Tham số điều chỉnh cường độ lọc cho kênh màu (chỉ dùng cho ảnh màu).
            template_window_size (int): Kích thước cửa sổ mẫu (phải là số lẻ).
            search_window_size (int): Kích thước cửa sổ tìm kiếm (phải là số lẻ).
        """
        if self._is_grayscale():
            self.image = cv2.fastNlMeansDenoising(self.image, None, h, template_window_size, search_window_size)
        else:
            self.image = cv2.fastNlMeansDenoisingColored(self.image, None, h, h_color, template_window_size, search_window_size)
        print(f"Đã áp dụng Non-Local Means Denoising.")
        return self

    # --- 3. Chuyển Đổi Không Gian Màu ---
    def convert_color(self, conversion_code):
        """
        Chuyển đổi không gian màu.

        Args:
            conversion_code: Mã chuyển đổi của OpenCV (ví dụ: cv2.COLOR_BGR2HSV).
        """
        try:
            self.image = cv2.cvtColor(self.image, conversion_code)
            print(f"Đã chuyển đổi không gian màu (mã: {conversion_code}).")
        except cv2.error as e:
            print(f"Lỗi chuyển đổi không gian màu: {e}")
        return self

    def to_hsv(self):
        """Chuyển ảnh sang không gian màu HSV."""
        if self._is_grayscale():
            print("Không thể chuyển ảnh xám sang HSV. Ảnh gốc không thay đổi.")
            return self
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        print("Đã chuyển sang không gian màu HSV.")
        return self

    def to_hsl(self):
        """Chuyển ảnh sang không gian màu HSL."""
        if self._is_grayscale():
            print("Không thể chuyển ảnh xám sang HSL. Ảnh gốc không thay đổi.")
            return self
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HLS)
        print("Đã chuyển sang không gian màu HSL.")
        return self

    def to_lab(self):
        """Chuyển ảnh sang không gian màu Lab."""
        if self._is_grayscale():
            print("Không thể chuyển ảnh xám sang Lab. Ảnh gốc không thay đổi.")
            return self
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2Lab)
        print("Đã chuyển sang không gian màu Lab.")
        return self
    
    def to_grayscale(self):
        """Chuyển ảnh sang ảnh xám."""
        if self.image.dtype != np.uint8:
            if self.image.max() <= 1.0:  # Nếu ảnh ở dạng float [0,1]
                self.image = (self.image * 255).astype(np.uint8)
            else:
                self.image = self.image.astype(np.uint8)
        if not self._is_grayscale():
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            print("Đã chuyển sang ảnh xám.")
        else:
            print("Ảnh đã là ảnh xám.")
        return self

    # --- 4. Kỹ Thuật Dựa Trên Tần Số ---
    def homomorphic_filter(self, d0=30, gamma_l=0.5, gamma_h=2.0, c=1):
        """
        Áp dụng lọc Đồng hình (Homomorphic Filter).
        Ảnh sẽ được chuyển sang ảnh xám nếu là ảnh màu.

        Args:
            d0 (float): Tần số cắt (cutoff frequency).
            gamma_l (float): Gamma cho tần số thấp.
            gamma_h (float): Gamma cho tần số cao.
            c (float): Hằng số định hình (sharpness constant).
        """
        if not self._is_grayscale():
            current_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            current_gray = self.image.copy()
            if current_gray.ndim > 2 and current_gray.shape[2] == 1: # (H, W, 1) -> (H, W)
                current_gray = current_gray.squeeze(axis=2)


        # Chuyển ảnh về kiểu float và thêm 1 để tránh log(0)
        img_float = np.float32(current_gray) + 1e-6 # Thêm epsilon nhỏ
        img_log = np.log(img_float)

        # Biến đổi Fourier
        dft = cv2.dft(img_log, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Tạo bộ lọc Homomorphic (Gaussian high-pass based)
        rows, cols = img_log.shape
        crow, ccol = rows // 2, cols // 2

        # Tạo lưới tọa độ cho bộ lọc
        x = np.arange(0, cols, 1)
        y = np.arange(0, rows, 1)
        u, v = np.meshgrid(x, y)
        
        # Tính khoảng cách D(u,v) từ tâm
        distance_squared = (u - ccol)**2 + (v - crow)**2

        # Bộ lọc Homomorphic
        # H(u,v) = (γH - γL) * (1 - exp(-c * D^2(u,v) / D0^2)) + γL
        filter_kernel = (gamma_h - gamma_l) * (1 - np.exp(-c * distance_squared / (d0**2))) + gamma_l

        # Áp dụng bộ lọc (nhân với cả phần thực và ảo)
        # filter_kernel cần được mở rộng để có 2 kênh cho dft_shift
        filter_kernel_complex = np.zeros_like(dft_shift)
        filter_kernel_complex[:,:,0] = filter_kernel
        filter_kernel_complex[:,:,1] = filter_kernel
        
        filtered_dft_shift = dft_shift * filter_kernel_complex

        # Biến đổi Fourier ngược
        dft_ishift = np.fft.ifftshift(filtered_dft_shift)
        img_back = cv2.idft(dft_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT) # DFT_REAL_OUTPUT để lấy phần thực

        # Hàm mũ ngược
        img_exp = np.exp(img_back)

        # Chuẩn hóa về [0, 255]
        img_exp_norm = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX)
        self.image = np.uint8(img_exp_norm)
        
        # Nếu ảnh gốc là màu, chúng ta chỉ xử lý kênh độ sáng.
        # Quyết định ở đây là trả về ảnh xám đã xử lý.
        # Hoặc bạn có thể áp dụng lại vào kênh L của Lab hoặc Y của YCrCb.
        # Hiện tại, nếu ảnh gốc là màu, self.image sẽ trở thành ảnh xám.
        print(f"Đã áp dụng Homomorphic Filter (d0={d0}, γL={gamma_l}, γH={gamma_h}, c={c}).")
        if not self._is_grayscale() and current_gray.shape == self.image.shape:
             print("Lưu ý: Ảnh màu đã được chuyển thành ảnh xám sau Homomorphic Filter.")
        return self

    # --- 5. Các Phép Toán Hình Thái Học ---
    def _get_morph_kernel(self, kernel_size=(5,5), kernel_shape=cv2.MORPH_RECT):
        """Tạo kernel cho các phép toán hình thái học."""
        return cv2.getStructuringElement(kernel_shape, kernel_size)

    def morphological_transform(self, operation, kernel_size=(5,5), kernel_shape=cv2.MORPH_RECT, iterations=1):
        """
        Áp dụng các phép toán hình thái học chung.

        Args:
            operation: Phép toán hình thái của OpenCV (ví dụ: cv2.MORPH_OPEN, cv2.MORPH_CLOSE).
            kernel_size (tuple): Kích thước kernel.
            kernel_shape: Hình dạng kernel (cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS).
            iterations (int): Số lần lặp lại phép toán.
        """
        kernel = self._get_morph_kernel(kernel_size, kernel_shape)
        self.image = cv2.morphologyEx(self.image, operation, kernel, iterations=iterations)
        print(f"Đã áp dụng phép toán hình thái (operation={operation}, kernel_size={kernel_size}, iterations={iterations}).")
        return self

    def top_hat(self, kernel_size=(5,5), kernel_shape=cv2.MORPH_RECT, iterations=1):
        """Áp dụng phép biến đổi Top-hat."""
        return self.morphological_transform(cv2.MORPH_TOPHAT, kernel_size, kernel_shape, iterations)

    def black_hat(self, kernel_size=(5,5), kernel_shape=cv2.MORPH_RECT, iterations=1):
        """Áp dụng phép biến đổi Black-hat."""
        return self.morphological_transform(cv2.MORPH_BLACKHAT, kernel_size, kernel_shape, iterations)

    def erosion(self, kernel_size=(5,5), kernel_shape=cv2.MORPH_RECT, iterations=1):
        """Áp dụng phép co (Erosion)."""
        kernel = self._get_morph_kernel(kernel_size, kernel_shape)
        self.image = cv2.erode(self.image, kernel, iterations=iterations)
        print(f"Đã áp dụng Erosion (kernel_size={kernel_size}, iterations={iterations}).")
        return self

    def dilation(self, kernel_size=(5,5), kernel_shape=cv2.MORPH_RECT, iterations=1):
        """Áp dụng phép giãn (Dilation)."""
        kernel = self._get_morph_kernel(kernel_size, kernel_shape)
        self.image = cv2.dilate(self.image, kernel, iterations=iterations)
        print(f"Đã áp dụng Dilation (kernel_size={kernel_size}, iterations={iterations}).")
        return self

    def opening(self, kernel_size=(5,5), kernel_shape=cv2.MORPH_RECT, iterations=1):
        """Áp dụng phép mở (Opening)."""
        return self.morphological_transform(cv2.MORPH_OPEN, kernel_size, kernel_shape, iterations)

    def closing(self, kernel_size=(5,5), kernel_shape=cv2.MORPH_RECT, iterations=1):
        """Áp dụng phép đóng (Closing)."""
        return self.morphological_transform(cv2.MORPH_CLOSE, kernel_size, kernel_shape, iterations)

    def gradient(self, kernel_size=(5,5), kernel_shape=cv2.MORPH_RECT, iterations=1):

        """Áp dụng phép Morphological Gradient."""
        return self.morphological_transform(cv2.MORPH_GRADIENT, kernel_size, kernel_shape, iterations)




