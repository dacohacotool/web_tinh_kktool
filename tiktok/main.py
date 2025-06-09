import cv2
import numpy as np

def ChuyenSangTrangDen (img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tính độ thay đổi (gradient) bằng Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(sobelx, sobely)

    # Chuẩn hóa và ngưỡng
    normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)

    # Chỉ giữ lại vùng có độ tương phản cao
    _, binary = cv2.threshold(normalized, 50, 255, cv2.THRESH_BINARY)  # Tham số 50 có thể chỉnh

    # Đảo màu nếu muốn: những vùng "mạnh" là đen, còn lại là trắng
    binary = cv2.bitwise_not(binary)

    return binary
def paint_at_coordinate(img, x, y, loDiff=(20,20,20), upDiff=(20,20,20)):
    h, w = img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("Tọa độ ngoài ảnh")
    mask = np.zeros((h+2, w+2), np.uint8)
    paint_color = (0,0,0)
    cv2.floodFill(img, mask, seedPoint=(x,y), newVal=paint_color, loDiff=loDiff, upDiff=upDiff)
    return img

def expand_black_and_sharpen(img, dilation_size=3):
    """
    img: ảnh grayscale (0-255), vùng đen là 0, trắng là 255
    dilation_size: kích thước kernel giãn nở
    """

    # Nếu ảnh màu, chuyển sang grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Tạo kernel hình chữ nhật
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_size, dilation_size))

    # Dilation - giãn nở vùng đen (chúng ta invert ảnh để dùng dilation cho vùng đen)
    # Bởi vì dilation trên vùng trắng sẽ mở rộng trắng, muốn mở rộng đen ta invert ảnh
    inverted = 255 - gray

    dilated = cv2.dilate(inverted, kernel, iterations=1)

    # Invert lại để trở về ảnh gốc với vùng đen được giãn nở
    dilated_inverted = 255 - dilated

    # Tạo mask vùng thay đổi (vùng đen lan rộng ra)
    mask = dilated_inverted < gray  # pixel bị tối hơn sau dilation

    # Phục hồi vùng không bị thay đổi để giữ rõ nét
    result = gray.copy()
    result[mask] = dilated_inverted[mask]

    return result
def replace_piece_with_square(img, threshold=200):
    """
    img: ảnh grayscale hoặc ảnh màu
    threshold: ngưỡng tách vùng trắng (0-255)
    Trả về: ảnh đã xử lý, tọa độ x trung tâm của hình vuông
    """

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, None  # không tìm thấy mảnh ghép

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    side = max(w, h)

    output = img.copy()
    if img.ndim == 3:
        white_square = np.ones((side, side, 3), dtype=img.dtype) * 255
    else:
        white_square = np.ones((side, side), dtype=img.dtype) * 255

    end_x = min(x + side, img.shape[1])
    end_y = min(y + side, img.shape[0])
    output[y:end_y, x:end_x] = white_square[:end_y - y, :end_x - x]

    center_x = x + (end_x - x) // 2
    center_y = y + (end_y - y) // 2

    return output, center_x, center_y
def BoiDenCanhTrai (img):
    height, width = img.shape[:2]

    # Tính 1/10 chiều ngang
    cut_x = width // 5
    cut_y = height // 4

    img[0:cut_y, :] = 0           # Cạnh trên
    img[:, 0:cut_x] = 0          # Cạnh trái
    # img[:, -cut_x:] = 0          # Cạnh phải
    return img
def XuLyAnhCaptchaPuzzle(input_path, output_path):
    img = cv2.imread(input_path)
    height, width = img.shape[:2]
    img = ChuyenSangTrangDen (img)
    
    img = paint_at_coordinate(img, 1, 2)
    img = paint_at_coordinate(img, 20, 5)
    img = paint_at_coordinate(img, int(width/10), int(height - height/5))
    img = BoiDenCanhTrai (img)
    img = expand_black_and_sharpen (img,3)
    img, centerx,centery =replace_piece_with_square (img, 200)

    img = cv2.imread(input_path)
    cv2.circle(img, (centerx, centery), radius=10, color=(0, 0, 255), thickness=-1)
    cv2.imwrite(output_path, img)
    return centerx

def decode_base64_image(base64_str):
        img_data = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def encode_image_to_base64(img):
        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode('utf-8')

def XuLyAnhCaptchaPuzzle_base64(input_base64):
    img = decode_base64_image(input_base64)
    height, width = img.shape[:2]

    img = ChuyenSangTrangDen(img)
    img = paint_at_coordinate(img, 1, 2)
    img = paint_at_coordinate(img, 20, 5)
    img = paint_at_coordinate(img, int(width / 10), int(height - height / 5))
    img = BoiDenCanhTrai(img)
    img = expand_black_and_sharpen(img, 3)
    img, centerx, centery = replace_piece_with_square(img, 200)

    # Vẽ lên ảnh gốc (decode lại từ base64 để giữ nguyên)
    img_origin = decode_base64_image(input_base64)
    cv2.circle(img_origin, (centerx, centery), radius=10, color=(0, 0, 255), thickness=-1)

    output_base64 = encode_image_to_base64(img_origin)
    return centerx, output_base64
# inpath = "capture_RF8M50WTVVR.png"
# output_file = 'filled_contrast_binary.jpg'

# XuLyAnhCaptchaPuzzle(inpath,output_file)
