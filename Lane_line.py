from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_ubyte
# %matplotlib inline

cx = 9
cy = 6
objp = np.zeros((cx * cy, 3), np.float32)
objp[:,:2] = np.mgrid[0:cx, 0:cy].T.reshape(-1, 2)



# Queue for smoothing the curve
class Queue:
    """
    Define a queue class for smoothing the slope and bias of the lane line.
    Smoothing is necessary for this task. 
    """

    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def Empty(self):
        while self.isEmpty() == False:
            self.items.get()
        return self.items

    def put(self, item):
        self.items.insert(0, item)

    def avg(self):
        return np.mean(self.items, axis=0)

    def get(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


images = glob.glob('camera_cal/calibration*.jpg')
count = 1
# fig = plt.figure(figsize=(20, 15))
mtx = []
dist = []
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.


def combined(img):
    result = np.zeros_like(img)
    w = np.int(img.shape[1] / 2)
    h = np.int(img.shape[0] / 2)

    mask, hls_res = hls(img, s_thresh=(170, 255), r_thresh=(220, 255))

    lower_yellow = np.array([10, 100, 90])
    upper_yellow = np.array([22, 220, 255])
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 25, 255])
    mask, hsv_res = hsv(img, lower_yellow, upper_yellow, lower_white, upper_white)

    lower_yellow = np.array([0, 0, 80])
    upper_yellow = np.array([255, 255, 110])
    lower_white = np.array([196, 0, 0])
    upper_white = np.array([255, 255, 255])
    mask, lab_res = lab(img, lower_yellow, upper_yellow, lower_white, upper_white)

    gradx = abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100)
    grady = abs_sobel_thresh(img, orient='y', thresh_min=20, thresh_max=100)

    # Combine image
    grad = np.zeros_like(gradx)
    grad[(hls_res[:, :, 1] > 20) | (hsv_res[:, :, 2] == 255) | (lab_res[:, :, 1] == 255) | (gradx == 1)] = 1
    return grad

def hsv(img, l_yellow, h_yellow, l_white, h_white):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, l_yellow, h_yellow)
    white = cv2.inRange(hsv, l_white, h_white)
    mask = cv2.bitwise_or(yellow, white)
    res = cv2.bitwise_and(img, img, mask=mask)
    return mask, res

def lab(img, l_yellow, h_yellow, l_white, h_white):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    yellow = cv2.inRange(lab, l_yellow, h_yellow)
    white = cv2.inRange(lab, l_white, h_white)
    mask = cv2.bitwise_or(yellow, white)
    res = cv2.bitwise_and(img, img, mask=mask)
    return mask, res

def hls(img, s_thresh, r_thresh):
    # Yellow Line filter
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # White Line filter
    r_channel = img[:, :, 0]
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])] = 1

    mask = np.zeros_like(s_channel)
    mask[(s_binary == 1) | (r_binary == 1)] = 1
    res = cv2.bitwise_and(img, img, mask=mask)
    return mask, res


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output








global left
left = Queue()
global right
right = Queue()

global left_pre_avg
left_pre_avg = 0
global right_pre_avg
right_pre_avg = 0


def calculate_curvature(leftx, rightx, lefty, righty):
    '''Calculate the radius of curvature in meters'''

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    # y_eval = np.max(ploty)
    y_eval = 719
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    #     print('my',right_fit_cr)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters

    return left_curverad, right_curverad


def calculate_offset(undist, left_fit, right_fit):
    '''Calculate the offset of the lane center from the center of the image'''

    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    ploty = undist.shape[0] - 1  # height
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    offset = (left_fitx + right_fitx) / 2 - undist.shape[1] / 2  # width
    offset = xm_per_pix * offset

    return offset

def warp(img):
    img_size = (img.shape[1], img.shape[0])
    leftupperpoint  = [568,470]
    rightupperpoint = [717,470]
    leftlowerpoint  = [260,680]
    rightlowerpoint = [1043,680]

    src = np.float32([leftupperpoint, leftlowerpoint, rightupperpoint, rightlowerpoint])
    dst = np.float32([[200,0], [200,680], [1000,0], [1000,680]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, Minv


def fit_line(combined_binary):
    binary_warped, minv = warp(combined_binary)
    #     plt.imshow(combined_binary)
    #     plt.imshow(binary_warped[int(binary_warped.shape[0]/2):,:])
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
    #     plt.plot(histogram)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    curv_pickle = {}
    curv_pickle["leftx"] = leftx
    curv_pickle["rightx"] = rightx
    curv_pickle["lefty"] = lefty
    curv_pickle["righty"] = righty
    curv_pickle["left_fit"] = left_fit
    curv_pickle["right_fit"] = right_fit

    return left_fitx, right_fitx, ploty, curv_pickle

def process_frame(img):
#    print(img.shape)

#    objpoints = []
#    imgpoints = []
#    images = glob.glob('camera_cal/calibration*.jpg')
#    for img_dir in images:
#        temp_img = cv2.imread(img_dir)
#        gray = cv2.cvtColor(temp_img,cv2.COLOR_BGR2GRAY)
#        ret, corners = cv2.findChessboardCorners(gray, (cx,cy), None)
#        if ret == True:
#            objpoints.append(objp)
#            imgpoints.append(corners)
#    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    pkl_file = open('data.pkl', 'rb')
    image = img
    curv_pickle = pickle.load(pkl_file)
    objpoints = curv_pickle["objpoints"]
    imgpoints = curv_pickle["imgpoints"]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (1280, 720), None, None)
#    print(gray.shape)
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    combined_binary = combined(undist)
    warped, minv = warp(combined_binary)

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    left_fitx, right_fitx, plot_y, c_pickle = fit_line(combined_binary)
    #     c_pickle = line_fit(combined_binary)
    left_fit = c_pickle["left_fit"]
    right_fit = c_pickle["right_fit"]
    leftx = c_pickle["leftx"]
    lefty = c_pickle["lefty"]
    rightx = c_pickle["rightx"]
    righty = c_pickle["righty"]
    # Smooth the curve line
    global left_pre_avg
    global right_pre_avg
    if left_pre_avg == 0:
        left_pre_avg = np.mean(left_fitx)
        right_pre_avg = np.mean(right_fitx)
    if left.size() <= 10 and abs(np.mean(left_fitx) - left_pre_avg) < 10:
        left.put(left_fitx)
        right.put(right_fitx)
    elif abs(np.mean(left_fitx) - left_pre_avg) < 10:
        left.get()
        right.get()
        left.put(left_fitx)
        right.put(right_fitx)

    left_pre_avg = np.mean(left_fitx)
    right_pre_avg = np.mean(right_fitx)

    left_x = left.avg()
    right_x = right.avg()

    pts_left = np.array([np.transpose(np.vstack([left_x, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, plot_y])))])
    pts = np.hstack((pts_left, pts_right))
    #     print(right_fitx)
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, minv, (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Calculate curvature
    # Assume first frame can be detected both lanes and curvatures
    right_curverad = 0
    left_curverad, right_curverad = calculate_curvature(leftx, rightx, lefty, righty)

    vehicle_offset = calculate_offset(undist, left_fit, right_fit)
    #     vehicle_offset = calculate_offset(undist, left_pre_avg, right_pre_avg)
    #     print('vehicle offcet',vehicle_offset)
    # Anotate curvature values
    ave_curvature = (left_curverad + right_curverad) / 2
    ave_text = 'Radius of average curvature: %.2f m' % ave_curvature
    cv2.putText(result, ave_text, (50, 50), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)
    # Anotate vehicle offset from the lane center
    if (vehicle_offset > 0):
        offset_text = 'Vehicle right offset from lane center: {:.2f} m'.format(vehicle_offset)
    else:
        offset_text = 'Vehicle left offset from the lane center: {:.2f} m'.format(-vehicle_offset)
    result_text = cv2.putText(result, offset_text, (50, 80), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)
    #     plt.imshow(result)
    #     result = final_drawing(undist, left_fit, right_fit, left_curverad, right_curverad, minv, vehicle_offset)


    return result_text

# image = plt.imread('frame4.jpg')
# aa = process_frame(image)
# plt.figure()
# plt.imshow(aa)
#
# plt.show()
