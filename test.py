import numpy as np
from cv2 import cv2

width = 852
height = 480


class out_paras:
    def __init__(self):
        self.radius = []
        self.offset = []
        self.msg = ""
        self.error_msg = ""
        self.pre_radius = None
        self.pre_offset = []


output = out_paras()


# output_data = output

def cal_curvature(left_fit, right_fit, last_pt_center):
    ym_per_pix = 30 / height
    xm_per_pix = 3.7 / width
    y_eval_left = height  # 720p video/image, so last (lowest on screen) y index is 719
    y_eval_right = height

    left_x = left_fit[1]
    left_y = left_fit[0]
    right_x = right_fit[1]
    right_y = right_fit[0]

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(left_y * ym_per_pix, left_x * xm_per_pix, 2)
    right_fit_cr = np.polyfit(right_y * ym_per_pix, right_x * xm_per_pix, 2)

    # Compute R_curve (radius of curvature)
    left_curveR = ((1 + (2 * left_fit_cr[0] * y_eval_left * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curveR = ((1 + (
            2 * right_fit_cr[0] * y_eval_right * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    output.radius = (left_curveR + right_curveR) // 2
    if output.pre_radius is None:
        output.pre_radius = output.radius
    else:
        if (output.radius // output.pre_radius > 3) | (output.pre_radius // output.radius > 3):
            output.error_msg = "Bad calculation"
            output.msg = output.msg + output.error_msg
            output.radius = output.pre_radius
        else:
            output.pre_radius = output.radius
            output.error_msg = "Good calculation"
            output.msg = output.msg + output.error_msg
    output.offset = (last_pt_center - width // 2) * xm_per_pix
    curvature_msg = "Curvature = {:.2f} m".format(output.radius)
    lat_msg = "Car is  {:.2f} m to the center   ".format(output.offset)
    output.msg = lat_msg + curvature_msg
    return output


def process(lane_image, result):
    # Ham xu ly canny
    green = lane_image[:, :, 1]
    is_ret, frame_thresh = cv2.threshold(green, 150, 255, cv2.THRESH_BINARY)
    edge = cv2.Canny(frame_thresh, 250, 255)
    # edge = cv2.dilate(edge, np.ones((3, 3), np.int8))
    cv2.imshow("Default Canny", cv2.resize(edge, (560, 300)))
    h_hist = np.sum(edge, axis=1)
    mid_h_point = np.argmax(h_hist)
    if mid_h_point > height * 5 / 6:
        edge[mid_h_point - 20:mid_h_point + 20, :] = 0
        cv2.imshow("Edit", cv2.resize(edge, (560, 300)))
        mid_h_point = np.argmax(h_hist[:mid_h_point - 50])
    w_hist = np.sum(edge[mid_h_point - 5: mid_h_point + 5, :], axis=0)
    line_horizontal = w_hist.nonzero()
    mid_w_point = np.int(np.average(line_horizontal))
    edge[:mid_h_point + 20, :] = 0
    edge_res = cv2.resize(edge, (560, 300))
    cv2.imshow("Edge", edge_res)
    # Ham xy ly non_zeros
    nonzero_left = edge[mid_h_point + 20:, :mid_w_point].nonzero()
    nonzero_right = edge[mid_h_point + 20:, mid_w_point:].nonzero()
    nonzero_left_x = nonzero_left[1]
    nonzero_left_y = nonzero_left[0] + mid_h_point + 20
    nonzero_right_x = nonzero_right[1] + mid_w_point
    nonzero_right_y = nonzero_right[0] + mid_h_point + 20
    if len(nonzero_right_y) & len(nonzero_right_x) & len(nonzero_left_y) & len(nonzero_left_x):
        center_point = (nonzero_left_x[len(nonzero_left_x) - 1] + nonzero_right_x[len(nonzero_right_x) - 1]) // 2
        left_fit_pixel = np.polyfit(nonzero_left_y, nonzero_left_x, 2)
        right_fit_pixel = np.polyfit(nonzero_right_y, nonzero_right_x, 2)
        # Draw the line
        left_plot_y = np.linspace(nonzero_left_y[0], height - 1, height - nonzero_left_y[0])
        right_plot_y = np.linspace(nonzero_right_y[0], height - 1, height - nonzero_right_y[0])
        if len(left_plot_y) == len(right_plot_y):
            left_fit_x = left_fit_pixel[0] * left_plot_y ** 2 + left_fit_pixel[1] * left_plot_y + left_fit_pixel[2]
            right_fit_x = right_fit_pixel[0] * right_plot_y ** 2 + right_fit_pixel[1] * right_plot_y + right_fit_pixel[
                2]
            if (max(right_fit_x) < width) & (min(right_fit_x) > 0):
                pts_left = np.array([np.transpose(np.vstack([left_fit_x, left_plot_y]))])
                pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, right_plot_y])))])
                center_fit_x = (left_fit_x + right_fit_x) // 2
                output_data = cal_curvature([left_plot_y, left_fit_x], [right_plot_y, right_fit_x],
                                            center_point)

                cv2.putText(result, output_data.msg, org=(10, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 255, 255),
                            thickness=2)
                # pts_center = np.array([np.transpose(np.stack([center_fit_x, left_plot_y]))])
                pts = np.hstack((pts_left, pts_right))
                black = np.zeros_like(edge)
                black[np.int_(left_plot_y), np.int_(left_fit_x)] = 255
                black[np.int_(right_plot_y), np.int_(right_fit_x)] = 255
                black[np.int_(left_plot_y), np.int_(center_fit_x)] = 255
                black = cv2.dilate(black, np.ones((7, 7), np.uint8))
                black = cv2.resize(black, (560, 300))
                cv2.imshow("Poly", black)
            else:
                output_data = None
        else:
            output_data = None
    else:
        cv2.putText(result, "Bad detection", org=(10, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(255, 255, 255),
                    thickness=2)
        output_data = None
    result = cv2.resize(result, (560, 300))
    cv2.imshow("Out", result)
    return result, output_data
