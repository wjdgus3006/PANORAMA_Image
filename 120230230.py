import cv2
import numpy as np
import random

def ransac(matches, keypoints1, keypoints2, threshold=20, max_iterations=10000):
    best_homography = None
    best_inliers = []

    for i in range(max_iterations):
        sampled_matches = random.sample(matches, 4)
        src_points = np.float32([keypoints1[match.queryIdx].pt for match in sampled_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([keypoints2[match.trainIdx].pt for match in sampled_matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_points, dst_points, method=0)

        inliers = []
        for match in matches:
            pt1 = np.array(keypoints1[match.queryIdx].pt + (1,))
            pt2 = np.array(keypoints2[match.trainIdx].pt + (1,))
            predicted_pt2 = np.dot(H,pt1)
            predicted_pt2 /= predicted_pt2[2]
            
            distance = np.linalg.norm(predicted_pt2 - pt2)
            if distance < threshold:
                inliers.append(match)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_homography = H

    return best_homography, best_inliers

def compute_homography(matches_list, keypoint1, keypoint2):
    if len(matches_list) >= 4:
        homography_matrix, inliers = ransac(matches_list, keypoint1, keypoint2)
        refined_matches = [match for match, valid in zip(matches_list, inliers) if valid]

        return homography_matrix, refined_matches
    else:
        return None, None
    
def perspective_transform(points, H):
    transformed_points = []
    for point in points:
        x, y = point[0][0], point[0][1]
        transformed = np.dot(H,np.array([x, y, 1]))
        transformed /= transformed[2]
        transformed_points.append([[transformed[0], transformed[1]]])
    return np.array(transformed_points)

def warp_perspective(src, H, shape):
    height, width = shape
    dst = np.zeros((height, width, src.shape[2]), dtype=src.dtype)
    
    inv_H = np.linalg.inv(H)
    
    for y in range(height):
        for x in range(width):
            src_coords = np.dot(inv_H,np.array([x, y, 1]))
            src_coords /= src_coords[2]
            src_x, src_y = int(src_coords[0]), int(src_coords[1])
            
            if 0 <= src_x < src.shape[1] and 0 <= src_y < src.shape[0]:
                dst[y, x] = src[src_y, src_x]
                
    return dst

def combine_images(image1, image2, homography):
    h_img1, w_img1 = image1.shape[:2]
    h_img2, w_img2 = image2.shape[:2]

    corner_points_img1 = np.float32([[0, 0], [0, h_img1], [w_img1, h_img1], [w_img1, 0]]).reshape(-1, 1, 2)
    corner_points_img2 = np.float32([[0, 0], [0, h_img2], [w_img2, h_img2], [w_img2, 0]]).reshape(-1, 1, 2)

    transformed_points = perspective_transform(corner_points_img2, homography)

    all_corner_points = np.concatenate((corner_points_img1, transformed_points), axis=0)

    [x_min, y_min] = np.int32(all_corner_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corner_points.max(axis=0).ravel() + 0.5)

    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    stitched_image = warp_perspective(image2, translation_matrix.dot(homography), (x_max-x_min, y_max-y_min))
    stitched_image[-y_min:h_img1-y_min, -x_min:w_img1-x_min] = image1

    return stitched_image

def stitch_panorama():
    
    image1 = cv2.imread('input1_1.jpg')
    image2 = cv2.imread('input1_2.jpg')

    image1 = cv2.resize(image1, (1080, 1080))
    image2 = cv2.resize(image2, (1080, 1080))

    orb_extractor = cv2.ORB_create()
    keypoints_img1, descriptors_img1 = orb_extractor.detectAndCompute(image1, None)
    keypoints_img2, descriptors_img2 = orb_extractor.detectAndCompute(image2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matched_features = matcher.match(descriptors_img1, descriptors_img2)
    matched_features = sorted(matched_features, key=lambda x: x.distance)

    homography, A = compute_homography(matched_features, keypoints_img1, keypoints_img2)
    if homography is None:
        return
    
    matched_image = cv2.drawMatches(image1, keypoints_img1, image2, keypoints_img2, A, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('matched_features_1.jpg', matched_image)

    panorama = combine_images(image2, image1, homography)

    cv2.imwrite('stitched_panorama_1.jpg', panorama)

if __name__ == '__main__':
    stitch_panorama()

