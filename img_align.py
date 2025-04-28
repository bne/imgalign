#!/usr/bin/env python3

import os
import sys

import cv2
import numpy as np

MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.1

def alignImages(img1, img2):
  img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)

  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = list(matcher.match(descriptors1, descriptors2, None))

  matches.sort(key=lambda x: x.distance, reverse=False)

  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", img_matches)

  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  height, width, channels = img2.shape
  img1_reg = cv2.warpPerspective(img1, h, (width, height))

  return img1_reg, h

if __name__ == '__main__':

  if len(sys.argv) != 3:
    print("Usage: python img_align.py <input_dir> <output_dir>")
    sys.exit(1)

  input_dir = sys.argv[1]
  output_dir = sys.argv[2]

  input_files = os.listdir(input_dir)
  reference_img = cv2.imread(os.path.join(input_dir, input_files[0]), cv2.IMREAD_COLOR)

  for filename in input_files[1:]:
    input_img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_COLOR)
    aligned_img, h = alignImages(input_img, reference_img)

    cv2.imwrite(os.path.join(output_dir, filename), aligned_img)

    print(f"Aligned {filename} to reference image with homography:\n{h}")

  # save the reference image
  cv2.imwrite(os.path.join(output_dir, input_files[0]), reference_img)

