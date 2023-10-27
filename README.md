## 비주얼오도메트리와증강현실(AIE6660-01) 중간시험과제 : automatic stitching of two images

## 개요
이 프로젝트는 서강대학교 내부에서 촬영한 두 이미지를 사용하여 파노라마 이미지를 생성하는 과정을 보여줍니다. 컴퓨터 비전 기술과 알고리즘을 사용하여 이미지를 이어 붙여 완벽한 파노라마 뷰를 만듭니다.

## 과정
1. **이미지 선택**: 서강대학교 내부에서 촬영한 두 이미지를 선택합니다.
2. **ORB 특징점 및 기술자 계산**: OpenCV를 사용하여 ORB 특징점 및 기술자를 계산합니다.
3. **해밍 거리를 사용한 브루트포스 매칭**: OpenCV를 사용하여 두 이미지 간의 최상의 매칭을 찾습니다.
4. **RANSAC 알고리즘 구현**: 이미지를 정확하게 정렬하는 데 도움이 되는 호모그래피 매트릭스를 계산하기 위해 RANSAC 알고리즘을 구현합니다.
5. **파노라마 이미지 준비**: 스티치된 파노라마를 수용할 수 있는 큰 캔버스를 준비합니다.
6. **호모그래피 매트릭스를 사용하여 이미지 왜곡**: 계산된 호모그래피 매트릭스를 사용하여 두 이미지를 파노라마 이미지에 왜곡 및 스티칭합니다.

## 함수 설명

- `ransac()`: 두 이미지 간의 호모그래피 행렬을 계산하기 위해 RANSAC 알고리즘을 구현합니다.
- `compute_homography()`: 일치하는 특징점 목록을 사용하여 호모그래피 행렬을 계산합니다.
- `perspective_transform()`: 주어진 호모그래피 행렬을 사용하여 점 집합을 변환합니다.
- `warp_perspective()`: 주어진 호모그래피 행렬을 사용하여 이미지를 새로운 관점으로 왜곡합니다.
- `combine_images()`: 계산된 호모그래피 행렬을 사용하여 두 이미지를 하나의 파노라마로 결합합니다.
- `stitch_panorama()`: 두 이미지를 파노라마로 스티칭하는 메인 함수입니다. 중간 및 최종 결과를 디스크에 저장합니다.

## 결과

**입력 이미지1**
(https://github.com/wjdgus3006/PANORAMA_Image/blob/main/input1_1.jpg)
(https://github.com/wjdgus3006/PANORAMA_Image/blob/main/input1_2.jpg)

**매칭 결과**
(https://github.com/wjdgus3006/PANORAMA_Image/blob/main/matched_features1.jpg)

**파노라마 이미지**
(https://github.com/wjdgus3006/PANORAMA_Image/blob/main/stitched_panorama_1.jpg)
