"""
Python + OpenCV Implementation of the vanishing point algorithm by Xiaohu Lu
et al. -
http://xiaohulugo.github.io/papers/Vanishing_Point_Detection_WACV2017.pdf.

Author: Ray Phan (https://github.com/rayryeng)
"""


from cProfile import label
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, MiniBatchKMeans, MeanShift, OPTICS, SpectralClustering
from sklearn.mixture import GaussianMixture
from itertools import combinations
from utils import pred_lines


class VPDetection(object):
    """
    VP Detection Object

    Args:
        length_thresh: Line segment detector threshold (default=30)
        principal_point: Principal point of the image (in pixels)
        focal_length: Focal length of the camera (in pixels)
        seed: Seed for reproducibility due to RANSAC
    """

    def __init__(self,
                length_thresh=30,
                principal_point=None,
                focal_length=1500,
                seed=None):
        self._length_thresh = length_thresh
        self._principal_point = principal_point
        self._focal_length = focal_length
        self._angle_thresh = np.pi / 5  # For displaying debug image
        self._vps = None  # For storing the VPs in 3D space
        self._vps_2D = None  # For storing the VPs in 2D space
        self._vps_gt = None
        self._vps_2D_gt = None
        self.__img = None  # Stores the image locally
        self.__clusters = None  # Stores which line index maps to what VP
        self.__tol = 1e-8  # Tolerance for floating point comparison
        self.__angle_tol = np.pi / 3  # (pi / 180 * (60 degrees)) = +/- 30 deg
        self.__lines = None  # Stores the line detections internally
        self.__zero_value = 0.001  # Threshold to check augmented coordinate
        # Anything less than __tol gets set to this
        self.__seed = seed  # Set seed for reproducibility
        noise_ratio = 0.5  # Outlier/inlier ratio for RANSAC estimation
        # Probability of all samples being inliers
        p = (1.0 / 3.0) * ((1.0 - noise_ratio)**2.0)  # 外れ値の割合

        # Total number of iterations for RANSAC(少なくとも一回は外れ値が含まれない確率を0.9999以上という条件)
        conf = 0.9999
        self.__ransac_iter = int(np.log(1 - conf) / np.log(1.0 - p))
        self.__idx_deg = None

    # property:外から簡単に値を得られない上に取り出しは簡単にしたい
    # @property:プロパティの値を取り出すメソッドを定義
    # @property_name.setter:プロパティの値を設定
    # 値を呼び出すときは @property
    # 値を変更するときは @property_name.setter が呼び出される
    # 予期せぬ値が入り込まないようにsetterで値を制限することができる

    @property  # 引数がないとき
    def length_thresh(self):
        """
        Length threshold for line segment detector

        Returns:
            The minimum length required for a line
        """
        return self._length_thresh

    @length_thresh.setter  # 引き奇数があるとき(条件を満たさなければならない)
    def length_thresh(self, value):
        """
        Length threshold for line segment detector

        Args:
            value: The minimum length required for a line #線の最小値

        Raises:
            ValueError: If the threshold is 0 or negative
        """
        if value <= 0:
            raise ValueError('Invalid threshold: {}'.format(value))

        self._length_thresh = value

    @property
    def principal_point(self):
        """
        Principal point for VP Detection algorithm

        Returns:
            The minimum length required for a line
        """
        return self._principal_point

    @principal_point.setter
    def principal_point(self, value):
        """
        Principal point for VP Detection algorithm

        Args:
            value: A list or tuple of two elements denoting the x and y
           coordinates

        Raises:
            ValueError: If the input is not a list or tuple and there aren't
            two elements
        """
        try:
            assert isinstance(value,
                              (list, tuple)) and not isinstance(value, str)
            assert len(value) == 2
        except AssertionError:
            raise ValueError('Invalid principal point: {}'.format(value))

        self._length_thresh = value

    @property
    def focal_length(self):
        """
        Focal length for VP detection algorithm # 焦点距離

        Returns:
            The focal length in pixels
        """
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):
        """
        Focal length for VP detection algorithm

        Args:
            value: The focal length in pixels

        Raises:
            ValueError: If the input is 0 or negative
        """
        if value < self.__tol:  # If the focal length is too small, reject
            raise ValueError('Invalid focal length: {}'.format(value))

        self._focal_length = value

    @property
    def vps(self):
        """
        Vanishing points of the image in 3D space.

        Returns:
            A numpy array where each row is a point and each column is a
            component / coordinate # 行が点、列が座標
        """
        return self._vps

    @property
    def vps_gt(self):
        """
        Vanishing points of the image in 3D space.

        Returns:
            A numpy array where each row is a point and each column is a
            component / coordinate # 行が点、列が座標
        """
        return self._vps_gt

    @property
    def vps_2D(self):
        """
        Vanishing points of the image in 2D image coordinates.

        Returns:
            A numpy array where each row is a point and each column is a
            component / coordinate
        """
        return self._vps_2D

    @property
    def vps_2D_gt(self):
        """
        Vanishing points of the image in 2D image coordinates.

        Returns:
            A numpy array where each row is a point and each column is a
            component / coordinate
        """
        return self._vps_2D_gt

    def __detect_lines(self, img, interpreter, input_details, output_details, input_shape, score_thr, dist_thr):
        """
        Detects lines using OpenCV LSD Detector
        """
        # Convert to grayscale if required
        if len(img.shape) == 3:
            img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_copy = img

        # Create LSD detector with default parameters
        #lsd = cv2.createLineSegmentDetector(0)

        # Detect lines in the image
        # Returns a NumPy array of type N x 1 x 4 of float32
        # such that the 4 numbers in the last dimension are (x1, y1, x2, y2) (直線の最初の点, 直線の最後の点)
        # These denote the start and end positions of a line
        #lines = lsd.detect(img_copy)[0]  # ライン取得

        lines = pred_lines(img, interpreter, input_details, output_details, input_shape, score_thr, dist_thr)

        # Filter out the lines whose length is lower than the threshold
        dx = lines[:, 2] - lines[:, 0]  # x2 - x1
        dy = lines[:, 3] - lines[:, 1]  # y2 - y1
        lengths = np.sqrt(dx * dx + dy * dy)  # 2点間の距離の長さ
        mask = lengths >= self._length_thresh
        lines = lines[mask]

        # Store the lines internally
        self.__lines = lines

        # Return the lines
        return lines

    def __find_vp_hypotheses_two_lines(self):
        """
        Finds the VP hypotheses using pairs of lines
        """
        # Number of detected lines
        N = self.__lines.shape[0]

        # Number of bins for longitude - 360 bins so 1 deg. per bin
        # For estimating second VP along the great circle distance of the
        # first VP
        num_bins_vp2 = 360
        vp2_step = np.pi / 180.0  # Step in radians

        # Store the equations of the line, lengths and orientations
        # for each line segment
        p1 = np.column_stack(
            (self.__lines[:, :2], np.ones(N, dtype=np.float32)))
        p2 = np.column_stack(
            (self.__lines[:, 2:], np.ones(N, dtype=np.float32)))
        cross_p = np.cross(p1, p2)
        dx = p1[:, 0] - p2[:, 0]
        dy = p1[:, 1] - p2[:, 1]
        lengths = np.sqrt(dx * dx + dy * dy)
        orientations = np.arctan2(dy, dx)

        # Perform wraparound - [-pi, pi] --> [0, pi]
        # All negative angles map to their mirrored positive counterpart
        orientations[orientations < 0] = orientations[orientations < 0] + np.pi

        # Keep these around
        self.__cross_p = cross_p
        self.__lengths = lengths
        self.__orientations = orientations

        # Stores the VP hypotheses - 3 per longitude for each RANSAC iteration
        # First dimension - VP triplet proposal for a RANSAC iteration
        # Second dimension - VPs themselves
        # Third dimension - VP component
        vp_hypos = np.zeros(
            (self.__ransac_iter * num_bins_vp2, 3, 3), dtype=np.float32)

        i = 0

        if self.__seed is not None:
            gen = np.random.RandomState(self.__seed)

        # For each iteration...
        while i < self.__ransac_iter:
            # Get two random indices
            if self.__seed is not None:
                (idx1, idx2) = gen.permutation(N)[:2]
            else:
                (idx1, idx2) = np.random.permutation(N)[:2]

            # Get the first VP proposal in the image
            vp1_img = np.cross(cross_p[idx1], cross_p[idx2])

            # Try again if at infinity
            if np.abs(vp1_img[2]) < self.__tol:
                continue

            # Find where it intersects in the sphere
            vp1 = np.zeros(3, dtype=np.float32)
            vp1[:2] = vp1_img[:2] / vp1_img[2] - self._principal_point
            vp1[2] = self._focal_length

            # Normalize
            vp1 /= np.sqrt(np.sum(np.square(vp1)))

            # Get the other two VPs
            # Search along the circumference of the sphere
            la = np.arange(num_bins_vp2) * vp2_step
            kk = vp1[0] * np.sin(la) + vp1[1] * np.cos(la)
            phi = np.arctan(-vp1[2] / kk)

            # Convert back to Cartesian coordinates
            vp2 = np.column_stack([
                np.sin(phi) * np.sin(la),
                np.sin(phi) * np.cos(la),
                np.cos(phi)
            ])

            # Enforce points at infinity to be finite
            vp2[np.abs(vp2[:, 2]) < self.__tol, 2] = self.__zero_value
            # Normalize
            vp2 /= np.sqrt(np.sum(np.square(vp2), axis=1, keepdims=True))
            vp2[vp2[:, 2] < 0, :] *= -1.0  # Ensure direction is +z

            vp3 = np.cross(vp1, vp2)  # Third VP is orthogonal to the two
            vp3[np.abs(vp3[:, 2]) < self.__tol, 2] = self.__zero_value
            vp3 /= np.sqrt(np.sum(np.square(vp3), axis=1, keepdims=True))
            vp3[vp3[:, 2] < 0, :] *= -1.0

            # Place proposals in corresponding locations
            vp_hypos[i * num_bins_vp2:(i + 1) * num_bins_vp2, 0, :] = vp1
            vp_hypos[i * num_bins_vp2:(i + 1) * num_bins_vp2, 1, :] = vp2
            vp_hypos[i * num_bins_vp2:(i + 1) * num_bins_vp2, 2, :] = vp3

            # Move to the next iteration
            i += 1

        return vp_hypos

    def __get_sphere_grids(self):
        """
        Builds spherical voting grid to determine which VP has the most support
        """

        # Determine number of bins for latitude and longitude
        bin_size = np.pi / 180.0
        lat_span = np.pi / 2.0
        long_span = 2.0 * np.pi
        num_bins_lat = int(lat_span / bin_size)
        num_bins_lon = int(long_span / bin_size)

        # Get indices for every unique pair of lines
        combos = list(combinations(range(self.__lines.shape[0]), 2))
        combos = np.asarray(combos, dtype=np.int)

        # For each pair, determine where the lines intersect
        pt_intersect = np.cross(self.__cross_p[combos[:, 0]],
                                self.__cross_p[combos[:, 1]])

        # Ignore if points are at infinity
        mask = np.abs(pt_intersect[:, 2]) >= self.__tol

        # To determine if two points map to the same VP in spherical
        # coordinates, their difference in angle must be less than
        # some threshold
        ang = np.abs(self.__orientations[combos[:, 0]] -
                     self.__orientations[combos[:, 1]])
        ang = np.minimum(np.pi - ang, ang)
        mask = np.logical_and(mask, np.abs(ang) <= self.__angle_tol)

        # Get the points, angles and combinations that are
        # left
        pt_intersect = pt_intersect[mask]
        ang = ang[mask]
        combos = combos[mask]

        # Determine corresponding lat and lon mapped to the sphere
        X = (pt_intersect[:, 0] /
             pt_intersect[:, 2]) - self._principal_point[0]
        Y = (pt_intersect[:, 1] /
             pt_intersect[:, 2]) - self._principal_point[1]
        Z = self._focal_length
        lat = np.arccos(Z / np.sqrt(X * X + Y * Y + Z * Z))
        lon = np.arctan2(X, Y) + np.pi

        # Get corresponding bin locations
        la_bin = (lat / bin_size).astype(np.int)
        lon_bin = (lon / bin_size).astype(np.int)
        la_bin[la_bin >= num_bins_lat] = num_bins_lat - 1
        lon_bin[lon_bin >= num_bins_lon] = num_bins_lon - 1

        # Add their weighted vote to the corresponding bin
        # Get 1D bin coordinate so we can take advantage
        # of bincount method, then reshape back to 2D
        bin_num = la_bin * num_bins_lon + lon_bin
        weights = np.sqrt(
            self.__lengths[combos[:, 0]] *
            self.__lengths[combos[:, 1]]) * (np.sin(2.0 * ang) + 0.2)

        sphere_grid = np.bincount(
            bin_num, weights=weights,
            minlength=num_bins_lat * num_bins_lon).reshape(
                (num_bins_lat, num_bins_lon)).astype(np.float32)

        # Add the 3 x 3 smoothed votes on top of the original votes for
        # stability (refer to paper)
        sphere_grid += cv2.filter2D(sphere_grid, -1, (1.0 / 9.0) * np.ones(
            (3, 3)))
        return sphere_grid

    def __get_best_vps_hypo(self, sphere_grid, vp_hypos, gt_vp):
        # Number of hypotheses(3x3)
        N = vp_hypos.shape[0]

        # Bin size - 1 deg. in radians
        bin_size = np.pi / 180.0

        #print("vp_hypos:", vp_hypos)
        #print("vp_hypos:", vp_hypos[:,:,2])

        # Ignore any values whose augmented coordinate are less than
        # the threshold or bigger than magnitude of 1
        # Each row is a VP triplet(3つで1つの行)
        # Each column is the z coordinate(それぞれz座標)
        # 引数より小さく，1より大きいものを無視
        # 3xN
        mask = np.logical_and(
            np.abs(vp_hypos[:, :, 2]) >= self.__tol,
            np.abs(vp_hypos[:, :, 2]) <= 1.0)

        #np.set_printoptions(threshold=np.inf)

        # Create ID array for VPs
        ids = np.arange(N).astype(np.int) #0~Nの数をリストに入れる
        ids = np.column_stack([ids, ids, ids]) #リストに入れた数字を3(同じ数字)ｘN個にスタックする
        ids = ids[mask] #1次元配列にする

        # Calculate their respective lat and lon
        lat = np.arccos(vp_hypos[:, :, 2][mask]) #z軸の逆余弦を求める(ラジアン)
        lon = np.arctan2(vp_hypos[:, :, 0][mask],
                        vp_hypos[:, :, 1][mask]) + np.pi #xとyの間の角度(ラジアン)

        # Determine which bin they map to
        la_bin = (lat / bin_size).astype(np.int) #ラジアンから角度に変換
        lon_bin = (lon / bin_size).astype(np.int) #ラジアンから角度に変換

        la_bin[la_bin == 90] = 89 #0~89
        lon_bin[lon_bin == 360] = 359 #0~359

        # For each hypotheses triplet of VPs, calculate their final
        # votes by summing the contributions of each VP for the
        # hypothesis
        # sphere_grid=(90, 360)
        weights = sphere_grid[la_bin, lon_bin]
        votes = np.bincount(ids, weights=weights,
                            minlength=N).astype(np.float32)

        # Find best hypothesis by determining which triplet has the largest
        # votes
        best_idx = np.argmax(votes)
        final_vps = vp_hypos[best_idx]
        #print(final_vps)
        #print(gt_vp)
        vps_2D = self._focal_length * (final_vps[:, :2] / final_vps[:, 2][:, None])
        vps_2D_gt = self._focal_length * (gt_vp[:, :2] / gt_vp[:, 2][:, None])
        vps_2D += self._principal_point
        vps_2D_gt += self._principal_point


        # Find the coordinate with the largest vertical value
        # This will be the last column of the output
        z_idx = np.argmax(np.abs(vps_2D[:, 1]))
        ind = np.arange(3).astype(np.int)
        mask = np.ones(3, dtype=np.bool)
        mask[z_idx] = False
        ind = ind[mask]

        z_idx_gt = np.argmax(np.abs(vps_2D_gt[:, 1]))
        ind_gt = np.arange(3).astype(np.int)
        mask_gt = np.ones(3, dtype=np.bool)
        mask_gt[z_idx_gt] = False
        ind_gt = ind_gt[mask_gt]

        # Next, figure out which of the other two coordinates has the smallest
        # x coordinate - this would be the left leaning VP
        vps_trim = vps_2D[mask]
        x_idx = np.argmin(vps_trim[:, 0])
        x_idx = ind[x_idx]

        vps_trim_gt = vps_2D_gt[mask_gt]
        x_idx_gt = np.argmin(vps_trim_gt[:, 0])
        x_idx_gt = ind_gt[x_idx_gt]

        # Finally get the right learning VP
        mask[x_idx] = False
        x2_idx = np.argmax(mask)

        mask_gt[x_idx_gt] = False
        x2_idx_gt = np.argmax(mask_gt)

        # Re-arrange the order
        # Right VP is first - x-axis would be to the right
        # Left VP is second - y-axis would be to the left
        # Vertical VP is third - z-axis would be vertical
        final_vps = final_vps[[x2_idx, x_idx, z_idx], :]
        gt_vp = gt_vp[[x2_idx_gt, x_idx_gt, z_idx_gt], :]
        #print(gt_vp)
        vps_2D = vps_2D[[x2_idx, x_idx, z_idx], :]
        vps_2D_gt = vps_2D_gt[[x2_idx_gt, x_idx_gt, z_idx_gt], :]


        # Save for later
        self._vps = final_vps
        self._vps_2D = vps_2D

        self._vps_gt = gt_vp
        self._vps_2D_gt = vps_2D_gt
        return final_vps

    def __cluster_lines(self, vps_hypos):
        """
        Groups the lines based on which VP they contributed to.
        Primarily for display purposes only when debugging the algorithm
        """

        # Extract out line coordinates
        x1 = self.__lines[:, 0]
        y1 = self.__lines[:, 1]
        x2 = self.__lines[:, 2]
        y2 = self.__lines[:, 3]

        # Get midpoint of each line
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0

        X = [xc, yc]
        Xnp = np.array(X).T

        self.clustering_AffinityPropagation(Xnp, x1, x2, y1, y2)
        self.clustering_AgglomerativeClustering(Xnp, x1, x2, y1, y2)
        self.clustering_BIRCH(Xnp, x1, x2, y1, y2)
        self.clustering_DBSCAN(Xnp, x1, x2, y1, y2)
        self.clustering_MiniBatchKMeans(Xnp, x1, x2, y1, y2)
        self.clustering_MeanShift(Xnp, x1, x2, y1, y2)
        self.clustering_OPTICS(Xnp, x1, x2, y1, y2)
        self.clustering_SpectralClustering(Xnp, x1, x2, y1, y2)
        self.clustering_GaussianMixture(Xnp, x1, x2, y1, y2)
        #plt.show()

        # Get the direction vector of the line detection
        # Also normalize
        dx = x1 - x2
        dy = y1 - y2
        norm_factor = np.sqrt(dx * dx + dy * dy)
        dx /= norm_factor # cos
        dy /= norm_factor # sin

        self.calculate_rad(dx, dy, xc, yc)

        # Get the direction vector from each detected VP
        # to the midpoint of the line and normalize
        xp = self._vps_2D[:, 0][:, None] - xc[None]
        yp = self._vps_2D[:, 1][:, None] - yc[None]
        norm_factor = np.sqrt(xp * xp + yp * yp)
        xp /= norm_factor
        yp /= norm_factor

        # Calculate the dot product then find the angle between the midpoint
        # of each line and each VPs
        # We calculate the angle that each make with respect to each line and
        # and choose the VP that has the smallest angle with the line
        dotp = dx[None] * xp + dy[None] * yp
        dotp[dotp > 1.0] = 1.0
        dotp[dotp < -1.0] = -1.0
        ang = np.arccos(dotp)
        ang = np.minimum(np.pi - ang, ang)

        # For each line, which VP is the closest?
        # Get both the smallest angle and index of the smallest
        min_ang = np.min(ang, axis=0) # 角度の最小値を3つの中から選びmin_angに格納
        idx_ang = np.argmin(ang, axis=0) # 角度の最小値がどのvpに属するかのindexを取得

        # Don't consider any lines where the smallest angle is larger than
        # a similarity threshold
        mask = min_ang <= self._angle_thresh

        # For each VP, figure out the line indices
        # Create a list of 3 elements
        # Each element contains which line index corresponds to which VP
        self.__clusters = [
            np.where(np.logical_and(mask, idx_ang == i))[0] for i in range(3)
        ]

    def find_vps(self, img, interpreter, input_details, output_details, input_shape, score_thr, dist_thr, gt_vp):
        """
        Find the vanishing points given the input image

        Args:
            img: Either the path to the image or the image read in with
        `cv2.imread`

        Returns:
            A numpy array where each row is a point and each column is a
            component / coordinate. Additionally, the VPs are ordered such that
            the right most VP is the first row, the left most VP is the second
            row and the vertical VP is the last row
        """

        # Detect the lines in the image
        if isinstance(img, str):
            img = cv2.imread(img, -1)

        self.__img = img  # Keep a copy for later

        # Reset principal point if we haven't set it yet
        if self._principal_point is None:
            rows, cols, _ = img.shape
            #rows, cols = img.shape[:2]
            self._principal_point = np.array([cols / 2.0, rows / 2.0],
                                            dtype=np.float32)

        # Detect lines
        _ = self.__detect_lines(img, interpreter, input_details, output_details, input_shape, score_thr, dist_thr)

        # Find VP candidates
        vps_hypos = self.__find_vp_hypotheses_two_lines()
        self.__vps_hypos = vps_hypos  # Save a copy

        # Map VP candidates to sphere
        sphere_grid = self.__get_sphere_grids()

        # Find the final VPs
        best_vps = self.__get_best_vps_hypo(sphere_grid, vps_hypos, gt_vp)
        self.__final_vps = best_vps  # Save a copy
        self.__clusters = None  # Reset because of new image
        return best_vps

    def create_debug_VP_image(self,save_image=None):
        """
        Once the VP detection algorithm runs, show which lines belong to
        which clusters by colouring the lines according to which VP they
        contributed to

        Args:
            show_image: Show the image in an OpenCV imshow window
                        (default=false)
            save_image: Provide a path to save the image to file
                        (default=None - no image is saved)

        Returns:
            The debug image

        Raises:
            ValueError: If the path to the image is not a string or None
        """

        # Group the line detections based on which VP they belong to
        # Only run if we haven't done it yet for this image
        if self.__clusters is None:
            self.__cluster_lines(self.__vps_hypos)

        if save_image is not None and not isinstance(save_image, str):
            raise ValueError('The save_image path should be a string')

        img = self.__img.copy()
        img_2 = self.__img.copy()

        if len(img.shape) == 2:  # If grayscale, artificially make into RGB
            img = np.dstack([img, img, img])

        colours = 255 * np.eye(3)  # 単位行列の作成 * 255
        # BGR format
        # First row is red, second green, third blue
        colours = colours[:, ::-1].astype(np.int).tolist()

        # Draw the outlier lines as black
        all_clusters = np.hstack(self.__clusters)
        status = np.ones(self.__lines.shape[0], dtype=np.bool)
        status[all_clusters] = False
        ind = np.where(status)[0]

        # # 白色の画像を準備
        # # h, w = img.shape[0], img.shape[1]
        # # img_edge = np.ones((h, w, 3), np.uint8)*255

        # 線を引く(線を引く画像, 座標, 線の色, 線の太さ, 線のタイプ)
        #(x1, y1)から(x2, y2)に向かって(0, 0, 0)の黒色、太さは2、線の種類はアンチエイリアス
        #print("(x1, y1) -> (x2, y2)")
        for (x1, y1, x2, y2) in self.__lines[ind]:
            #print("{0} -> {1}".format((x1, y1), (x2, y2)))
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2, cv2.LINE_AA)
            #cv2.line(img_edge, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 3, cv2.LINE_8)  # draw black-line at white image
            # cv2.line(img_vp, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3, cv2.LINE_8)  # draw white-line at edge image with mask

        # For each cluster of lines, draw them in their right colour
        for i in range(3):
            for (x1, y1, x2, y2) in self.__lines[self.__clusters[i]]:
                #print("{0} -> {1}".format((x1, y1), (x2, y2)))
                # colours[i]:3色 -> (0, 0, 0):黒色
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), colours[i], 2, cv2.LINE_AA)
                #cv2.line(img_edge, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 3, cv2.LINE_8)  # draw black-line at white image
                #cv2.line(img_vp, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3, cv2.LINE_8)


        for (x1, y1, x2, y2), i in zip(self.__lines, self.__idx_deg):
            cv2.line(img_2, (int(x1), int(y1)), (int(x2), int(y2)), colours[i.astype(int)], 2, cv2.LINE_AA)
        cv2.imshow("img", img_2)
        cv2.waitKey()

        return img

    def clustering_AffinityPropagation(self, X, x1, x2, y1, y2):
        plt.subplot(331).invert_yaxis()
        model = AffinityPropagation(damping=0.9)
        model.fit(X)
        yhat = model.predict(X)
        clusters = np.unique(yhat)
        for cluster in clusters:
            row_ix = np.where(yhat == cluster)
            plt.scatter(X[row_ix, 0], X[row_ix, 1])

        plt.title("Affinity Propagation")
        plt.plot([x1, x2], [y1, y2], color="gray")
        plt.grid(True)

    def clustering_AgglomerativeClustering(self, X, x1, x2, y1, y2):
        plt.subplot(332).invert_yaxis()
        model = AgglomerativeClustering(n_clusters=3)
        yhat = model.fit_predict(X)
        clusters = np.unique(yhat)
        for cluster in clusters:
            row_ix = np.where(yhat == cluster)
            plt.scatter(X[row_ix, 0], X[row_ix, 1])

        plt.title("Agglomerative Clustering")
        plt.plot([x1, x2], [y1, y2], color="gray")
        plt.grid(True)

    def clustering_BIRCH(self, X, x1, x2, y1, y2):
        plt.subplot(333).invert_yaxis()
        model = Birch(threshold=0.01, n_clusters=3)
        model.fit(X)
        yhat = model.predict(X)
        clusters = np.unique(yhat)
        for cluster in clusters:
            row_ix = np.where(yhat == cluster)
            plt.scatter(X[row_ix, 0], X[row_ix, 1])

        plt.title("BIRCH")
        plt.plot([x1, x2], [y1, y2], color="gray")
        plt.grid(True)

    def clustering_DBSCAN(self, X, x1, x2, y1, y2):
        plt.subplot(334).invert_yaxis()
        model = DBSCAN(eps=0.30, min_samples=9)
        yhat = model.fit_predict(X)
        clusters = np.unique(yhat)
        for cluster in clusters:
            row_ix = np.where(yhat == cluster)
            plt.scatter(X[row_ix, 0], X[row_ix, 1])

        plt.title("DBSCAN")
        plt.plot([x1, x2], [y1, y2], color="gray")
        plt.grid(True)

    def clustering_MiniBatchKMeans(self, X, x1, x2, y1, y2):
        plt.subplot(335).invert_yaxis()
        model = MiniBatchKMeans(n_clusters=3)
        model.fit(X)
        yhat = model.predict(X)
        clusters = np.unique(yhat)
        for cluster in clusters:
            row_ix = np.where(yhat == cluster)
            plt.scatter(X[row_ix, 0], X[row_ix, 1])

        plt.title("MiniBatchKMeans")
        plt.plot([x1, x2], [y1, y2], color="gray")
        plt.grid(True)

    def clustering_MeanShift(self, X, x1, x2, y1, y2):
        plt.subplot(336).invert_yaxis()
        model = MeanShift()
        yhat = model.fit_predict(X)
        clusters = np.unique(yhat)
        for cluster in clusters:
            row_ix = np.where(yhat == cluster)
            plt.scatter(X[row_ix, 0], X[row_ix, 1])

        plt.title("MeanShift")
        plt.plot([x1, x2], [y1, y2], color="gray")
        plt.grid(True)

    def clustering_OPTICS(self, X, x1, x2, y1, y2):
        plt.subplot(337).invert_yaxis()
        model = OPTICS(eps=0.8, min_samples=10)
        yhat = model.fit_predict(X)
        clusters = np.unique(yhat)
        for cluster in clusters:
            row_ix = np.where(yhat == cluster)
            plt.scatter(X[row_ix, 0], X[row_ix, 1])

        plt.title("OPTICS")
        plt.plot([x1, x2], [y1, y2], color="gray")
        plt.grid(True)

    def clustering_SpectralClustering(self, X, x1, x2, y1, y2):
        plt.subplot(338).invert_yaxis()
        model = SpectralClustering(n_clusters=3)
        yhat = model.fit_predict(X)
        clusters = np.unique(yhat)
        for cluster in clusters:
            row_ix = np.where(yhat == cluster)
            plt.scatter(X[row_ix, 0], X[row_ix, 1])

        plt.title("SpectralClustering")
        plt.plot([x1, x2], [y1, y2], color="gray")
        plt.grid(True)

    def clustering_GaussianMixture(self, X, x1, x2, y1, y2):
        plt.subplot(339).invert_yaxis()
        model = GaussianMixture(n_components=3)
        model.fit(X)
        yhat = model.predict(X)
        clusters = np.unique(yhat)
        for cluster in clusters:
            row_ix = np.where(yhat == cluster)
            plt.scatter(X[row_ix, 0], X[row_ix, 1])

        plt.title("GaussianMixture")
        plt.plot([x1, x2], [y1, y2], color="gray")
        plt.grid(True)

    def calculate_rad(self, dx, dy, xc, yc):
        rad = np.arctan2(dy, dx)
        deg_pi = np.degrees(rad) + np.degrees(np.pi)
        idx_deg = copy.deepcopy(deg_pi)
        for i, _ in enumerate(idx_deg):
            if (
                ((6 < idx_deg[i] < 84))
                | ((96 < idx_deg[i] < 174))
                | ((186 < idx_deg[i] < 264))
                | ((276 < idx_deg[i] < 354))
            ):
                idx_deg[i] = 0 # have vanishing point
            elif (
                ((0 <= idx_deg[i] <= 6))
                | ((174 <= idx_deg[i] <= 186))
                | ((354 <= idx_deg[i] <= 360))
            ):
                idx_deg[i] = 1 # holizontal line
            elif ((84 <= idx_deg[i] <= 96)) | ((264 <= idx_deg[i] <= 276)):
                idx_deg[i] = 2 # vertical line
        print("deg_pi:")
        print(deg_pi)
        print("idx_deg:")
        print(idx_deg)
        self.__idx_deg = idx_deg

        # plot 3D (xc, yc, deg)
        fig = plt.figure()
        ax = fig.add_subplot(1,2,1,projection="3d")
        ax.set_xlabel("X",size=15,color="black")
        ax.set_ylabel("Y",size=15,color="black")
        ax.set_zlabel("degree",size=15,color="black")
        ax.invert_yaxis()
        ax.scatter(xc, yc, deg_pi, s=10, c="red")
        ax2 = fig.add_subplot(1,2,2,projection="3d")
        ax2.set_xlabel("X",size=15,color="black")
        ax2.set_ylabel("Y",size=15,color="black")
        ax2.set_zlabel("degree(manual clustered)",size=15,color="black")
        ax2.invert_yaxis()
        ax2.scatter(xc, yc, idx_deg, s=10, c="red")
        #plt.show()
