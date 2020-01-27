import cv2 as cv

class MotionAnalyser:
    _blocks = []
    _bg_subtractor = None
    _bg_mask = None
    _frame_list = []
    _frame_sample_num = 16

    def __init__(self):
        pass

    # generate blocks' range
    # image_size: (tuple) size of image
    # block_size: (tuple) size of each block
    # return None
    def generate_blocks(self, image_size, block_size):
        image_width, image_height = image_size
        for y in range(0, image_height - block_size, block_size):
            for x in range(0, image_width - block_size, block_size):
                w = image_width - x if block_size + x > image_width else block_size
                h = image_height - y if block_size + y > image_height else block_size
                self._blocks.append((x, y, w, h))

    # initialize KNN Subtractor
    # return None
    def initialize_detect_object(self,
                                 frame_sample_num,
                                 dist_to_threshold,
                                 detect_shadows):
        self._frame_sample_num = frame_sample_num
        self._bg_subtractor = cv.createBackgroundSubtractorKNN(frame_sample_num,
                                                               dist_to_threshold,
                                                               detect_shadows)
        # self._bg_subtractor = cv.bgsegm.createBackgroundSubtractorMOG(frame_sample_num)
        return

    # return motion mask
    def detect_motion(self):
        if self._bg_subtractor is None:
            return 0
        ret, thresh = cv.threshold(self._bg_mask, 244, 255, cv.THRESH_BINARY)
        thresh = cv.erode(thresh, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3), (-1, -1)), iterations=2)
        thresh = cv.dilate(thresh, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3), (-1, -1)), iterations=2)
        return thresh

    # return motion block list
    def get_motion_blcoks(self, motion_map):
        intgral_image = cv.integral(motion_map)
        result_list = []
        for block in self._blocks:
            x, y, w, h = block
            t11 = intgral_image[y, x]
            t12 = intgral_image[y, x + w]
            t21 = intgral_image[y + h, x]
            t22 = intgral_image[y + w, x + h]
            if t22 - t12 - t21 + t11 > 0:
                result_list.append(block)
        return result_list

    # depend on is_auto apply return a copy of bg_mask or None
    def feed_image(self, image, is_auto_apply):
        if self._bg_subtractor is None:
            return 0
        self._frame_list.append(image)
        if len(self._frame_list) > self._frame_sample_num:
            self._frame_list.pop(0)
        if is_auto_apply:
            self._bg_mask = self._bg_subtractor.apply(image)
            return self._bg_mask.copy()
        return

    def apply_img(self):
        if self._bg_subtractor is None:
            return 0
        for frame in self._frame_list:
            self._bg_mask = self._bg_subtractor.apply(frame)
        return self._bg_mask.copy()
