import cv2 as cv
from classifier import Classifier
from motion_analyser import MotionAnalyser
import tqdm
import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0 1"

labels = ['fire', 'normal', 'smoke']

# input options
video_file = './example/product_video/Pexels Videos 2629.mp4'
model = './example/output_graph_mobilenet_v2_100_224.pb'
model_input_width = 224
model_input_height = 224
model_faster = './example/output_graph_mobilenet_v2_100_224.pb'
model_faster_width = 224
model_faster_height = 224

# output options
output_video = True
output_video_filename = './example/detected_test.mp4'
output_photo = True
output_photo_directory = './example/out_photo/'
previous_wnd = True

# detect options
sample_capacity = 16
block_size = 30
disable_motion_block = False

mask_colors = [(0x00, 0x2c, 0xdd), (0x50, 0xaf, 0x4c), (0xb5, 0x51, 0x3f)]
mask_alpha = 0.5


def main():
    # init
    classifier = Classifier((model_input_width, model_input_height),
                            graph_filename=model,
                            input_layer='Placeholder',
                            output_layer='final_result')
    faster_classifier = Classifier((model_faster_width, model_faster_height),
                                   graph_filename=model_faster,
                                   input_layer='Placeholder',
                                   output_layer='final_result')
    classifier.init_sess()
    faster_classifier.init_sess()

    capt_source = cv.VideoCapture(video_file)
    if not capt_source.isOpened():
        print('[Error!]: Open video failed')
        return
    video_writer = cv.VideoWriter()
    if output_video:
        video_writer.open(output_video_filename,
                          int(capt_source.get(cv.CAP_PROP_FOURCC)),
                          capt_source.get(cv.CAP_PROP_FPS),
                          (int(capt_source.get(cv.CAP_PROP_FRAME_WIDTH)),
                           int(capt_source.get(cv.CAP_PROP_FRAME_HEIGHT))))
        if not video_writer.isOpened():
            print('[Error!]: Create output video file failed')

    if previous_wnd:
        cv.namedWindow("Fire Detector - view", cv.WINDOW_AUTOSIZE)

    motion_analyser = MotionAnalyser()
    motion_analyser.initialize_detect_object(sample_capacity, 400.0, False)
    motion_analyser.generate_blocks(
        (int(capt_source.get(cv.CAP_PROP_FRAME_WIDTH)), int(capt_source.get(cv.CAP_PROP_FRAME_HEIGHT))),
        block_size)

    count = 0
    frame_max_count = capt_source.get(cv.CAP_PROP_FRAME_COUNT)
    start_time = cv.getTickCount()

    process_bar = tqdm.tqdm(total=frame_max_count)
    while capt_source.isOpened():
        ret, frame = capt_source.read()
        if output_photo:
            cv.imwrite('{base_path}/output_ori_{filename}.jpg'.format(base_path=output_photo_directory, filename=count),
                       frame)
        if not ret:
            break
        classification_result, classification_topk = classifier.classify(frame)
        if not disable_motion_block:
            motion_analyser.feed_image(frame, True)
            if count > sample_capacity:
                bg_mask = motion_analyser.detect_motion()
                if output_photo:
                    cv.imwrite('{base_path}/output_bg_{filename}.jpg'.format(base_path=output_photo_directory, filename=count),
                            bg_mask)
                # cv.imshow('bw', bg_mask)
                # cv.waitKey(1)
                blocks = motion_analyser.get_motion_blcoks(bg_mask)
                # here we ignored multi-thread
                for block in blocks:
                    x, y, w, h = block
                    block_image = frame[y:y + h, x:x + w]
                    block_result, block_topk = faster_classifier.classify(block_image)
                    # if block_result != 1:
                    cv.rectangle(frame, (x + 1, y + 1), (x + w - 1, y + h - 1), mask_colors[block_result], 1)
        count = count + 1
        process_bar.update(1)
        # frame = cv.addWeighted(mask, mask_alpha, frame, 1.0 - mask_alpha, 0)

        cv.rectangle(frame, (20, 10), (400, 155), mask_colors[classification_result], 1)
        cv.putText(frame,
                   'Frame:{0}/{1}'.format(count, frame_max_count),
                   (30, 30),
                   cv.FONT_HERSHEY_COMPLEX_SMALL,
                   1,
                   mask_colors[classification_result])
        cv.putText(frame,
                   labels[classification_result],
                   (30, 60),
                   cv.FONT_HERSHEY_COMPLEX_SMALL,
                   1,
                   mask_colors[classification_result])
        cv.putText(frame,
                   'Fire:{0}'.format(classification_topk[0]),
                   (30, 90),
                   cv.FONT_HERSHEY_COMPLEX_SMALL,
                   1,
                   mask_colors[classification_result])
        cv.putText(frame,
                   'Normal:{0}'.format(classification_topk[1]),
                   (30, 120),
                   cv.FONT_HERSHEY_COMPLEX_SMALL,
                   1,
                   mask_colors[classification_result])
        cv.putText(frame,
                   'Smoke:{0}'.format(classification_topk[2]),
                   (30, 150),
                   cv.FONT_HERSHEY_COMPLEX_SMALL,
                   1,
                   mask_colors[classification_result])

        if previous_wnd:
            cv.imshow('Fire Detector - view', frame)
            cv.waitKey(1)

        if output_video:
            video_writer.write(frame)
        if output_photo:
            cv.imwrite('{base_path}/output_{filename}.jpg'.format(base_path=output_photo_directory, filename=count),
                       frame)

    process_bar.close()
    capt_source.release()
    print('\nProcess Over')
    print('Result:{0}'.format(output_video_filename))
    end_time = cv.getTickCount()
    cost_time = (end_time - start_time) * 1000 / cv.getTickFrequency()
    print('Cost time:{0}ms'.format(cost_time))
    print('AVR FPS:{0}'.format(frame_max_count / (cost_time / 1000)))


if __name__ == '__main__':
    main()
