'''
M-LSD
Copyright 2021-present NAVER Corp.
Apache License v2.0
'''
import os
import numpy as np
import cv2


def pred_lines(image, interpreter, input_details, output_details, input_shape=[512, 512], score_thr=0.10, dist_thr=20.0):
    h, w = image.shape[:2]
    h_ratio, w_ratio = [h / input_shape[0], w / input_shape[1]]
    resized_image = np.concatenate([cv2.resize(image, (input_shape[0],input_shape[1]), interpolation=cv2.INTER_AREA), np.ones([input_shape[0], input_shape[1], 1])], axis=-1)

    batch_image = np.expand_dims(resized_image, axis=0).astype('float32')
    # indexにテンソルデータのポインタをセット
    interpreter.set_tensor(input_details[0]['index'], batch_image)
    # 推論実行
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    pts = interpreter.get_tensor(output_details[0]['index'])[0]
    pts_score = interpreter.get_tensor(output_details[1]['index'])[0]
    vmap = interpreter.get_tensor(output_details[2]['index'])[0]

    start = vmap[:,:,:2] #(x1, y1)
    end = vmap[:,:,2:] #(x2, y2)
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1)) # 2点間の距離

    segments_list = []
    for center, score in zip(pts, pts_score):
        y, x = center
        distance = dist_map[y, x]
        # print("center:   ", center)
        # print("distance: ", distance)
        #print("score:    ", score)
        #print("vmap:     ", vmap[y, x, :])
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])

    lines = 2 * np.array(segments_list) # 256 > 512
    lines[:,0] = lines[:,0] * w_ratio
    lines[:,1] = lines[:,1] * h_ratio
    lines[:,2] = lines[:,2] * w_ratio
    lines[:,3] = lines[:,3] * h_ratio

    return lines
