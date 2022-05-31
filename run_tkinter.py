import argparse
#from lu_vp_detect import VPDetection
import vp_mlsd
import tensorflow as tf
import os
from pathlib import Path
import scipy.io


# Set up argument parser + options
parser = argparse.ArgumentParser(
    description="Main script for Lu's Vanishing Point Algorithm")
# parser.add_argument('-i',
#                     '--image-path',
#                     help='Path to the input image',
#                     required=True)
parser.add_argument('-m',
                    '--model_path',
                    default='tflite_models/M-LSD_512_large_fp32.tflite',
                    type=str,
                    help='path to tflite model')
parser.add_argument('-is',
                    '--input_size',
                    default=512,
                    type=int,
                    choices=[512, 320],
                    help='input size')
parser.add_argument('-st',
                    '--score_thr',
                    default=0.2,
                    type=float,
                    help='score thr')
parser.add_argument('-dt',
                    '--dist_thr',
                    default=10,
                    type=float,
                    choices=range(0, 21),
                    help='dist thr')

parser.add_argument('-lt',
                    '--length-thresh',
                    default=30,
                    type=float,
                    help='Minimum line length (in pixels) for detecting lines')  # 線の最小ピクセル数
parser.add_argument(
                    '-pp',
                    '--principal-point',
                    default=None,
                    nargs=2,
                    type=float,
                    help='Principal point of the camera (default is image centre)')
parser.add_argument('-f',
                    '--focal-length',
                    default=1500,
                    type=float,
                    help='Focal length of the camera (in pixels)')
parser.add_argument('-d',
                    '--debug',
                    action='store_true',
                    help='Turn on debug image mode')
parser.add_argument('-ds',
                    '--debug-show',
                    action='store_true',
                    help='Show the debug image in an OpenCV window')
parser.add_argument('-dp',
                    '--debug-path',
                    default=None,
                    help='Path for writing the debug image')
parser.add_argument('-s',
                    '--seed',
                    default=None,
                    type=int,
                    help='Specify random seed for reproducible results')
args = parser.parse_args()

def main(score, dist, length, model_path, image_path, pp, focal_len, vp_path, vp):
    input_path = image_path
    length_thresh = length
    principal_point = pp
    focal_length = focal_len
    debug_mode = args.debug
    debug_show = args.debug_show
    debug_path = args.debug_path
    seed = args.seed
    model_path = model_path
    input_size = args.input_size
    score_thr = score
    dist_thr = dist

    # Load tflite model
    interpreter = tf.lite.Interpreter(model_path)
    # メモリ確保，モデル読み込み直後に必須
    interpreter.allocate_tensors()
    # 学習モデルの入力層・出力層のプロパティをGet 辞書型のリスト構造
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create object
    vpd = vp_mlsd.VPDetection(length_thresh, principal_point, focal_length, seed)

    vp_path = vp_path
    gt_vp=vp['vp_orthogonal']
    #gt_vp=vp['vp']

    vpd.find_vps( input_path,
                        interpreter,
                        input_details,
                        output_details,
                        input_shape=[input_size, input_size],
                        score_thr=score_thr,
                        dist_thr=dist_thr,
                        gt_vp=gt_vp)  # 画像のパスを生成、その画像に対し実行

    vp2D = vpd.vps_2D
    vp2D_gt = vpd.vps_2D_gt
    img = vpd.create_debug_VP_image(debug_path)

    return vp2D, vp2D_gt, img

if __name__ == "__main__":
    main()
