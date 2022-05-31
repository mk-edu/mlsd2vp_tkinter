import scipy.io
import os
import run_tkinter
import numpy as np
import warnings


def test(score, dist, length, image_path, image_name, width, height):
    camera = scipy.io.loadmat('./YorkUrbanDB/cameraParameters.mat')
    model_path = "./tflite_models/M-LSD_512_large_fp32.tflite"
    focal = camera['focal'][0,0]
    pp = camera['pp'][0,:]

    focal_len = width*(focal-5.7)
    vp_path = os.path.join("./YorkUrbanDB", image_name, image_name+"GroundTruthVP_Orthogonal_CamParams.mat")
    #vp_path = os.path.join("./YorkUrbanDB", image_name, image_name+"GroundTruthVP_CamParams.mat")

    vp = scipy.io.loadmat(vp_path)
    # vp1_orthogonal = vp['vp_orthogonal'][0,:]
    # vp2_orthogonal = vp['vp_orthogonal'][1,:]
    # vp3_orthogonal = vp['vp_orthogonal'][2,:]

    print("-----------detail--------------")
    print("score:", score)
    print("dist:", dist)
    print("len:", length)
    print("model_path:", model_path)
    print("image_path:", image_path)
    print("image_name:", image_name)
    print("image width:", width)
    print("image height:", height)
    print('principal points:', pp)
    print("focal length:", focal_len)
    print("vp_path:", vp_path)
    print("-------------------------------")

    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    vp2D_mlsd, vp2D_gt, img = np.array(run_tkinter.main(score, dist, length, model_path, image_path, pp, focal_len, vp_path, vp))
    np.set_printoptions(precision=2, suppress=True)

    return vp2D_mlsd, vp2D_gt, img




if __name__ =="__main__":
    test()
