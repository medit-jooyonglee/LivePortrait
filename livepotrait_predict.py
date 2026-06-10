# coding: utf-8

"""
LivePortrait inference script with simple file-path I/O interface.

Required pretrained weights (relative to repo root pretrained_weights/):
  liveportrait/base_models/
    appearance_feature_extractor.pth   # F: appearance feature extractor
    motion_extractor.pth               # M: motion extractor
    spade_generator.pth                # G: SPADE generator
    warping_module.pth                 # W: warping module
  liveportrait/retargeting_models/
    stitching_retargeting_module.pth   # S: stitching & retargeting
  liveportrait/
    landmark.onnx                      # facial landmark detector
  insightface/                         # face detection models
    models/buffalo_l/                  # or buffalo_s

  (animal mode only)
  liveportrait_animals/base_models_v1.1/
    appearance_feature_extractor.pth
    motion_extractor.pth
    spade_generator.pth
    warping_module.pth
  liveportrait_animals/
    xpose.pth                          # animal keypoint detector

Usage examples:
  python predict.py --source portrait.jpg --driving drive.mp4 --output result.mp4
  python predict.py -s portrait.jpg -d drive.mp4 -o out.mp4 --no-half --cpu
  python predict.py -s portrait.jpg -d drive.mp4 -o out.mp4 --region lip --no-stitch
"""
import os
import glob
import cv2
import os.path as osp
import sys
import argparse
import subprocess
import numpy as np
from sklearn import pipeline
# ensure imports resolve from this file's directory


from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
from src.live_portrait_pipeline import LivePortraitPipeline, load_image_rgb


def _check_ffmpeg():
    ffmpeg_dir = osp.join(_HERE, "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += os.pathsep + ffmpeg_dir
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except Exception:
        raise RuntimeError(
            "FFmpeg not found. Install FFmpeg and make sure it is on PATH. "
            "https://ffmpeg.org/download.html"
        )


def _partial(cls, d: dict):
    return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


def parse_args():
    p = argparse.ArgumentParser(
        description="LivePortrait: animate a source portrait driven by a video or image."
    )

    # I/O
    p.add_argument("-s", "--source", 
                #    required=True,
                   default='assets/examples/source/yellowteeth.jpg',
                   help="Source portrait image (.jpg/.png) or video (.mp4/.avi)")
    p.add_argument("-d", "--driving",
                #    required=True,
                   default='assets/examples/driving/d30.jpg',
                   help="Driving video (.mp4/.avi), image (.jpg/.png), or motion template (.pkl)")
    p.add_argument("-o", "--output", default='res.jpg',
                   help="Output file path (.mp4 for video, .jpg/.png for image). "
                        "Default: animations/<source>--<driving>.mp4")

    # Motion / animation
    p.add_argument("--region",
                   choices=["all", "exp", "pose", "lip", "eyes"],
                   default="all",
                   help="Animation region (default: all)")
    p.add_argument("--driving-option",
                   choices=["expression-friendly", "pose-friendly"],
                   default="expression-friendly",
                   help="Motion transfer mode (default: expression-friendly)")
    p.add_argument("--driving-multiplier", type=float, default=1.5,
                   help="Motion intensity multiplier for expression-friendly mode (default: 1.0)")
    p.add_argument("--no-relative", action="store_true",
                   help="Disable relative motion (use absolute driving motion)")
    p.add_argument("--no-stitch", action="store_true",
                   help="Disable stitching (recommended for large head movements or animals)")
    p.add_argument("--no-pasteback", action="store_true",
                   help="Do not paste the animated face back onto the original image")
    p.add_argument("--no-crop", action="store_true",
                   help="Skip face cropping (source is already cropped/aligned)")
    p.add_argument("--crop-driving", action="store_true",
                   help="Crop the driving video before processing")

    # Lip / eye retargeting
    p.add_argument("--normalize-lip", action="store_true",
                   help="Normalize lip to closed state before animation")
    p.add_argument("--eye-retargeting", action="store_true",
                   help="Transfer eye-open ratio from driving to source (WIP)")
    p.add_argument("--lip-retargeting", action="store_true", default=False,
                   help="Transfer lip-open ratio from driving to source (WIP)")

    # Audio
    p.add_argument("--audio-priority", choices=["source", "driving"], default="driving",
                   help="Which video's audio to keep in output (default: driving)")

    # Hardware
    p.add_argument("--no-half", action="store_true",
                   help="Use FP32 instead of FP16 (try if black boxes appear on some GPUs)")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU inference (slow)")
    p.add_argument("--device-id", type=int, default=0,
                   help="CUDA device index (default: 0)")

    # Cropping geometry
    p.add_argument("--scale", type=float, default=2.3,
                   help="Face crop scale — larger = more context area (default: 2.3)")
    p.add_argument("--vx", type=float, default=0.0,
                   help="Horizontal crop offset ratio (default: 0)")
    p.add_argument("--vy", type=float, default=-0.125,
                   help="Vertical crop offset ratio, negative = shift up (default: -0.125)")
    p.add_argument("--source-max-dim", type=int, default=1280,
                   help="Max dimension of source image/video before processing (default: 1280)")

    return p.parse_args()


g_model: LivePortraitPipeline = None


def get_config():
    
    try:    
        _check_ffmpeg()
    except Exception as e:
        print(f"FFmpeg check failed: {e}")
        # pass
        # raise RuntimeError("FFmpeg is required for video processing. Please install FFmpeg and ensure it is on PATH.")
    ns = parse_args()

    # if not osp.exists(ns.source):
    #     raise FileNotFoundError(f"Source not found: {ns.source}")
    # if not osp.exists(ns.driving):
    #     raise FileNotFoundError(f"Driving not found: {ns.driving}")

    # output_dir = osp.join(_HERE, "animations")
    if ns.output:
        out_abs = osp.abspath(ns.output)
        output_dir = osp.dirname(out_abs) or output_dir
        os.makedirs(output_dir, exist_ok=True)

    arg_dict = dict(
        source=osp.abspath(ns.source),
        driving=osp.abspath(ns.driving),
        output_dir=output_dir,
        # motion
        animation_region=ns.region,
        driving_option=ns.driving_option,
        driving_multiplier=ns.driving_multiplier,
        flag_relative_motion=not ns.no_relative,
        flag_stitching=not ns.no_stitch,
        flag_pasteback=not ns.no_pasteback,
        flag_do_crop=not ns.no_crop,
        flag_crop_driving_video=ns.crop_driving,
        # retargeting
        flag_normalize_lip=ns.normalize_lip,
        flag_eye_retargeting=ns.eye_retargeting,
        flag_lip_retargeting=ns.lip_retargeting,
        # audio
        audio_priority=ns.audio_priority,
        # hardware
        flag_use_half_precision=not ns.no_half,
        flag_force_cpu=ns.cpu,
        device_id=ns.device_id,
        # geometry
        scale=ns.scale,
        vx_ratio=ns.vx,
        vy_ratio=ns.vy,
        source_max_dim=ns.source_max_dim,
    )

    args = _partial(ArgumentConfig, arg_dict)
    inference_cfg = _partial(InferenceConfig, arg_dict)
    crop_cfg = _partial(CropConfig, arg_dict)
    return args, inference_cfg, crop_cfg
    
    
def get_model() -> LivePortraitPipeline:
    global g_model
    output_dir = 'temp/results'
    if g_model is None:
        args, inference_cfg, crop_cfg = get_config()

        pipeline = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
        g_model = pipeline
    return g_model


def update_config(dest_config, new_config):
    if isinstance(new_config, dict):
        for key, value in new_config.items():
            if hasattr(dest_config, key):
                print(f"Set inference config: {key} = {value}")
            else:
                print(f"Warning: unknown inference config key: {key}, setting anyway")
            setattr(dest_config, key, value)
            
def predict(src_img, driving_img, 
        
            args:ArgumentConfig=None,
            inference_config:InferenceConfig=None, **kwargs):
    try:
        pipeline = get_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        pass
        # return None, None, None
    # if args is None:
    default_args, inference_cfg, crop_cfg = get_config()
    if args is not None:
        update_config(default_args, args)
        
    
    # driving_img
    
    # pipeline.live_portrait_wrapper.inference_cfg.driving_multiplier = driving_multiplier
    if inference_config is not None:
        update_config(pipeline.live_portrait_wrapper.inference_cfg, inference_config)
        # if isinstance(inference_config, dict):
        #     for key, value in inference_config.items():
        #         if hasattr(pipeline.live_portrait_wrapper.inference_cfg, key):
        #             print(f"Set inference config: {key} = {value}")
        #         else:
        #             # setattr(pipeline.live_portrait_wrapper.inference_cfg, key, value)
        #             print(f"Warning: unknown inference config key: {key}, setting anyway")
        #         setattr(pipeline.live_portrait_wrapper.inference_cfg, key, value)
            # inference_config = _partial(InferenceConfig, inference_config)
        # pipeline.live_portrait_wrapper.inference_cfg = inference_config
    # pinpeline.live_portrait_wrapper.inference_cfg.driving_option = "pose-friendly"
    wfp, wfp_concat, image_lists = pipeline.execute(default_args, 
                                                    write_image=False, 
                                                    src_image=src_img, 
                                                    driving_image=driving_img)
    
    return image_lists[0], (wfp, wfp_concat)
    

def main():
    pipeline = get_model()
    
    args, inference_cfg, crop_cfg = get_config()
    origin_run = False
    if origin_run:
        pipeline.live_portrait_wrapper.inference_cfg.driving_option = "pose-friendly"
        pipeline.live_portrait_wrapper.inference_cfg.flag_lip_retargeting = True
        pipeline.live_portrait_wrapper.inference_cfg.driving_multiplier = 1.4
        # args.source = 'yellow2.jpg'
        # args.source = 'E:/temp/smile/smile02.jpg'
        src = 'E:/temp/smile/yellow/yellow05.jpg'
        args.source = src
        # args.driving = 'assets/examples/teeth/good02.jpg'
        args.driving = 'assets/examples/driving/d30.jpg'
        # args.driving = 'assets/examples/source/yellowteeth.jpg'
        
        wfp, wfp_concat = pipeline.execute(args)

        # Move to caller-specified path if given
        if ns.output and osp.abspath(ns.output) != osp.abspath(wfp):
            os.replace(wfp, ns.output)
            print(f"Output: {ns.output}")
        else:
            print(f"Output: {wfp}")

        print(f"Concat: {wfp_concat}")

    
    # pipeline.live_portrait_wrapper.inference_cfg.driving_option = "pose-friendly"
    # pipeline.live_portrait_wrapper.inference_cfg.flag_lip_retargeting = True
    # pipeline.live_portrait_wrapper.inference_cfg.driving_multiplier = 1.3
    # args.source = 'yellow2.jpg'
    args.source = 'res.jpg'
    # args.output = 're'
    # args.driving = 'assets/examples/teeth/good02.jpg'
    # args.driving = 'E:/temp/smile/smile01.jpg'
    # args.driving = 'E:/temp/smile/smile04.jpg'
    args.driving = 'E:/temp/smile/yellow/yellow03.jpg'
    pipeline.live_portrait_wrapper.inference_cfg.driving_option = "pose-friendly"
    pipeline.live_portrait_wrapper.inference_cfg.flag_lip_retargeting = True
    pipeline.live_portrait_wrapper.inference_cfg.driving_multiplier = 1.2
    wfp, wfp_concat, image_lists = pipeline.execute(args, write_image=False)

    # new_outpout = 'E:/temp/smile/smile01.jpg'
    
    import shutil
    new_outout = 'res1.jpg'
    # Move to caller-specified path if given
    if new_outout and osp.abspath(new_outout) != osp.abspath(wfp):
        shutil.copy(wfp, new_outout)
        print(f"Output: {new_outout}")
    else:
        print(f"Output: {wfp}")

    print(f"Concat: {wfp_concat}")
    
    files = [src, 'res.jpg', 'res1.jpg']
    
    import cv2
    imgs = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        imgs.append(img)
        
    shape = imgs[-1].shape[:2]
    for idx, img in enumerate(imgs):
        if img.shape[:2] == shape:
            pass
        else:
            imgs[idx] = cv2.resize(img, (shape[1], shape[0]))
    concat_image = np.concatenate(imgs, axis=1)
    cv2.imwrite('res_concat.jpg', concat_image)
    
        # print(f"{f}: {os.path.getsize(f)} bytes")

    # concat_image = 
    
    
def concatenate_results(image_lists, refer_index=-1, axis=1):
    
    refer_image = image_lists[refer_index]
    
    size = tuple(refer_image.shape[:2][::-1])
    
    out_image_lists = []
    for idx, img in enumerate(image_lists):
        img = cv2.resize(img, size)
        out_image_lists.append(img)
        
    out_image_lists = np.concatenate(out_image_lists, axis=axis)
    return out_image_lists
    

def predict_main():
    
    # src_image_file = 'liveportrait/res--d30.jpg'
    src_image_file = 'E:/temp/dataset/samples/teeth03.png'
    test_src_image_files = []
    
    src_image_files = glob.glob('E:/temp/dataset/samples/*.*')
    # target_image_file = 'liveportrait/res--smile01.jpg'
    # target_image_file = 'E:/temp/dataset/closed/refer01.png'
    target_image_file = 'samples/img02.png'
    target_image_file2 = 'E:/temp/dataset/opend_good/good03.jpg'
    # 'D:\workspace\repositories\iSmileNet\liveportrait/'
    # hidden-middle-image
    driving_img = load_image_rgb(target_image_file)
    driving_img2 = load_image_rgb(target_image_file2)
    for src_image_file in src_image_files:
            # print(f"Processing {src_image_file} with driving {target_image_file}...")
        src_img = load_image_rgb(src_image_file)
        
        infer_config = {
            'driving_multiplier': 1.0,
            'flag_lip_retargeting': True,
        }
        # res, (wfp, _) = predict(src_img, driving_img, 
        #               args=None, inference_config=infer_config)
        # cv2.imwrite('temp.png', res[..., ::-1])
        # infer_config['flag_lip_retargeting'] = False
        # res2, (wfp, _) = predict(res, driving_img2, 
        #         args=None, inference_config=infer_config)
        # # cv2.imwrite('temp2.png', src_img[..., ::-1])
        # cv2.imwrite('temp2.png', res2[..., ::-1])
        res = src_img
        
        
        # driving_video = 'D:/workspace/repositories/iSmileNet/liveportrait/assets/examples/driving/d18.mp4'
        
        driving_video = 'C:/Users/medit/Downloads/smile_video.mp4'
        
        ####
        
        if True:
            
            try:
                pipeline = get_model()
            except Exception as e:
                print(f"Error loading model: {e}")
                pass
                # return None, None, None
            args, inference_cfg, crop_cfg = get_config()
            
            args.driving = driving_video
            args.output_dir = 'temp/results'
            args.wfp_concat_write = False
            os.makedirs(args.output_dir, exist_ok=True)
            pipeline.live_portrait_wrapper.inference_cfg.driving_multiplier = 1.0
            pipeline.live_portrait_wrapper.inference_cfg.flag_lip_retargeting = False
            
            # pinpeline.live_portrait_wrapper.inference_cfg.driving_option = "pose-friendly"
            wfp, wfp_concat, image_lists = pipeline.execute(args, 
                                                            
                                                            src_image=res, 
                                                            )
            
        # load_image_rgb
        # # import cv2
        # fname = os.path.splitext(os.path.basename(src_image_file))[0]
        # cv2.imwrite(f'{fname}_res1.png', res[..., ::-1])
        # cv2.imwrite(f'{fname}_res2.png', res2[..., ::-1])
        # cv2.imwrite(f'{fname}_res_concat.png', concatenated[..., ::-1])
        
    
    
if __name__ == "__main__":
    _HERE = ''
    # _HERE = osp.dirname(osp.realpath(__file__))
    # os.chdir(_HERE)
    # sys.path.insert(0, _HERE)

    # main()
    predict_main()