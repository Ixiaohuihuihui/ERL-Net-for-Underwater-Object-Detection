from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import os
import mmcv

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
    # video = mmcv.VideoReader('/data3/dailh/Brackish/archive/dataset/videos/crab/2019-03-21_07-40-40to2019-03-21_07-40-50_1.avi')
    # i = 0
    # for frame in video:
    #     i += 1
    #     result = inference_detector(model, frame)
    #     model.show_result(frame, result, wait_time=1, out_file='./demo/brackish/' + str(i) + '.jpg')


if __name__ == '__main__':
    main()
