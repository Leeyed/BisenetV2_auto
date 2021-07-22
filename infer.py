# infer script
import time
import copy
from lines import GL
from utils_ls import *


def predict(net:segment_model, test_root:str, save_dir:str, class_num=int, highlight=False):
    imgs = os.listdir(test_root)
    imgs = list(filter(lambda x:x.endswith(('.jpg', '.png')), imgs))
    color_list = [
        (255, 165, 0),
        (255, 0, 0),
        (160, 32, 240),
        (144, 238, 144),
        (238, 238, 0),
    ]
    label_list = ['zsb', 'fs']
    # label_list = ['bp']
    for img in imgs:
        cv2_img = cv2.imread(os.path.join(test_root, img))
        start = time.time()
        out = net.forward(cv2_img)
        print(f'infer time: {time.time()-start}s per image!, img{test_root, img}')

        save_img = copy.deepcopy(cv2_img)

        for i in range(1, class_num):
            # resize
            predict_np = np.where(out == i, 1, 0).astype(np.uint8)
            resize_mask = cv2.resize(predict_np,
                                     (cv2_img.shape[1], cv2_img.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

            try:
                contours, _ = cv2.findContours(resize_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except:
                __, contours, _ = cv2.findContours(resize_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contours.sort(key=cnt_area, reverse=True)
            if len(contours) > 0:
                # debug
                cv2.drawContours(save_img, contours, 0, color_list[i], -1)

                new_mask = np.zeros((cv2_img.shape[0], cv2_img.shape[1], 3), dtype=np.uint8)
                cv2.drawContours(new_mask, contours, 0, (1, 1, 1), -1)

            # boxes = GL.get_lines_boxs(cv2_img, new_mask, boxs_num=5, box_size=128)
            # cut_imgs, boxes = cut_object_imgs(cv2_img, resize_mask)
            #
            # if len(boxes)!=5:
            #     print(f'img name: {img}, {label_list[i-1]}, {i}')
            #     # print(np.where(new_mask==1)[0].shape)
            #     # cv2.imwrite(os.path.join(save_dir, img), save_img)
            #     # cv2.imwrite(os.path.join(save_dir, 'mask_'+img), new_mask)
            #     # exit()
            #
            # print(np.where(new_mask == 1)[0].shape)
            # for index,box in enumerate(boxes):
            #     if not os.path.exists(save_dir):
            #         os.mkdir(save_dir)
            #     x0,y0,x1,y1 = box
            #     print(box)
            #     try:
            #         c_img = cv2_img[y0:y1,x0:x1,:]
            #         cv2.imwrite(os.path.join(save_dir,f'{label_list[i-1]}_{index+1}_{img}'), c_img)
            #     except:
            #         continue
            # print('save to', save_dir, img)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        cv2.imwrite(os.path.join(save_dir, img), save_img)
        # cv2.imwrite(os.path.join(save_dir, 'mask_'+img), new_mask)


def main():
    logger = init_logger(log_path='./logs', logger_name="bisenet_logger", phrase='test')

    args = parse_args()
    args.project_config_path = './config_project.yaml'
    configs = merge_configs(logger, args.project_config_path, args.default_config_path)

    class_data = parse_class_info(configs['data_set']['class_name_path'])
    class_num = len(class_data)

    # weight_path = os.path.join(configs['data_set']['model_and_temp_file_save_path'], 'segment_model', 'final_model.pth')
    weight_path = os.path.join('/home/liusheng/data/images4code/ai_model_ckpt/sjht/sjht-bdnb-275/1.1/model_and_temp_file', 'segment_model', 'final_model.pth')
    logger.info(newline(f"load weights: {weight_path}"))
    logger.info(newline(f"class number: {class_num}"))
    net = segment_model(
        weight_path = weight_path,
        class_num = class_num,
        configs = configs
    )

    # bt_list = os.listdir(configs['data_set']['data_save_path_temp'])
    bt_list = [
        'test'
    ]
    for batch in bt_list:
        predict(net,
                os.path.join(configs['data_set']['data_save_path_temp'], batch),
                os.path.join(configs['data_set']['data_save_path_temp'], batch+'_boxes'),
                class_num,
                True
                )


if __name__ == "__main__":
    main()