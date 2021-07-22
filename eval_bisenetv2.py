from utils_ls import *
import sys
sys.path.insert(0, '.')


def main():
    # step 1. init logger file
    args = parse_args()
    logger = init_logger(log_path=args.log_path, logger_name="bisenet_logger", phrase='val')

    # step 2. configs update and review
    configs = merge_configs(logger, args.project_config_path, args.default_config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = configs['device']['other_gpu_device']

    # step 3. init torch.distributed
    # init_distributed(configs['train_process']['port'], args.local_rank)

    # step 4. create dataset & dataloader
    class_data = parse_class_info(configs['data_set']['class_name_path'])
    # dataloader = gen_dataLoader(logger, 'train', configs, class_data)
    dataloader = gen_dataLoader(logger, 'val', configs, class_data)

    # step 5. declare network and load weights
    class_num = len(class_data)
    weight_path = os.path.join(configs['data_set']['model_and_temp_file_save_path'],
                               'segment_model', 'final_model.pth')
    net = gen_net(logger, class_num, weight_path)

    # setp 6. eval
    eval(logger, net, dataloader, class_num, configs['data_set']['model_and_temp_file_save_path'])


if __name__ == "__main__":
    main()
