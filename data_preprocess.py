from utils_ls import *

# apex
has_apex = True
try:
    from apex import amp, parallel
except ImportError:
    has_apex = False


def main():
    # step 1. init logger file
    args = parse_args()
    # args.log_path = '/home/liusheng/data/images4code/ai_model_ckpt/sjht/sjht-bdnb-275/1.0/model_and_temp_file/logs_serial'
    # args.project_config_path = '/home/liusheng/data/images4code/ai_model_ckpt/sjht/sjht-bdnb-275/1.0/config_project.yaml'

    logger = init_logger(log_path=args.log_path, logger_name="bisenet_logger", phrase='train')

    # step 2. configs update and review
    configs = merge_configs(logger, args.project_config_path, args.default_config_path)

    # step 4. init random seeds
    fix_random_seeds(logger, configs['train_process']['random_seed'])
    # step 3. data movement: from nfs to local disk
    dataset_preparation(logger,  configs['data_set'])


if __name__ == "__main__":
    main()


