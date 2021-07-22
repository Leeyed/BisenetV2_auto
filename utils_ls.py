import shutil
import random
import argparse
from torch.utils.data import DataLoader

from bisenetV2.meters import *
from bisenetV2.dataset import *
from bisenetV2.sampler import *
from bisenetV2.bisenetv2 import *
from bisenetV2.ohem_ce_loss import *
from bisenetV2.lr_scheduler import *

has_apex = True
try:
    from apex import amp, parallel
except ImportError:
    has_apex = False


# used in logger output
def newline(msg: str):
    return '\n' + msg


# step 1. init logger file
def init_logger(log_path: str, logger_name: str, phrase:str, log_level=logging.DEBUG):
    date = time.strftime("%Y-%m-%d", time.localtime())
    assert phrase in ('train', 'val', 'test')
    name = date+f' {phrase.capitalize()}.log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file_name = os.path.join(log_path, name)
    # 创建一个logger
    logger = logging.getLogger(logger_name)
    # 指定日志的最低输出级别，默认为WARN级别
    logger.setLevel(log_level)
    # create a folder
    path, file = os.path.split(log_file_name)
    if not os.path.exists(path):
        os.makedirs(path)
    # 创建一个handler用于写入日志文件
    file_handler = logging.FileHandler(log_file_name)
    # 创建一个handler用于输出控制台
    console_handler = logging.StreamHandler()
    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)-10s: %(filename)s, line:%(lineno)d %(levelname)s: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# step 2. configs update and review
def parse_args():
    parser = argparse.ArgumentParser(description="BiSeNetV2 training procedure.")
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument("--project_config_path", type=str,
                        default="./config_project.yaml",
                        help="项目配置文件.")
    parser.add_argument("--default_config_path", type=str,
                        default="./config_common.yaml",
                        help="默认配置文件.")
    parser.add_argument("--log_path", type=str,
                        default="./",
                        help="日志目录.")
    return parser.parse_args()


# step 2. configs update and review
def merge_configs(logger: logging.Logger, project_config_path: str, default_config_path: str):
    import yaml
    with open(project_config_path, 'r') as fd:
        project_configs = yaml.load(fd, Loader=yaml.FullLoader)
    with open(default_config_path, 'r') as fd:
        default_configs = yaml.load(fd, Loader=yaml.FullLoader)

    # merge configs
    default_configs.update(project_configs)

    def check_params(logger: logging.Logger, configs: dict):
        # necessary parameters 必要参数
        dataset_params = configs.get('data_set', 'NO_THIS_KEY')
        if not isinstance(dataset_params, dict):
            logger.error(f'Parameter: data_set = {dataset_params}')
            exit(1)

        # 'data_save_path_temp' 是不需要存在的
        nec_params = ['segmentation_train_file_path',
                      'segmentation_val_file_path']
        for key in nec_params:
            if not os.path.exists(dataset_params.get(key, 'NO_THIS_KEY')):
                logger.error(f'Parameter: {key} = {dataset_params.get(key, "NO_THIS_KEY")}')
                # the interrupt is not necessary to catch.
                # apply exit(), not sys.exit()
                exit(1)

        def list_params(configs):
            def parse_dict(param: dict, depth: int):
                res = ""
                for key, value in param.items():
                    if isinstance(value, dict):
                        res += f"{'    ' * depth}{key}\n"
                        res += parse_dict(value, depth + 1)
                    else:
                        res += f"{'    ' * depth} {key}:{value}\n"
                return res

            return '\nParameters Start:\n' + parse_dict(configs, 1) + 'Parameters End!\n'

        logger.info(list_params(configs))

    if logger is not None:
        check_params(logger, configs=default_configs)

    return default_configs


# step 3. data movement: from nfs to local disk
def dataset_movement(logger: logging.Logger, src: str, dst: str):
    if os.path.exists(dst):
        logger.info(newline(f'Parameter: local_root = {dst} exists'))
    else:
        # 目标目录不能存在，注意对dst目录父级目录要有可写权限，ignore的意思是排除
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('*.pyc', 'tmp*'))
        logger.info(newline(f'Copy data from {src} to {dst}!'))


class ImgInfo:
    def __init__(self, img_path, labels:list):
        self.img_path = img_path
        self.labels = labels


class Label:
    def __init__(self, label_name, points:list):
        self.label_name=label_name
        assert len(points)%2==0
        _pts = []
        points = list(map(lambda x: int(x), points))
        for i in range(len(points)>>1):
            _pts.append([points[2*i], points[2*i+1]])
        self.points = _pts


def parse_label_info(logger: logging.Logger, txt:str):
    """
    解析成：
    Class ImgInfo():
        self.img_name
        self.Labels

        sub class Label:
            self.label_name
            self.points_list
    """
    try:
        res = []
        with open(txt, 'r') as f:
            data_lines = f.readlines()
        for line in data_lines:
            piece_data = line.strip().split(',')
            _img_name, _labels = piece_data[0], piece_data[1:]
            # label = Label
            lbs = []
            for label in _labels:
                label = label.strip().split(' ')
                lbs.append(Label(label[0], label[1:]))
            res.append(ImgInfo(_img_name, lbs))
    except Exception as exp_msg:
        logger.error(newline(exp_msg))
        exit(1)
    return res


def move_data_and_makeMask(logger:logging.Logger, imgInfos:list, phrase:str, save_path:str, label2id:dict):
    if phrase == 'train':
        ori_phrase = 'TR-Image'
        mask_phrase = 'TR-Mask0'
    else:
        ori_phrase = 'VL-Image'
        mask_phrase = 'VL-Mask0'

    # create root
    if not os.path.exists(os.path.join(save_path, ori_phrase)):
        os.makedirs(os.path.join(save_path, ori_phrase))
    if not os.path.exists(os.path.join(save_path, mask_phrase)):
        os.makedirs(os.path.join(save_path, mask_phrase))

    # copy ori img
    start = time.time()
    for img_index,imgInfo in enumerate(imgInfos):
        # 'xxxx.jpg or .png ...'
        img_name = imgInfo.img_path.split('/')[-1]
        # only 'xxxx'
        img_name = img_name.split('.')[0]
        ori_path = os.path.join(save_path, ori_phrase, '%s.jpg' % img_name)
        cv2_img = cv2.imread(imgInfo.img_path)
        # shutil.copyfile(imgInfo.img_path, dst)
        cv2.imwrite(ori_path, cv2_img)

        ori_w, ori_h = cv2_img.shape[1], cv2_img.shape[0]
        mask = np.zeros([ori_h, ori_w, 1], np.uint8)
        for i, label in enumerate(imgInfo.labels):
            if label.label_name not in label2id.keys(): continue
            p = np.array(label.points)
            cv2.fillPoly(mask, [p], label2id[label.label_name])

        mask_path = os.path.join(save_path, mask_phrase, '%s.png' % img_name)
        # print(f'write {mask_path}')
        if time.time()-start>10:
            start = time.time()
            logger.info(newline(f'{phrase} data copying... {img_index+1}/{len(imgInfos)}'))
        cv2.imwrite(mask_path, mask)


def dataset_preparation(logger: logging.Logger, dataset_params: dict):
    # at least "val.txt" or "None"
    with open(dataset_params['segmentation_val_file_path'], 'r') as f:
        val_data = f.readlines()
    if len(val_data)<=1:
        imgInfos = parse_label_info(logger, dataset_params['segmentation_train_file_path'])
        train_radio = 0.8
        train_num = int(len(imgInfos)*train_radio)
        random.shuffle(imgInfos)
        trainImgInfos, valImgInfos = imgInfos[:train_num], imgInfos[train_num:]
    else:
        trainImgInfos = parse_label_info(logger, dataset_params['segmentation_train_file_path'])
        valImgInfos = parse_label_info(logger, dataset_params['segmentation_val_file_path'])
    # return trainImgInfos, valImgInfos

    # parse class_name.txt
    label2id = dict()
    with open(dataset_params['class_name_path']) as f:
        class_names = f.readlines()
    class_names = map(lambda x: x.strip(), class_names)
    for i, cls_name in enumerate(class_names):
        label2id[cls_name] = i

    try:
        move_data_and_makeMask(logger, trainImgInfos, 'train',
                               dataset_params['data_save_path_temp'], label2id)
        move_data_and_makeMask(logger, valImgInfos, 'val',
                               dataset_params['data_save_path_temp'], label2id)
    except Exception as exp_msg:
        logger.error(newline(exp_msg))
        exit(1)
    logger.info(newline(f'dataset prepared!'))


# step 4. init random seeds
def fix_random_seeds(logger: logging.Logger, seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    if logger is not None:
        logger.info(newline(f'Set all seeds to {seed}'))
        logger.info(newline(f'Set torch.backends.cudnn.deterministic to True'))


# step 5. init torch.distributed
def init_distributed(port: int, local_rank: int):
    torch.cuda.set_device(local_rank)
    # print('local_rank', local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        # init_method='tcp://127.0.0.1:9999',
        world_size=torch.cuda.device_count(),
        # 进程的编号，也是其优先级
        rank=local_rank
    )


# step 6. create dataset & dataloader
def parse_class_info(txt: str):
    with open(txt, 'r') as f:
        data = f.readlines()
    data = list(map(lambda x: x.strip(), data))
    return data


# step 6. create dataset & dataloader
def gen_dataLoader(logger: logging.Logger, phrase: str, configs: dict, class_data: list):
    # global anns_path, trans_func, shuffle, drop_last
    if phrase == 'train':
        trans_func = TransformationTrain(logger, configs)
        shuffle = True
        drop_last = True
    elif phrase == 'val':
        trans_func = TransformationVal(logger, configs)
        shuffle = False
        drop_last = False

    dataset_path = configs['data_set']['data_save_path_temp']
    batchsize = configs['train_process']['ims_per_gpu']
    # ds = sjht(datapth, annpath, numclass, trans_func=trans_func, mode=mode)
    dataset = sjhtCommon(dataset_path, class_data, trans_func, phrase)
    if dist.is_initialized():
        assert dist.is_available(), "dist should be initialzed"
        if phrase == 'train':
            assert isinstance(configs['train_process']['max_iter'], int)
            n_train_imgs = batchsize * dist.get_world_size() * configs['train_process']['max_iter']
            sampler = RepeatedDistSampler(dataset, n_train_imgs, shuffle=shuffle)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=shuffle)
            # sampler = RepeatedDistSampler(dataset, dist.get_world_size(), shuffle=shuffle)

        batchsampler = torch.utils.data.sampler.BatchSampler(
            sampler, batchsize, drop_last=drop_last
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batchsampler,
            num_workers=4,
            # num_workers=os.cpu_count(),
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batchsize,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4,
            # num_workers=os.cpu_count(),
            pin_memory=True,
        )
    # dist, and other process except main process
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(newline(f'Data loader has generated!'))
    return dataloader


# step 7. network, criteria
def gen_net_criteria(class_num: int, configs: dict):
    net = BiSeNetV2(class_num)

    # finetune
    # if not args.finetune_from is None:
    #     net.load_state_dict(torch.load(args.finetune_from, map_location='cpu'))
    def set_syncbn(_net):
        if has_apex:
            _net = parallel.convert_syncbn_model(_net)
        else:
            _net = nn.SyncBatchNorm.convert_sync_batchnorm(_net)
        return _net

    if configs['train_process']['use_sync_bn']: net = set_syncbn(net)
    net.cuda()
    net.train()
    criteria_pre = OhemCELoss(0.7)
    criteria_aux = [OhemCELoss(0.7) for _ in range(configs['train_process']['num_aux_heads'])]
    return net, criteria_pre, criteria_aux


def gen_net(logger:logging.Logger, class_num:int, weight_path:str):
    logger.info(newline(f'class number: {class_num}'))
    logger.info(newline(f'load weights: {weight_path}'))
    net = BiSeNetV2(class_num)
    net.load_state_dict(torch.load(weight_path))
    net.cuda()
    return net


# step 8. meters, optimizer, lr scheduler
def gen_meters(max_iter, num_aux_heads):
    time_meter = TimeMeter(max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i)) for i in range(num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters


# step 8. DistributedDataParallel training
def set_model_dist(net):
    if has_apex:
        net = parallel.DistributedDataParallel(net, delay_allreduce=True)
    else:
        local_rank = dist.get_rank()
        # print('in set_model_dist', local_rank)
        net = nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank, ],
            output_device=local_rank)
    return net


# step 8. meters, optimizer, lr scheduler
def gen_optimizer(model, lr_start: float, weight_decay: float):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': 0},
            {'params': lr_mul_wd_params, 'lr': lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr': lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=lr_start,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    return optim


# step 9. used in  train modules,
def cvt_train_info(iter: int, max_iter: int, lr, time_meter,
                   loss_meter, loss_pre_meter, loss_aux_meters):
    t_intv, eta = time_meter.get()
    loss_avg, _ = loss_meter.get()
    loss_pre_avg, _ = loss_pre_meter.get()
    loss_aux_avg = ' | '.join(['{}:{:.4f}'.format(el.name, el.get()[0]) for el in loss_aux_meters])
    msg = ' | '.join([
        'iteration:{it}/{max_it}',
        'learning rate:{lr:4f}',
        'eta:{eta}',
        'time:{time:.2f}',
        'loss:{loss:.4f}',
        'loss_pre:{loss_pre:.4f}',
    ]).format(
        it=iter + 1,
        max_it=max_iter,
        lr=lr,
        time=t_intv,
        eta=eta,
        loss=loss_avg,
        loss_pre=loss_pre_avg,
    )
    msg = '\n' + msg + ' | ' + loss_aux_avg
    # iter: 100 / 2000, lr: 0.006295, eta: 0:22: 41, time: 72.40, loss: 4.7601, loss_pre: 0.7768, loss_aux0: 0.9336, loss_aux1: 0.9652, loss_aux2: 1.0415, loss_aux3: 1.0430
    # res='iteration:%6s/%6s | learning rate:%-8s | time:%-6s | loss premeter:%-6s | loss_aux_meters:%-6s'\
    #     %(iter, max_iter, lr, time_meter, loss_meter, loss_pre_meter, loss_aux_meters)
    return msg


# step 9. train modules
def train(net, dataloader, optimizer, criteria_pre, criteria_aux, lr_scheduler, time_meter, loss_meter, loss_pre_meter,
          loss_aux_meters, logger, configs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(newline('Train start!'))
    for it, (im, lb) in enumerate(dataloader):
        im = im.cuda()
        lb = lb.cuda()

        lb = torch.squeeze(lb, 1)

        optimizer.zero_grad()
        logits, *logits_aux = net(im)
        loss_pre = criteria_pre(logits, lb)
        loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
        loss = loss_pre + sum(loss_aux)
        if has_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        lr_scheduler.step()

        time_meter.update()
        loss_meter.update(loss.item())
        loss_pre_meter.update(loss_pre.item())
        _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]

        ## print training log message
        if (it + 1) % 100 == 0:
            lr = lr_scheduler.get_lr()
            lr = sum(lr) / len(lr)
            logger.info(cvt_train_info(it, configs['train_process']['max_iter'], lr, time_meter,
                                       loss_meter, loss_pre_meter, loss_aux_meters))
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(newline('Train End'))


class MscEvalV0(object):
    def __init__(self, scales=(0.5, ), flip=False, ignore_label=255):
        self.scales = scales
        self.flip = flip
        self.ignore_label = ignore_label

    def __call__(self, net, dl, n_classes):
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        for i, (imgs, label) in enumerate(dl):
            N, _, H, W = label.shape
            label = label.squeeze(1).cuda()
            size = label.size()[-2:]
            probs = torch.zeros(
                    (N, n_classes, H, W), dtype=torch.float32).cuda().detach()
            for scale in self.scales:
                sH, sW = int(scale * H), int(scale * W)
                im_sc = F.interpolate(imgs, size=(sH, sW),
                        mode='bilinear', align_corners=True)

                im_sc = im_sc.cuda()
                logits = net(im_sc)[0]
                logits = F.interpolate(logits, size=size,
                        mode='bilinear', align_corners=True)
                probs += torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            keep = label != self.ignore_label
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
                ).view(n_classes, n_classes)
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        # print('miou', miou)
        return miou.item()


# eval module
@torch.no_grad()
def eval(logger:logging.Logger, net:BiSeNetV2, dataloader:DataLoader, class_num:int, result_path:str):
    net.eval()
    # heads, mious = [], []
    mious = []
    single_scale = MscEvalV0((1.,), False)
    mIOU = single_scale(net, dataloader, class_num)

    # heads.append('single_scale')
    mious.append(mIOU)
    logger.info(newline(f'single mIOU is: {mIOU}\n'))
    _res_path = os.path.join(result_path, 'test_result')
    if not os.path.exists(_res_path):
        os.makedirs(_res_path)
    with open(os.path.join(_res_path, 'result.txt'), 'w') as f:
        f.write(f'single mIOU is: {mIOU}\n')


# step 10. save the weights
def save_weights(save_dir: str, name: str, net: BiSeNetV2, logger: logging.Logger):
    save_dir = os.path.join(save_dir, 'segment_model')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_pth = os.path.join(save_dir, name)
    state = net.module.state_dict()
    if not dist.is_initialized() or dist.get_rank() == 0:
        # print(f'local rank{dist.get_rank() }')
        logger.info('save models to {}'.format(save_pth))
        torch.save(state, save_pth)


def cnt_area(contour):
    xmin = np.min(contour[:, :, 0])
    ymin = np.min(contour[:, :, 1])
    xmax = np.max(contour[:, :, 0])
    ymax = np.max(contour[:, :, 1])
    return np.sqrt(np.square(xmax - xmin) + np.square(ymax - ymin))


class segment_model:
    def __init__(self, weight_path:str, class_num:int, configs:dict):
        self.net = BiSeNetV2(class_num)
        self.net.load_state_dict(torch.load(weight_path))
        self.net.cuda()
        self.net.eval()
        self.to_tensor = T.ToTensor(
            mean=(0.3257, 0.3690, 0.3223), # city, rgb
            std=(0.2112, 0.2148, 0.2115),
        )
        trans=[]
        if configs['train_process']['use_resize']:
            trans.append(T.Resized(configs['train_process']['new_size']))
        if configs['train_process']['use_grey_img']:
            trans.append(T.ImgTransformGray())
        self.transform = T.Compose(trans)

    def forward(self, cv2_img):
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        im_lb = {"im": img_rgb, "lb": None}
        im = self.to_tensor(self.transform(im_lb))['im'].unsqueeze(0).cuda()
        out = self.net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
        return out
        # pass


def cut_object_imgs(img, mask, nums=5, boxsize=128, mode='', zoom=0.1):
    def takefirst(s):
        return s[0]

    def takesecond(s):
        return s[1]

    cut_imgs = []
    boxes = []
    h, w = mask.shape[:2]
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(boxsize * zoom), int(boxsize * zoom)))
    maskresize = cv2.resize(mask, (int(w * zoom), int(h * zoom)), cv2.INTER_NEAREST)
    maskresize = cv2.morphologyEx(maskresize, cv2.MORPH_ERODE, element)
    x = np.where(maskresize == 1)

    xys = np.vstack([x[1], x[0]]).T.tolist()
    if mode != 'random':
        w1 = np.max(x[1]) - np.min(x[1])
        h1 = np.max(x[0]) - np.min(x[0])
        if w1 > h1:
            xys = sorted(xys, key=takefirst)
        else:
            xys = sorted(xys, key=takesecond)
    centers = []
    distance = int(len(xys) / (nums + 1))
    for i in range(nums):
        centers.append(xys[int(distance * (i + 1))])

    for center in centers:
        boxes.append([int(center[0] / zoom - boxsize / 2), int(center[1] / zoom - boxsize / 2),
                      int(center[0] / zoom - boxsize / 2) + boxsize, int(center[1] / zoom - boxsize / 2) + boxsize])

    for box in boxes:
        cut_imgs.append(img[box[1]:box[3], box[0]:box[2], :])
        # cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 4)

    return cut_imgs, boxes