

import argparse
import math
import os
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from geopy import distance as geopy_distance
from tqdm import tqdm

from path_utils import dataset_data_dir, normalize_runtime_paths, tile_emb_filename, tiles_dir


DATASET_CONFIGS = {
    'porto': {
        'boundary': {
            'min_lat': 41.140092, 'max_lat': 41.185969,
            'min_lng': -8.690261, 'max_lng': -8.549155,
        },
        'grid_size': 0.1,
        'zoom': 18,
    },
    'cd': {
        'boundary': {
            'min_lat': 30.655, 'max_lat': 30.727,
            'min_lng': 104.043, 'max_lng': 104.129,
        },
        'grid_size': 0.1,
        'zoom': 18,
    },
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DINO_REPO = 'facebookresearch/dinov2'


def grid_mapping(boundary, grid_size):
    lat_dist = geopy_distance.distance(
        (boundary['min_lat'], boundary['min_lng']),
        (boundary['max_lat'], boundary['min_lng'])
    ).km
    lat_size = (boundary['max_lat'] - boundary['min_lat']) / lat_dist * grid_size

    lng_dist = geopy_distance.distance(
        (boundary['min_lat'], boundary['min_lng']),
        (boundary['min_lat'], boundary['max_lng'])
    ).km
    lng_size = (boundary['max_lng'] - boundary['min_lng']) / lng_dist * grid_size

    lat_grid_num = int(lat_dist / grid_size) + 1
    lng_grid_num = int(lng_dist / grid_size) + 1
    return lat_size, lng_size, lat_grid_num, lng_grid_num


def grid_id_to_latlon(grid_id, boundary, lat_size, lng_size, lng_grid_num):
    grid_i = grid_id // lng_grid_num
    grid_j = grid_id % lng_grid_num
    lat = boundary['min_lat'] + (grid_i + 0.5) * lat_size
    lng = boundary['min_lng'] + (grid_j + 0.5) * lng_size
    return lat, lng


def latlon_to_tile(lat, lng, zoom):
    n = 2 ** zoom
    tile_x = int((lng + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    tile_y = int(
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    )
    return tile_x, tile_y


def default_image_size(backbone: str) -> int:
    if backbone == 'resnet50':
        return 224
    return 518


def get_transform(backbone: str, image_size: int):
    _ = backbone
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_resnet50(device):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()
    return model, 2048


def load_dinov2(backbone: str, device, repo_path: str = '', weights_path: str = ''):
    repo_or_dir = DINO_REPO
    hub_kwargs = {}
    repo_path = str(repo_path or '').strip()
    weights_path = str(weights_path or '').strip()

    if repo_path:
        repo_or_dir = str(Path(repo_path).expanduser().resolve())
        hub_kwargs['source'] = 'local'
    else:
        hub_kwargs['trust_repo'] = True

    if weights_path:
        hub_kwargs['weights'] = str(Path(weights_path).expanduser().resolve())

    try:
        model = torch.hub.load(repo_or_dir, backbone, **hub_kwargs)
    except TypeError:
        hub_kwargs.pop('trust_repo', None)
        model = torch.hub.load(repo_or_dir, backbone, **hub_kwargs)
    model = model.to(device)
    model.eval()
    feature_dim = int(getattr(model, 'embed_dim', 0) or 0)
    if feature_dim <= 0:
        dummy = torch.zeros(1, 3, default_image_size(backbone), default_image_size(backbone), device=device)
        with torch.inference_mode():
            outputs = model.forward_features(dummy)
        feature_dim = int(outputs['x_norm_clstoken'].shape[-1])
    return model, feature_dim


def resolve_dinov2_pooling(outputs, pooling: str):
    cls_token = outputs['x_norm_clstoken']
    patch_tokens = outputs['x_norm_patchtokens']

    if pooling == 'cls':
        return cls_token
    patch_mean = patch_tokens.mean(dim=1)
    if pooling == 'patch_mean':
        return patch_mean
    if pooling == 'cls_patch_mean':
        return 0.5 * (cls_token + patch_mean)
    raise ValueError(f'Unsupported dinov2_pooling: {pooling}')


def load_backbone(
    backbone: str,
    device,
    dinov2_repo_path: str = '',
    dinov2_weights_path: str = '',
    dinov2_pooling: str = 'cls',
):
    if backbone == 'resnet50':
        model, feature_dim = load_resnet50(device)

        def extract_fn(batch_tensor):
            feats = model(batch_tensor)
            return feats.squeeze(-1).squeeze(-1)

        return model, extract_fn, feature_dim

    if backbone in {'dinov2_vits14', 'dinov2_vitb14'}:
        model, feature_dim = load_dinov2(
            backbone,
            device,
            repo_path=dinov2_repo_path,
            weights_path=dinov2_weights_path,
        )

        def extract_fn(batch_tensor):
            outputs = model.forward_features(batch_tensor)
            return resolve_dinov2_pooling(outputs, dinov2_pooling)

        return model, extract_fn, feature_dim

    raise ValueError(f'Unsupported tile backbone: {backbone}')


def extract_tile_features(image_paths, extract_fn: Callable[[torch.Tensor], torch.Tensor], transform, device, batch_size=32):
    tile_features = {}

    for i in tqdm(range(0, len(image_paths), batch_size), desc='Extracting tile features'):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        valid_names = []

        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                batch_images.append(transform(img))
                valid_names.append(os.path.splitext(os.path.basename(img_path))[0])
            except Exception as e:
                print(f'  Warning: 无法处理 {img_path}: {e}')

        if not batch_images:
            continue

        batch_tensor = torch.stack(batch_images).to(device)
        with torch.inference_mode():
            feats = extract_fn(batch_tensor)

        for name, feat in zip(valid_names, feats):
            tile_features[name] = feat.detach().cpu().float().numpy()

    return tile_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='porto', choices=['porto', 'cd'], help='Dataset Name')
    parser.add_argument('--device', type=str, default='cuda:0', help='computing devices')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size when extracting features')
    parser.add_argument(
        '--tile_backbone',
        type=str,
        default='resnet50',
        choices=['resnet50', 'dinov2_vits14', 'dinov2_vitb14'],
        help='瓦片视觉骨干网络',
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=0,
        help='Input image size; 0 This indicates that the default value (ResNet) should be used based on the backbone.=224, DINOv2=518）',
    )
    parser.add_argument('--project_root', type=str, default='')
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--dataset_data_dir', type=str, default='')
    parser.add_argument('--tiles_root', type=str, default='')
    parser.add_argument('--tiles_dir', type=str, default='')
    parser.add_argument('--dinov2_repo_path', type=str, default='', help='Optional local DINOv2 repo path for offline loading.')
    parser.add_argument('--dinov2_weights_path', type=str, default='', help='Optional local DINOv2 weight path for offline loading.')
    parser.add_argument(
        '--dinov2_pooling',
        type=str,
        default='cls',
        choices=['cls', 'patch_mean', 'cls_patch_mean'],
        help='Feature pooling used for DINOv2 outputs.',
    )
    parser.add_argument('--output_path', type=str, default='')
    args = parser.parse_args()
    args = normalize_runtime_paths(args)

    cfg = DATASET_CONFIGS[args.dataset]
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    image_size = int(args.image_size) if int(args.image_size) > 0 else default_image_size(args.tile_backbone)
    output_path = args.output_path or str(dataset_data_dir(args) / tile_emb_filename(args.tile_backbone))

    print(f'dataset: {args.dataset}')
    print(f'device: {device}')
    print(f'tile_backbone: {args.tile_backbone}')
    print(f'size: {image_size}')

    boundary = cfg['boundary']
    lat_size, lng_size, lat_grid_num, lng_grid_num = grid_mapping(boundary, cfg['grid_size'])
    token_size = lat_grid_num * lng_grid_num
    print(f'\gird: {lat_grid_num} x {lng_grid_num} = {token_size} point')

    current_tiles_dir = tiles_dir(args)
    if not os.path.exists(current_tiles_dir):
        print(f'error: not exist: {current_tiles_dir}')
        return

    image_paths = sorted([
        os.path.join(current_tiles_dir, f)
        for f in os.listdir(current_tiles_dir)
        if f.endswith('.png') or f.endswith('.jpg')
    ])
    print(f'find {len(image_paths)} tile images')

    _, extract_fn, feature_dim = load_backbone(
        args.tile_backbone,
        device,
        dinov2_repo_path=args.dinov2_repo_path,
        dinov2_weights_path=args.dinov2_weights_path,
        dinov2_pooling=args.dinov2_pooling,
    )
    transform = get_transform(args.tile_backbone, image_size)
    tile_features = extract_tile_features(image_paths, extract_fn, transform, device, args.batch_size)
    print(f'Successfully extracted {len(tile_features)} Features and dimensions of an image={feature_dim}')

    zoom = cfg['zoom']
    grid_tile_emb = {}
    found_count = 0
    missing_grids = []

    for grid_id in range(token_size):
        lat, lng = grid_id_to_latlon(grid_id, boundary, lat_size, lng_size, lng_grid_num)
        tile_x, tile_y = latlon_to_tile(lat, lng, zoom)
        tile_name = f'{tile_x}_{tile_y}'

        if tile_name in tile_features:
            grid_tile_emb[grid_id] = tile_features[tile_name]
            found_count += 1
        else:
            missing_grids.append(grid_id)

    missing_count = len(missing_grids)
    print('\n-mapping statistics:')
    print(f'  Image embedding: {found_count}/{token_size} ({100 * found_count / token_size:.1f}%)')
    print(f'  Missing tiles:   {missing_count}/{token_size} ({100 * missing_count / token_size:.1f}%)')
    if 0 < missing_count <= 20:
        print(f'  missing grid_id: {missing_grids[:20]}')

    payload = {
        'meta': {
            'dataset': args.dataset,
            'backbone': args.tile_backbone,
            'dinov2_pooling': args.dinov2_pooling if args.tile_backbone.startswith('dinov2_') else '',
            'feature_dim': int(feature_dim),
            'image_size': int(image_size),
            'token_size': int(token_size),
            'found_count': int(found_count),
            'missing_count': int(missing_count),
        },
        'grid_tile_emb': grid_tile_emb,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(payload, f)

    print(f'\nsave: {output_path}')
    print(f'File format: bundle {{ meta, grid_tile_emb }}，Feature Dimension={feature_dim}')
    print('finished')


if __name__ == '__main__':
    main()
