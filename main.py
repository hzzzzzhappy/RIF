import contextlib
import logging
import os
import sys
import numpy as np
import torch
import time
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
import click
from bisect import bisect
from dataset_pc import Dataset3dad_train, Dataset3dad_test
from patchcore import PatchCore  
from cpu_knn import Aggregator, NearestNeighbourScorer, FaissNN, IdentitySampler, ApproximateGreedyCoresetSampler
LOGGER = logging.getLogger(__name__)

@click.group(chain=True)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--memory_size", type=int, default=10000, show_default=True)
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
@click.option("--faiss_on_gpu", is_flag=True, default=True)
@click.option("--faiss_num_workers", type=int, default=8)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    gpu,
    seed,
    memory_size,
    anomaly_scorer_num_nn,
    faiss_on_gpu,
    faiss_num_workers
):
    methods = {key: item for (key, item) in methods}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result_collect = []
    root_dir = 'data/anomaly-shapenetraw'
    log_file_path = '?.txt'
    real_3d_classes = ['ashtray0', 'bag0', 'bottle0', 'bottle1', 'bottle3', 'bowl0', 'bowl1', 'bowl2', 'bowl3', 'bowl4', 'bowl5', 'bucket0', 'bucket1', 'cap0', 'cap3', 'cap4', 'cap5', 'cup0', 'cup1', 'eraser0', 'headset0', 'headset1', 'helmet0', 'helmet1', 'helmet2', 'helmet3', 'jar0', 'microphone0', 'shelf0', 'tap0', 'tap1', 'vase0', 'vase1', 'vase2', 'vase3', 'vase4', 'vase5', 'vase7', 'vase8', 'vase9']
    
    for dataset_count, dataset_name in enumerate(real_3d_classes):
        LOGGER.info(f"Evaluating dataset [{dataset_name}] ({dataset_count + 1}/{len(real_3d_classes)})...")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        train_loader = DataLoader(Dataset3dad_train(root_dir, dataset_name, 1024, True), num_workers=1, batch_size=1, shuffle=False, drop_last=False)
        test_loader = DataLoader(Dataset3dad_test(root_dir, dataset_name, 1024, True), num_workers=1, batch_size=1, shuffle=False, drop_last=False)
        
        torch.cuda.empty_cache()
        sampler = methods["get_sampler"](device)

        patchcore_model = PatchCore(device=device,num_groups=512, feature_extractor_type="CTF")
        nn_method = FaissNN(faiss_on_gpu, faiss_num_workers)

        torch.cuda.empty_cache()
        memory_feature = patchcore_model.fit(train_loader)
        aggregator_fpfh = {"scores": [], "segmentations": []}
        start_time = time.time()

        scores_fpfh, segmentations_fpfh, labels_gt, masks_gt = patchcore_model.predict(test_loader)
        aggregator_fpfh["scores"].append(scores_fpfh)
        scores_fpfh = np.array(aggregator_fpfh["scores"])
        min_scores_fpfh = scores_fpfh.min(axis=-1).reshape(-1, 1)
        max_scores_fpfh = scores_fpfh.max(axis=-1).reshape(-1, 1)
        scores_fpfh = (scores_fpfh - min_scores_fpfh) / (max_scores_fpfh - min_scores_fpfh)
        scores_fpfh = np.mean(scores_fpfh, axis=0)

         
        ap_seg_fpfh = np.concatenate(segmentations_fpfh, axis=1).flatten()
        ap_seg_fpfh = (ap_seg_fpfh - np.min(ap_seg_fpfh)) / (np.max(ap_seg_fpfh) - np.min(ap_seg_fpfh))

        end_time = time.time()
        time_cost = (end_time - start_time) / len(test_loader)

        LOGGER.info("Computing evaluation metrics.")
        scores = scores_fpfh
        ap_seg = ap_seg_fpfh
        auroc = roc_auc_score(labels_gt, scores)
        img_ap = average_precision_score(labels_gt, scores)
        ap_mask = np.concatenate([mask for mask in masks_gt], axis=1).flatten().astype(np.int32)
        pixel_ap = average_precision_score(ap_mask, ap_seg)
        full_pixel_auroc = roc_auc_score(ap_mask, ap_seg)

        print(f'Task:{dataset_name}, image_auc:{auroc}, pixel_auc:{full_pixel_auroc}, image_ap:{img_ap}, pixel_ap:{pixel_ap}, time_cost:{time_cost}')
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Task:{dataset_name}, image_auc:{auroc}, pixel_auc:{full_pixel_auroc}, image_ap:{img_ap}, pixel_ap:{pixel_ap}, time_cost:{time_cost}\n")
        result_collect.append((auroc, full_pixel_auroc, img_ap, pixel_ap, time_cost))

    avg_results = np.mean(result_collect, axis=0)
    LOGGER.info(f"Average Results - AUC: {avg_results[0]}, Pixel AUC: {avg_results[1]}, AP: {avg_results[2]}, Pixel AP: {avg_results[3]}, Time Cost: {avg_results[4]}")


@main.command("sampler")
@click.argument("name", type=str, default="approx_greedy_coreset")
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return IdentitySampler()
        elif name == "greedy_coreset":
            return GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return ApproximateGreedyCoresetSampler(percentage, device)

    return ("get_sampler", get_sampler)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
