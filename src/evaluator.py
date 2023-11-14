#
#
#
import time
import datetime
import numpy as np
import itertools
from tabulate import tabulate
import torch

import detectron2.utils.comm as comm
from detectron2.utils.logger import create_small_table, log_every_n_seconds
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator, inference_context
from detectron2.data import DatasetMapper, build_detection_test_loader


__all__=['COCOEvaluator','LossEvalHook']

########################################################################
#
# monkey patch for calculate AP_x for each class
#
def _derive_coco_results_x(self, coco_eval, iou_type, class_names=None):
    metrics = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }[iou_type]
        
    if coco_eval is None:
        self._logger.warn("No predictions from the model!")
        return {metric: float("nan") for metric in metrics}
        
    results = {
        metric: float(
            coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan"
        ) for idx, metric in enumerate(metrics)
    }
    self._logger.info(
        "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
    )
    if not np.isfinite(sum(results.values())):
        self._logger.info("Some metrics cannot be computed and is shown as NaN.")
    
    if class_names is None or len(class_names) <= 1:
        return results
    
    precisions = coco_eval.eval["precision"]
    assert len(class_names) == precisions.shape[2]
    
    results_per_category = []
    for idx, name in enumerate(class_names):
        for t,n in zip((slice(None,None),0,5),('','50','75')):
            precision = precisions[t, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}{}".format(name,n), float(ap * 100)))
    
    N_COLS = min(6, len(results_per_category) * 2)
    results_flatten = list(itertools.chain(*results_per_category))
    results_2d = itertools.zip_longest(*[
        results_flatten[i::N_COLS] for i in range(N_COLS)
    ])
    table = tabulate(
        results_2d,
        tablefmt="pipe",
        floatfmt=".3f",
        headers=["category", "AP"] * (N_COLS // 2),
        numalign="left",
    )
    self._logger.info("Per-category {} AP: \n".format(iou_type) + table)
    
    results.update({"AP-" + name: ap for name, ap in results_per_category})
    return results

COCOEvaluator._derive_coco_results=_derive_coco_results_x
#
#
#
########################################################################



########################################################################
#
# Evalhook for loss calculation wile validation
#
# originate from 
# https://eidos-ai.medium.com/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
# https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b
#
class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
        
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = \
                    (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(
                    seconds=int(total_seconds_per_img * (total - idx - 1))
                )
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
            
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()
        
        return losses
    
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)
#
#
#
########################################################################
