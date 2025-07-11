from argparse import ArgumentParser
from PIL import Image
import torch
import torch.utils.data
import numpy as np
import torchvision
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import matplotlib
from matplotlib import pyplot as plt
import pathlib
import datetime
import copy

from stp3.datas.NuscenesData import FuturePredictionDataset
from stp3.trainer import TrainingModule
from stp3.metrics import IntersectionOverUnion, PanopticMetric, PlanningMetric
from stp3.utils.network import preprocess_batch, NormalizeInverse
from stp3.utils.instance import predict_instance_segmentation_and_trajectories
from stp3.utils.visualisation import make_contour

def mk_save_dir():
    now = datetime.datetime.now()
    string = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
    save_path = pathlib.Path('imgs') / string
    save_path.mkdir(parents=True, exist_ok=False)
    return save_path

def eval(checkpoint_path, dataroot):
    save_path = mk_save_dir()

    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
    print(f'Loaded weights from \n {checkpoint_path}')
    trainer.eval()

    device = torch.device('cuda:0')
    trainer.to(device)
    model = trainer.model

    cfg = model.cfg
    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 1
    cfg.LIFT.GT_DEPTH = False
    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.MAP_FOLDER = dataroot
    cfg.DATASET.VERSION='mini'

    dataroot = cfg.DATASET.DATAROOT
    #nworkers = cfg.N_WORKERS
    nworkers = 2  
    
    nusc = NuScenes(version='v1.0-{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=False)
    valdata = FuturePredictionDataset(nusc, 1, cfg)
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False
    )

    n_classes = len(cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS)
    hdmap_class = cfg.SEMANTIC_SEG.HDMAP.ELEMENTS
    metric_vehicle_val = IntersectionOverUnion(n_classes).to(device)
    future_second = int(cfg.N_FUTURE_FRAMES / 2)

    if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
        metric_pedestrian_val = IntersectionOverUnion(n_classes).to(device)

    if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
        metric_hdmap_val = []
        for i in range(len(hdmap_class)):
            metric_hdmap_val.append(IntersectionOverUnion(2, absent_score=1).to(device))

    if cfg.INSTANCE_SEG.ENABLED:
        metric_panoptic_val = PanopticMetric(n_classes=n_classes).to(device)

    if cfg.PLANNING.ENABLED:
        metric_planning_val = []
        for i in range(future_second):
            metric_planning_val.append(PlanningMetric(cfg, 2*(i+1)).to(device))


    for index, batch in enumerate(tqdm(valloader)):
        preprocess_batch(batch, device)
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']
        command = batch['command']
        trajs = batch['sample_trajectory']
        
        
        ######
        curr_traj = batch['current_direction_trajectory']
        
        
        
        target_points = batch['target_point']
        B = len(image)
        labels = trainer.prepare_future_labels(batch)

        with torch.no_grad():
            output = model(
                image, intrinsics, extrinsics, future_egomotion
            )
        #['image', 'intrinsics', 'extrinsics', 'depths', 'segmentation', 'instance', 'centerness', 'offset', 'flow', 'pedestrian', 'future_egomotion', 'hdmap', 'gt_trajectory', 'indices', 'command', 'sample_trajectory', 'current_direction_trajectory', 'target_point']
        # --- Calcolo CaCE per il concetto 'pedestrian' ---
        # Crea una copia del batch con l'intervento
        intervened_batch = copy.deepcopy(batch)
        intervened_image = intervened_batch['image']
        intervened_intrinsics = intervened_batch['intrinsics']
        intervened_extrinsics = intervened_batch['extrinsics']
        intervened_future_egomotion = intervened_batch['future_egomotion']

        with torch.no_grad():
            output_intervened = model(
                intervened_image, intervened_intrinsics, intervened_extrinsics, intervened_future_egomotion
            )

        

        n_present = model.receptive_field

        # semantic segmentation metric
        seg_prediction = output['segmentation'].detach()
        seg_prediction = torch.argmax(seg_prediction, dim=2, keepdim=True)
        metric_vehicle_val(seg_prediction[:, n_present - 1:], labels['segmentation'][:, n_present - 1:])

        if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            pedestrian_prediction = output['pedestrian'].detach()
            pedestrian_prediction = torch.argmax(pedestrian_prediction, dim=2, keepdim=True)
            metric_pedestrian_val(pedestrian_prediction[:, n_present - 1:],
                                       labels['pedestrian'][:, n_present - 1:])
        else:
            pedestrian_prediction = torch.zeros_like(seg_prediction)

        if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
            for i in range(len(hdmap_class)):
                hdmap_prediction = output['hdmap'][:, 2 * i:2 * (i + 1)].detach()
                hdmap_prediction = torch.argmax(hdmap_prediction, dim=1, keepdim=True)
                metric_hdmap_val[i](hdmap_prediction, labels['hdmap'][:, i:i + 1])

        if cfg.INSTANCE_SEG.ENABLED:
            pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
                output, compute_matched_centers=False, make_consistent=True
            )
            metric_panoptic_val(pred_consistent_instance_seg[:, n_present - 1:],
                                     labels['instance'][:, n_present - 1:])

        if cfg.PLANNING.ENABLED:
            occupancy = torch.logical_or(seg_prediction, pedestrian_prediction)
            _, final_traj, aggregated_costs, aggregated_costs_worst, norm_best, norm_worst = model.planning(
                cam_front=output['cam_front'].detach(),
                trajs=trajs[:, :, 1:],
                gt_trajs=labels['gt_trajectory'][:, 1:],
                cost_volume=output['costvolume'][:, n_present:].detach(),
                semantic_pred=occupancy[:, n_present:].squeeze(2),
                hd_map=output['hdmap'].detach(),
                commands=command,
                target_points=target_points
            )

        if cfg.PLANNING.ENABLED:

                # Traiettoria senza pedoni (intervento)
            seg_prediction_intervened = torch.zeros_like(seg_prediction)
            pedestrian_prediction_intervened = torch.zeros_like(pedestrian_prediction)

            occupancy_intervened = torch.logical_or(seg_prediction_intervened, pedestrian_prediction_intervened)
            _, final_traj_intervened, aggregated_costs_intervened, _, norm_cost_intervened, _ = model.planning(
                cam_front=output_intervened['cam_front'].detach(),
                trajs=trajs[:, :, 1:],
                gt_trajs=labels['gt_trajectory'][:, 1:],
                cost_volume=output_intervened['costvolume'][:, n_present:].detach(),
                semantic_pred=occupancy_intervened[:, n_present:].squeeze(2),
                hd_map=output_intervened['hdmap'].detach(),
                commands=command,
                target_points=target_points
            )

            # Computation of costs for the current trajectory projection
            _, _, aggregated_costs_current, _, norm_current, _ = model.planning(
              cam_front=output['cam_front'].detach(),
              trajs=curr_traj[:, :, 1:],
              gt_trajs=labels['gt_trajectory'][:, 1:],
              cost_volume=output['costvolume'][:, n_present:].detach(),
              semantic_pred=occupancy[:, n_present:].squeeze(2),
              hd_map=output['hdmap'].detach(),
              commands=command,
              target_points=target_points,
              all_traj=trajs[:, :, 1:]
            )

            occupancy = torch.logical_or(labels['segmentation'][:, n_present:].squeeze(2),
                                         labels['pedestrian'][:, n_present:].squeeze(2))
            for i in range(future_second):
                cur_time = (i+1)*2
                metric_planning_val[i](final_traj[:,:cur_time].detach(), labels['gt_trajectory'][:,1:cur_time+1], occupancy[:,:cur_time])
            
           
        n_present = model.receptive_field

        # Calcola la differenza media assoluta tra le traiettorie (CaCE)
        cace_effect = (final_traj_intervened - final_traj).abs().mean().item()



        if index % 10 == 0:
            save(output, labels, batch, n_present, index, save_path, planned_traj=final_traj, aggregated_costs=aggregated_costs, aggregated_costs_worst=aggregated_costs_worst, norm_best=norm_best, norm_worst=norm_worst, aggregated_costs_current=aggregated_costs_current, norm_current=norm_current,aggregated_costs_intervened=aggregated_costs_intervened, final_traj_intervened=final_traj_intervened, norm_cost_intervened= norm_cost_intervened,cace_effect=cace_effect)


    results = {}

    scores = metric_vehicle_val.compute()
    results['vehicle_iou'] = scores[1]

    if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
        scores = metric_pedestrian_val.compute()
        results['pedestrian_iou'] = scores[1]

    if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
        for i, name in enumerate(hdmap_class):
            scores = metric_hdmap_val[i].compute()
            results[name + '_iou'] = scores[1]

    if cfg.INSTANCE_SEG.ENABLED:
        scores = metric_panoptic_val.compute()
        for key, value in scores.items():
            results['vehicle_'+key] = value[1]

    if cfg.PLANNING.ENABLED:
        for i in range(future_second):
            scores = metric_planning_val[i].compute()
            for key, value in scores.items():
                results['plan_'+key+'_{}s'.format(i+1)]=value.mean()

    for key, value in results.items():
        print(f'{key} : {value.item()}')

def check_pixel_collision(trajectory, image_array, bx, dx, color_targets):
    trajectory_np = trajectory.astype(int)
    #print(trajectory_np)
    for x, y in trajectory_np:
        if 0 <= y < image_array.shape[0] and 0 <= x < image_array.shape[1]:
            pixel = image_array[y, x]
            #print(f"Pixel at ({x},{y}): {pixel}")
            for target_color in color_targets:
                if np.allclose(pixel, target_color, atol=0.1):  # tolleranza di colore
                    return (x, y)
    return None


def save(output, labels, batch, n_present, frame, save_path,
         planned_traj=None, aggregated_costs=None, aggregated_costs_worst=None,
         norm_best=None, norm_worst=None, aggregated_costs_current=None,
         norm_current=None, aggregated_costs_intervened=None, final_traj_intervened=None,norm_cost_intervened=None,cace_effect=None):

    import torchvision
    import matplotlib.pyplot as plt
    import matplotlib.gridspec
    import numpy as np
    import torch
    from PIL import Image

    # Funzione per invertire normalizzazione immagine
    denormalise_img = torchvision.transforms.Compose(
        (NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         torchvision.transforms.ToPILImage(),)
    )

    val_w = 2.99
    val_h = 2.99 * (224. / 480.)
    plt.figure(1, figsize=(5*val_w, 2*val_h))  # spazio più largo per 2 subplot traiettorie
    width_ratios = (val_w, val_w, val_w, val_w, val_w)
    gs = matplotlib.gridspec.GridSpec(2, 5, width_ratios=width_ratios)
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    images = batch['image']
    hdmap = output['hdmap'].detach()
    segmentation = output['segmentation'][:, n_present - 1].detach()
    pedestrian = output['pedestrian'][:, n_present - 1].detach()
    gt_trajs = labels['gt_trajectory']

    # --- Subplot immagini (come nel tuo codice originale) ---
    # FRONT LEFT
    plt.subplot(gs[0, 0])
    plt.annotate('FRONT LEFT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0,n_present-1,0].cpu()))
    plt.axis('off')

    # FRONT
    plt.subplot(gs[0, 1])
    plt.annotate('FRONT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0,n_present-1,1].cpu()))
    plt.axis('off')

    # FRONT RIGHT
    plt.subplot(gs[0, 2])
    plt.annotate('FRONT RIGHT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0,n_present-1,2].cpu()))
    plt.axis('off')

    # BACK LEFT
    plt.subplot(gs[1, 0])
    plt.annotate('BACK LEFT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    showing = denormalise_img(images[0,n_present-1,3].cpu())
    showing = showing.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(showing)
    plt.axis('off')

    # BACK
    plt.subplot(gs[1, 1])
    plt.annotate('BACK', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    showing = denormalise_img(images[0, n_present - 1, 4].cpu())
    showing = showing.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(showing)
    plt.axis('off')

    # BACK RIGHT
    plt.subplot(gs[1, 2])
    plt.annotate('BACK', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    showing = denormalise_img(images[0, n_present - 1, 5].cpu())
    showing = showing.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(showing)
    plt.axis('off')


       # --- Subplot mappa hdmap e segmentazione (gs[:,3]) ---
    plt.subplot(gs[:, 3])
    showing = torch.zeros((200, 200, 3)).numpy()
    showing[:, :] = np.array([219 / 255, 215 / 255, 215 / 255])

    # drivable
    area = torch.argmax(hdmap[0, 2:4], dim=0).cpu().numpy()
    hdmap_index = area > 0
    showing[hdmap_index] = np.array([161 / 255, 158 / 255, 158 / 255])

    # lane
    area = torch.argmax(hdmap[0, 0:2], dim=0).cpu().numpy()
    hdmap_index = area > 0
    showing[hdmap_index] = np.array([84 / 255, 70 / 255, 70 / 255])

    # semantic
    semantic_seg = torch.argmax(segmentation[0], dim=0).cpu().numpy()
    semantic_index = semantic_seg > 0
    showing[semantic_index] = np.array([255 / 255, 128 / 255, 0 / 255])

    pedestrian_seg = torch.argmax(pedestrian[0], dim=0).cpu().numpy()
    pedestrian_index = pedestrian_seg > 0
    showing[pedestrian_index] = np.array([28 / 255, 81 / 255, 227 / 255])

    pedestrian_color = np.array([28 / 255, 81 / 255, 227 / 255])  # blu
    semantic_color = np.array([255 / 255, 128 / 255, 0 / 255])    # arancione
    lane_color = np.array([84 / 255, 70 / 255, 70 / 255])  #grigio scuro color
    collision_colors = [pedestrian_color, semantic_color,np.array([1,1,1]),lane_color]
    #showing plot without traj
    showing_flipped = np.flipud(showing)
    plt.imshow(make_contour(showing_flipped))
    plt.axis('off')
    showing_corrected = np.fliplr(showing)

    # --- Prepara coordinate e dimensioni veicolo ---
    bx = np.array([-50.0 + 0.5/2.0, -50.0 + 0.5/2.0])
    dx = np.array([0.5, 0.5])
    w, h = 1.85, 4.084
    pts = np.array([
        [-h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, -w / 2.],
        [-h / 2. + 0.5, -w / 2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]

    # --- Plot Ground Truth e traiettoria originale (gs[:,4]) ---
    ax_traj_orig = plt.subplot(gs[:, 4])
    ax_traj_orig.imshow(make_contour(showing))
    ax_traj_orig.axis('off')
    ax_traj_orig.fill(pts[:, 0], pts[:, 1], '#76b900')
    ax_traj_orig.set_xlim((200, 0))
    ax_traj_orig.set_ylim((0, 200))

    gt_trajs = labels['gt_trajectory'].cpu().numpy()
    gt_trajs[0, :, :1] = gt_trajs[0, :, :1] * -1
    gt_trajs = (gt_trajs[0, :, :2] - bx) / dx
    ax_traj_orig.plot(gt_trajs[:, 0], gt_trajs[:, 1], linewidth=3.0, label='Ground Truth')

    if planned_traj is not None:
        planned = planned_traj[0].detach().cpu().numpy()
        planned[:, 0] = -planned[:, 0]
        planned = (planned[:, :2] - bx) / dx
        ax_traj_orig.plot(planned[:, 0], planned[:, 1], linewidth=2.0, color='red', label='Planned Trajectory')
        hit = check_pixel_collision(planned, showing_corrected, bx, dx, collision_colors)

        if hit:
          ax_traj_orig.plot(hit[0], hit[1], 'x', color='black', markersize=10) 
    ax_traj_orig.set_title('Original Trajectory')

    # --- Plot traiettoria dopo intervento (CaCE) (gs[:,5]) ---
    if final_traj_intervened is not None:
        # Se gs ha solo 5 colonne, usa un nuovo figure per evitare problemi
        fig2 = plt.figure(2, figsize=(val_w*2, val_h*2))
        ax_traj_interv = fig2.add_subplot(1,1,1)
        ax_traj_interv.imshow(make_contour(showing))
        ax_traj_interv.axis('off')
        ax_traj_interv.fill(pts[:, 0], pts[:, 1], '#76b900')
        ax_traj_interv.set_xlim((200, 0))
        ax_traj_interv.set_ylim((0, 200))

        final_interv = final_traj_intervened[0].detach().cpu().numpy()
        final_interv[:, 0] = -final_interv[:, 0]
        final_interv = (final_interv[:, :2] - bx) / dx

        if planned_traj is not None:
          planned = planned_traj[0].detach().cpu().numpy()
          planned[:, 0] = -planned[:, 0]
          planned = (planned[:, :2] - bx) / dx
          ax_traj_interv.plot(planned[:, 0], planned[:, 1], linewidth=2.0, color='red', label='Planned Trajectory')
        
        ax_traj_interv.plot(final_interv[:, 0], final_interv[:, 1], linewidth=2.0, color='blue', label='Intervened Trajectory')
        hit = check_pixel_collision(final_interv, showing_corrected, bx, dx, collision_colors)

        if hit:
          ax_traj_interv.plot(hit[0], hit[1], 'x', color='black', markersize=10)     
        
        ax_traj_interv.legend()
        ax_traj_interv.set_title('Trajectory after Intervention')
        fig2.savefig(save_path / ('%04d_intervened.png' % frame))
        plt.close(fig2)

    plt.savefig(save_path / ('%04d.png' % frame))
    plt.close()

    import subprocess
    
    

    def explain_with_ollama(costs_current, costs_planned, model='mistral'):
        # Helper to format cost dict with tensors on CUDA
        def format_costs(costs):
            return ', '.join(
                f"{k}={v.cpu().item():.4f}"
                for k, v in costs.items()
            )

        prompt = f"""
        You are an expert autonomous vehicle assistant specialized in Explainable AI.

        Here is important context about the cost components used to evaluate trajectories:

        - The safety cost penalizes trajectories that intersect with predicted obstacles such as vehicles or pedestrians. It is sensitive to vehicle velocity, assigning higher penalties for collisions at higher speeds, relying on accurate obstacle prediction from semantic segmentation.

        - The headway cost enforces safe following distances by penalizing trajectories that enter a zone approximately ten meters behind detected vehicles, ensuring proper longitudinal spacing.

        - The lane divider cost discourages unsafe or illegal lane changes by penalizing trajectories close to lane boundaries, using detailed lane marking segmentation.

        - The cost volume is a learned component that captures latent preferences for comfort, efficiency, and risk avoidance, based on training data rather than fixed rules.

        - The rule cost enforces legal and physical constraints by penalizing trajectories that leave the predicted drivable area, preventing paths that cross sidewalks or off-road zones.

        Below are the costs for two trajectories:

        Current trajectory costs: {format_costs(costs_current)}
        Planned trajectory costs: {format_costs(costs_planned)}

        Please explain shortly why the planned trajectory was chosen over the current one.
        Provide the explaination in form of a list.  
        Discuss which costs the model weighted more heavily in making this decision.  
        Note: For the progress cost, assume that lower values indicate better outcomes.
         
        """
        # Run ollama CLI with prompt encoded as bytes
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print("Error:", result.stderr)
            return None
        return result.stdout.strip()


    def explain_trajectory_change(current, planned, thresholds=None):
        """
        Generate explainability insights based on the difference between current and planned trajectory costs.

        Parameters:
            current (dict): Normalized costs for the current trajectory.
            planned (dict): Normalized costs for the selected/planned trajectory.
            thresholds (dict): Optional thresholds for explainability triggers.

        Returns:
            insights (list): List of human-readable reasons for choosing the planned trajectory.
        """

        if thresholds is None:
            thresholds = {
                "safety": 0.05,
                "costvolume": 0.05,
                "progress_gain_min": 0.15,
                "progress_drop_tolerated": -0.15
            }

        insights = []

        delta = {key: current[key] - planned[key] for key in current}

        # Rule 1: Progress alone isn’t worth increased risk or complexity
        if delta["progress"] > thresholds["progress_gain_min"]:
            if delta["safety"] > thresholds["safety"] or delta["costvolume"] > thresholds["costvolume"]:
                insights.append(f"Model avoided a faster trajectory due to increased safety risk or execution complexity.\n")

        # Rule 2: Chose safer trajectory
        if delta["safety"] > thresholds["safety"]:
            insights.append(f"Selected trajectory is significantly safer than current direction.\n")

        # Rule 3: Chose simpler (lower costvolume) trajectory
        if delta["costvolume"] > thresholds["costvolume"]:
            insights.append(f"Selected trajectory is more aligned with training data.\n")

        # Rule 4: Progress tradeoff was acceptable
        if delta["progress"] < thresholds["progress_drop_tolerated"]:
            insights.append(f"Model sacrificed speed for better safety or simplicity.\n")

        # Rule 5: Everything is nearly the same — no strong reason
        if not insights:
            insights.append(f"Selected trajectory is marginally better overall.\n")

        return insights
    
    # --- Funzione per etichettare i costi ---
    def label_cost(val):
        if val < 0.3:
            return "low"
        elif val < 0.7:
            return "medium"
        else:
            return "high"
        
    

    # --- Salvataggio costi con label ---
    if aggregated_costs is not None:
        txt_path = save_path / ('%04d.txt' % frame)
        with open(txt_path, 'w') as f:
            f.write(f"Planned Trajectory costs: \n\n")
            for concept, val in aggregated_costs.items():
                try:
                    cost_val = val[0].item()
                except Exception:
                    cost_val = float(val)
                f.write(f"{concept}: {cost_val:.4f} ({label_cost(cost_val)})\n")

            f.write(f"####################\n\n")

            f.write(f"Worst Planned Trajectory costs: \n\n")
            for concept, val in aggregated_costs_worst.items():
                try:
                    cost_val = val[0].item()
                except Exception:
                    cost_val = float(val)
                f.write(f"{concept}: {cost_val:.4f} ({label_cost(cost_val)})\n")

            f.write(f"####################\n\n")

            f.write(f"NORMALIZED Planned Trajectory costs: \n\n")
            for concept, val in norm_best.items():
                try:
                    cost_val = val[0].item()
                except Exception:
                    cost_val = float(val)
                f.write(f"{concept}: {cost_val:.4f} ({label_cost(cost_val)})\n")

            f.write(f"####################\n\n")

            f.write(f"NORMALIZED Worst Planned Trajectory costs: \n\n")
            for concept, val in norm_worst.items():
                try:
                    cost_val = val[0].item()
                except Exception:
                    cost_val = float(val)
                f.write(f"{concept}: {cost_val:.4f} ({label_cost(cost_val)})\n")

            f.write(f"####################\n\n")

            f.write(f"Current direction Trajectory costs: \n\n")
            for concept, val in aggregated_costs_current.items():
                try:
                    cost_val = val[0].item()
                except Exception:
                    cost_val = float(val)
                f.write(f"{concept}: {cost_val:.4f} ({label_cost(cost_val)})\n")

            f.write(f"####################\n\n")

            f.write(f"NORMALIZED Current direction Trajectory costs: \n\n")
            for concept, val in norm_current.items():
                try:
                    cost_val = val[0].item()
                except Exception:
                    cost_val = float(val)
                f.write(f"{concept}: {cost_val:.4f} ({label_cost(cost_val)})\n")

            f.write(f"####################\n\n")

            # Insights from current trajectory and future planned one:
            # translation of the numeric differences into semantically meaningful insights
            #insights = explain_trajectory_change(norm_current, norm_best)
            print(norm_current)
            insights = explain_with_ollama(norm_current, norm_best, 'mistral') 
            f.write(f"Insights from LLM of current trajectory and future planned one:\n\n")
            #for insight in insights:
            #    f.write(insight) 
            f.write(insights)

            
            
            
            f.write(f"\n####################\n\n")

            # Costi dopo intervento (CaCE)
            if aggregated_costs_intervened is not None:
                f.write(f"Costs after Intervention (CaCE): \n\n")
                for concept, val in aggregated_costs_intervened.items():
                    try:
                        cost_val = val[0].item()
                    except Exception:
                        cost_val = float(val)
                    f.write(f"{concept}: {cost_val:.4f} ({label_cost(cost_val)})\n")

                f.write(f"####################\n\n")

            if norm_cost_intervened is not None:
                f.write(f"NORMALIZED Costs after Intervention (CaCE): \n\n")
                for concept, val in norm_cost_intervened.items():
                    try:
                        cost_val = val[0].item()
                    except Exception:
                        cost_val = float(val)
                    f.write(f"{concept}: {cost_val:.4f} ({label_cost(cost_val)})\n")

                f.write(f"####################\n\n")

            if cace_effect is not None:
              f.write(f"Cace Effect: {cace_effect}\n\n ")

           
                

    plt.close()

if __name__ == '__main__':
    parser = ArgumentParser(description='STP3 evaluation')
    parser.add_argument('--checkpoint', default='last.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default=None, type=str)

    args = parser.parse_args()

    eval(args.checkpoint, args.dataroot)
