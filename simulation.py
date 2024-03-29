import argparse
from membrane import Membrane_jax, Cytoskeleton
from lipids import Lipid_simple
from visualize import combine_metric_scatter, scatter_animation, metric_scatter_animation
import numpy as np
import os
# os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ['PATH']
from tqdm import tqdm
import sys
from datetime import datetime, timezone, timedelta
from utils import load_radius
from jax.config import config
config.update("jax_enable_x64", True)

def main():
    # parser
    if 'AMLT_OUTPUT_DIR' in os.environ:
        default_output_dir = os.environ['AMLT_OUTPUT_DIR']
    else:
        default_output_dir = './'

    parser = argparse.ArgumentParser(description="Membrane simulation")
    parser.add_argument("-n", type=int, default=1000, help="Number of particles")
    # parser.add_argument("--radius", type=str, default="./radius/circle_N100_neighbor1.npy", help="Radius of particles")
    parser.add_argument('-f', type=float, default=1e-6, help="Force constant")
    parser.add_argument('-k', type=float, default=1e-3, help="Spring constant")
    parser.add_argument('-e', type=float, default=1e-6, help="epsilon for vdW force")
    parser.add_argument('-dt', type=float, default=0.001, help="Time step")
    parser.add_argument('-t', type=float, default=4e7, help="Total time")
    parser.add_argument('-o', type=str, default=default_output_dir, help="Output directory")
    parser.add_argument('-s', type=int, default=None, help="Save frequency")
    parser.add_argument('-v', type=int, default=int(1e2), help="Number of visualization frames")
    parser.add_argument('--all', action='store_true', help="Initialize with all particles as neighbors")
    parser.add_argument('--update', action='store_true', help="Update neighbor list")
    parser.add_argument('--neighbor_threshold', type=float, default=0.375*1.5, help="Neighbor threshold")
    parser.add_argument('--num_neighbor', type=int, default=1, help="Number of neighbors")
    parser.add_argument('--constant_velocity', type=float, default=None, help='Constant velocity')
    parser.add_argument('--vmin', type=float, default=None, help='Velocity threshold')
    parser.add_argument('--vlim', type=float, default=None, help='Velocity threshold')
    parser.add_argument('--string', action='store_true', help="Lipid connnected with string")
    parser.add_argument('--angle_penalty', type=float, default=None, help="Angle penalty")
    # adding lipids
    parser.add_argument('--total_num_add_lipids', type=int, default=0, help="Total number of lipids to add")
    parser.add_argument('--add_dist_threshold', type=float, default=None, help="Distance threshold for adding lipids")
    parser.add_argument('--max_added_perstep', type=int, default=None, help="Maximum number of lipids to add perstep")
    parser.add_argument('--add_cooldown_steps', type=float, default=None, help="Number of steps for cooldown after adding new lipids")
    parser.add_argument('--power', type=int, default=2, help="Power of the force")
    parser.add_argument('--prob', action='store_true', help="Probability of adding lipids")
    parser.add_argument('--no_overlap', action='store_true', help="No overlap when adding lipids")
    parser.add_argument('--zero_added_vel', action='store_true', help="Zero added velocity")
    parser.add_argument('--zero_added_vel_y', action='store_true', help="Zero added y velocity")
    parser.add_argument('--start_adding_steps', type=float, default=0, help="Steps to start adding lipids")
    parser.add_argument('--grow_lipid_range_size', type=int, default=None, help="Grow lipid range size")
    # actin
    parser.add_argument('--cytoskeleton', action='store_true', help="Cytoskeleton simulation")
    parser.add_argument('--num_actins', type=int, default=None, help="Number of actins")
    parser.add_argument('--actin_r0', type=float, default=1, help="Actin radius")
    parser.add_argument('--actin_vel_update', type=float, default=None, help="Actin velocity update")
    parser.add_argument('--actin_vel', type=float, default=0.1, help="Actin velocity")
    parser.add_argument('--num_lines', type=int, default=6, help="Number of lines")
    parser.add_argument('--cell_r0', type=float, default=59.68, help="Cell radius")
    parser.add_argument('--sing_cell_r0', type=float, default=57, help="Cell radius")
    parser.add_argument('--min_dt', type=float, default=1e-16, help="Minimum dt for actin simulation")
    parser.add_argument('--cyto_string', action='store_true', default=False, help="Cytoskeleton & lipid connected with string when close")
    parser.add_argument('--relax_steps', type=int, default=1, help="Number of steps for relaxation after actin movement")
    parser.add_argument('--update_total_steps', action='store_true', default=False, help="Update total steps when using relax steps")
    parser.add_argument('--actin_distance_threshold', type=float, default=None, help="Distance threshold to actions for adding lipids")
    parser.add_argument('--min_actin_distance_threshold', type=float, default=None, help="Distance threshold to actions for adding lipids")
    parser.add_argument('--max_cyto_force_threshold', type=float, default=None, help="Max cytoskeleton force threshold")
    # model
    parser.add_argument('--save_model', action='store_true', help="Save model")
    # arguments
    args = parser.parse_args()
    N = args.n
    # radius = np.load(args.radius)
    f = args.f if args.f > 0 else None
    k = args.k if args.k > 0 else None
    eps = args.e if args.e > 0 else None
    dt = args.dt
    t = args.t if not args.update_total_steps else args.t * args.relax_steps
    output_dir = args.o
    save_freq = args.s if args.s is not None else t // args.v
    vis_frames = args.v
    all_neighbor = args.all
    update_neighbor = args.update
    neighbor_threshold = args.neighbor_threshold
    num_neighbor = args.num_neighbor
    constant_velocity = args.constant_velocity
    vmin = args.vmin
    vlim = args.vlim
    string = args.string
    radius = load_radius(N=N, all=all_neighbor, num_neighbor=num_neighbor, string=string, k=k, eps=eps)
    angle_penalty = args.angle_penalty
    total_num_add_lipids = args.total_num_add_lipids
    add_dist_threshold = args.add_dist_threshold
    max_added_perstep = args.max_added_perstep
    power = args.power
    simple = False if add_dist_threshold is not None else True
    add_cooldown_steps = int(args.add_cooldown_steps) if args.add_cooldown_steps is not None else 0
    prob = args.prob
    no_overlap = args.no_overlap
    zero_added_vel = args.zero_added_vel
    zero_added_vel_y = args.zero_added_vel_y
    grow_lipid_range_size = args.grow_lipid_range_size
    
    with_cytoskeleton = args.cytoskeleton
    num_actins = args.num_actins
    actin_r0 = args.actin_r0
    actin_vel = args.actin_vel if args.actin_vel_update is None else args.actin_vel_update
    max_actin_vel = args.actin_vel if args.actin_vel_update is not None else None
    actin_vel_update = args.actin_vel_update
    num_lines = args.num_lines
    cell_r0 = args.cell_r0
    sing_cell_r0 = args.sing_cell_r0
    min_dt = args.min_dt
    cyto_string = args.cyto_string
    relax_steps = args.relax_steps
    actin_distance_threshold = args.actin_distance_threshold
    min_actin_distance_threshold = args.min_actin_distance_threshold
    max_cyto_force_threshold = args.max_cyto_force_threshold
    # time
    timezone_offset = +8.0  # Pacific Standard Time (UTC−08:00)
    tzinfo = timezone(timedelta(hours=timezone_offset))

    # model
    if not with_cytoskeleton:
        model = Membrane_jax(N, Lipid_simple, k1=None, k2=None, r0=radius, epsilon=eps, k=k, 
                    update_area=True, update_perimeter=True, update_rg=True, update_r_mcc=True,
                    jump_step=save_freq, dt=dt, init_shape='polygon', distance=0.375, 
                    all_neighbor=all_neighbor, num_neighbor=num_neighbor, string=string,
                    angle_penalty=angle_penalty, simple=simple, prob=prob)
    else:
        model = Cytoskeleton(N, num_actins=num_actins, num_lines=num_lines, k1=None, k2=None, r0=radius, epsilon=eps, k=k, 
                    update_area=True, update_perimeter=True, update_rg=True, update_r_mcc=True,
                    jump_step=save_freq, dt=dt, init_shape='polygon', distance=0.375, 
                    all_neighbor=all_neighbor, num_neighbor=num_neighbor, string=string,
                    angle_penalty=angle_penalty, simple=simple, prob=prob,
                    actin_r0=actin_r0, actin_vel=actin_vel, cell_r0=cell_r0, sing_cell_r0=sing_cell_r0,
                    max_actin_vel=max_actin_vel)

    # simulation
    num_lipids = model.N + total_num_add_lipids
    cooldown_count = 0
    for i in tqdm(range(int(t)), file=sys.stdout, miniters=save_freq, mininterval=30):
        if not with_cytoskeleton:
            model.step(friction_force_factor=None, pull_force_factor=f, 
                        update_neighbor=update_neighbor, 
                        neighbor_distance_cutoff=neighbor_threshold,
                        constant_velocity=constant_velocity, vmin=vmin, vlim=vlim)
        else:
            update_actin = i % relax_steps == 0
            model.step(friction_force_factor=None, pull_force_factor=f, 
                        update_neighbor=update_neighbor, 
                        neighbor_distance_cutoff=neighbor_threshold,
                        constant_velocity=constant_velocity, vmin=vmin, vlim=vlim,
                        actin_vel_update=actin_vel_update, min_dt=min_dt,
                        cyto_string=cyto_string, update_actin=update_actin,
                        max_cyto_force_threshold=max_cyto_force_threshold)
        
        if i % save_freq == 0:
            tqdm.write("Steps = {}, Time: {}".format(i, datetime.now(tzinfo)))
            
        if cooldown_count > 0:
            cooldown_count -= 1
        if cooldown_count == 0:
            if (model.N < num_lipids) & (i >= args.start_adding_steps):
                if model.membrane_growth(max_added_perstep=max_added_perstep,
                                        distance_threshold=add_dist_threshold,
                                        power=power, no_overlap=no_overlap, 
                                        actin_distance_threshold=actin_distance_threshold, min_actin_distance_threshold=min_actin_distance_threshold,
                                        zero_added_vel=zero_added_vel, zero_added_vel_y=zero_added_vel_y,
                                        grow_lipid_range_size=grow_lipid_range_size):
                    tqdm.write(f"Current number of lipids: {model.N}")
                    cooldown_count = add_cooldown_steps
                    tqdm.write(f"Cooling down for steps: {cooldown_count}")

    # visualization
    total = model.schedule.steps
    jump = int(total / vis_frames)
    fig, summary = scatter_animation(model=model, annotate_molecule=with_cytoskeleton, speed=100, subset=np.arange(0,total,jump), prev=None, axis_range=70)
    metric_fig, ratios = metric_scatter_animation(model=model, speed=100, subset=np.arange(0,total,jump), prev=None)
    combine_fig = combine_metric_scatter(fig, metric_fig, annotate_molecule=with_cytoskeleton)

    # save
    combine_fig.write_html(os.path.join(output_dir, "simulation.html"))
    combine_fig.write_json(os.path.join(output_dir, "simulation.json"))
    summary.to_csv(os.path.join(output_dir, "summary.tsv"), sep='\t', index=False)
    ratios.to_csv(os.path.join(output_dir, "metrics.tsv"), sep='\t', index=False)
    if args.save_model:
        model.save_ckpt(os.path.join(output_dir, "model_ckpt.npz"))

if __name__ == "__main__":
    main()
