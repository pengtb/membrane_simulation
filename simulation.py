import argparse
from membrane import Membrane_jax
from lipids import Lipid_simple
from visualize import combine_metric_scatter, scatter_animation, metric_scatter_animation
import numpy as np
import os
# os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ['PATH']
from tqdm import tqdm
import sys
from datetime import datetime, timezone, timedelta

def main():
    # parser
    if 'AMLT_OUTPUT_DIR' in os.environ:
        default_output_dir = os.environ['AMLT_OUTPUT_DIR']
    else:
        default_output_dir = './'

    parser = argparse.ArgumentParser(description="Membrane simulation")
    parser.add_argument("-n", type=int, default=1000, help="Number of particles")
    parser.add_argument("--radius", type=str, default="./radius/circle_N100_neighbor1.npy", help="Radius of particles")
    parser.add_argument('-f', type=float, default=1e-6, help="Force constant")
    parser.add_argument('-k', type=float, default=1e-3, help="Spring constant")
    parser.add_argument('-dt', type=float, default=0.001, help="Time step")
    parser.add_argument('-t', type=float, default=8e7, help="Total time")
    parser.add_argument('-o', type=str, default=default_output_dir, help="Output directory")
    parser.add_argument('-s', type=int, default=int(1e4), help="Save frequency")
    parser.add_argument('-v', type=int, default=int(1e2), help="Number of visualization frames")
    parser.add_argument('--all', action='store_true', help="Initialize with all particles as neighbors")
    parser.add_argument('--update', action='store_true', help="Update neighbor list")
    parser.add_argument('--neighbor_threshold', type=float, default=0.375*1.5, help="Neighbor threshold")
    parser.add_argument('--num_neighbor', type=int, default=1, help="Number of neighbors")
    parser.add_argument('--constant_velocity', type=float, default=None, help='Constant velocity')
    parser.add_argument('--vmin', type=float, default=None, help='Velocity threshold')
    # arguments
    args = parser.parse_args()
    N = args.n
    radius = np.load(args.radius)
    f = args.f if args.f > 0 else None
    k = args.k
    dt = args.dt
    t = args.t
    output_dir = args.o
    save_freq = args.s
    vis_frames = args.v
    all_neighbor = args.all
    update_neighbor = args.update
    neighbor_threshold = args.neighbor_threshold
    num_neighbor = args.num_neighbor
    constant_velocity = args.constant_velocity
    vmin = args.vmin

    # time
    timezone_offset = +8.0  # Pacific Standard Time (UTC−08:00)
    tzinfo = timezone(timedelta(hours=timezone_offset))

    # model
    model = Membrane_jax(N, Lipid_simple, k1=None, k2=None, r0=radius, epsilon=k, 
                update_area=True, update_perimeter=True, update_rg=True, update_r_mcc=True,
                jump_step=save_freq, dt=dt, init_shape='polygon', distance=0.375, 
                all_neighbor=all_neighbor, num_neighbor=num_neighbor)

    # simulation
    for i in tqdm(range(int(t)), file=sys.stdout, miniters=save_freq, mininterval=30):
        model.step(friction_force_factor=None, pull_force_factor=f, 
                    update_neighbor=update_neighbor, 
                    neighbor_distance_cutoff=neighbor_threshold,
                    constant_velocity=constant_velocity, vmin=vmin)
        if i % save_freq == 0:
            tqdm.write("Steps = {}, Time: {}".format(i, datetime.now(tzinfo)))

    # visualization
    total = model.schedule.steps
    jump = int(total / vis_frames)
    fig, summary = scatter_animation(model=model, annotate_velocity=True, speed=100, subset=np.arange(0,total,jump), prev=None, axis_range=70)
    metric_fig, ratios = metric_scatter_animation(model=model, speed=100, subset=np.arange(0,total,jump), prev=None)
    combine_fig = combine_metric_scatter(fig, metric_fig)

    # save
    combine_fig.write_html(os.path.join(output_dir, "simulation.html"))
    combine_fig.write_json(os.path.join(output_dir, "simulation.json"))
    summary.to_csv(os.path.join(output_dir, "summary.tsv"), sep='\t', index=False)
    ratios.to_csv(os.path.join(output_dir, "metrics.tsv"), sep='\t', index=False)

if __name__ == "__main__":
    main()
