import subprocess

start_tar = 2
end_tar = 1030         # e.g., ShapeNet is split into 10 .tar files
batch_size = 4          # how many batches per tar

for num in range(start_tar, end_tar):
    for batch_iter in range(batch_size):
        print(f"Processing tar {num}, batch {batch_iter}/{batch_size}")
        subprocess.run([
            "blenderproc", "run", "render_megapose_shapenet_tex_AB.py",
            str(num), str(batch_size), str(batch_iter)
        ])