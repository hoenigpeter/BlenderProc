import subprocess

# Settings
scene_seeds = range(5)
script_path = "./main_tless_random_texture_final.py"
bop_path = "/ssd3/datasets_bop"
textures_path = "/home/hoenig/BlenderProc/resources/cc_textures"
output_path = "./output/tless_random"
num_views = 5

for seed in scene_seeds:
    for variant in ["A", "B"]:
        print(f"Rendering scene {seed} - Variant {variant}")

        cmd = [
            "blenderproc", "run", script_path,
            bop_path,
            textures_path,
            output_path,
            "--variant", variant,
            "--scene_seed", str(seed),
            "--views", str(num_views)
        ]

        result = subprocess.run(cmd)