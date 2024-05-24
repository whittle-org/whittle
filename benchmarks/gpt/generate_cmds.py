super_network = "pythia-2.8b"
cmd = f"litgpt download --repo_id EleutherAI/{super_network}"

print(cmd)

sub_networks = ["pythia-70m", "pythia-160m", "pythia-410m"]

for sub_network in sub_networks:

    cmd = (
        f"python save_sub_networks.py --super_network {super_network} --sub_network {sub_network} "
        f"--output_dir sub_network_checkpoints/{sub_network} --checkpoint_dir checkpoints/EleutherAI/{super_network}"
    )

    print(cmd)


# for sub_network in sub_networks:
#
#     cmd = f"litgpt download --repo_id EleutherAI/{sub_network}"
#
#     print(cmd)

for type in ["original", "compressed"][-1:]:

    for sub_network in sub_networks:

        if type == "original":
            checkpoint_dir = f"checkpoints/EleutherAI/{sub_network}"
            output_dir = f"original/{sub_network}"
        elif type == "compressed":
            checkpoint_dir = f"sub_network_checkpoints/{sub_network}/sub_network_{sub_network}_from_super_net_{super_network}/"
            output_dir = f"compressed/{sub_network}"

        cmd = f"python run_lm_eval_harness.py  --output_dir {output_dir} --checkpoint_dir {checkpoint_dir}"
        print(cmd)
