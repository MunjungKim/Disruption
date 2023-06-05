# Disruption

This repository contains the code for:
https://munjungkim.github.io/files/ic2s2_2023_03_02_disruptiveness.pdf


# Snakemake

Since the current python code requires many parameters, it would be better to use snakemake tool. After creating a container, you can simply use the following commands.

```
snakemake '{working_dirctory}/data/original/150_5_q_1_ep_5_bs_4096_embedding/distance.npy' -j
```

This command will locate the rule for generating the `{working_directory}/data/original/150_5_q_1_ep_5_bs_4096_embedding/distance.npy` file, which corresponds to the `calculating_distance` rule in our Snakefile. This rule takes the `{working_directory}/data/original/150_5_q_1_ep_5_bs_4096_embedding/in.npy` and `{working_directory}/data/original/150_5_q_1_ep_5_bs_4096_embedding/out.npy` files as inputs, and Snakemake will execute the rule responsible for generating these files, named as the `embedding_all_network` rule in our Snakefile. The `embedding_all_network` rule executes the following command: `python3 scripts/Embedding.py {input} {params.Dsize} {params.Window} {params.Device1} {params.Device2} {params.Name} {params.q} {params.epoch} {params.batch}`. The parameters for this command are defined within the embedding_all_network rule as follows:s

```

 params:
        Name = "{network_name}",
        Dsize = "{dim}",
        Window = "{win}",
        Device1 = "6",
        Device2 = "7",
        q = "{q}",
        epoch = "{ep}",
        batch = "{bs}"

```


