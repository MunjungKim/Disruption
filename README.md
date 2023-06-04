# Disruption

This repository contains the code for:
https://munjungkim.github.io/files/ic2s2_2023_03_02_disruptiveness.pdf


# Snakemake

Since the current python code requires many parameters, it would be better to use snakemake tool. After creating a container you can simply use the following commands.

```
snakemake '{working_dirctory}/data/original/150_5_q_1_ep_5_bs_4096_embedding/distance.npy' -j
```

This command will find the rule that making the '{working_dirctory}/data/original/150_5_q_1_ep_5_bs_4096_embedding/distance.npy' file (which is rule calculating_distance in our snakefile). The input of this rule is again '{working_dirctory}/data/original/150_5_q_1_ep_5_bs_4096_embedding/in.npy' and ''{working_dirctory}/data/original/150_5_q_1_ep_5_bs_4096_embedding/out.npy' files, and snakemake again will implement the rule that making these files, which is rule embedding_all_network in our snakefile. rule embedding_all_network will run the command ` python3 scripts/Embedding.py {input} {params.Dsize} {params.Window} {params.Device1} {params.Device2} {params.Name} {params.q} {params.epoch} {params.batch}  `  The parameters are defined in rule embedding_all_network as follow

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