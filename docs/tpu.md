# Running on Google TPU Research Cloud (TRC) for $0

Check every step against **your TRC acceptance email**. It lists the TPU types, zones,
and whether they are on-demand or spot/preemptible. Using anything else gets billed.

## What costs money, and how we avoid it

| Thing | Free? | What we do |
|---|---|---|
| TPU chips in the zones and types from the TRC email | Yes (TRC) | Only create TPUs that match the email exactly |
| TPU in any other zone or type | **No** | Copy zone and type from the email, never from examples |
| The TPU VM's disk | Probably charged to the $300 trial credit | Delete the TPU VM when a job ends; don't leave it idle |
| Cloud Storage buckets | **No** (storage and operations) | Not needed: checkpoints stay on host 0's local disk |
| Downloads *into* Google Cloud (Hugging Face to VM) | Yes (ingress is free) | Download datasets directly on the VM |
| Downloads *out* of Google Cloud (VM to laptop) | **No** (internet egress) | Only bring back small things: final weights, metrics |

The free-trial account is not upgraded to paid, so charges come out of the $300 credit
and stop there. Still, check **Billing → Reports** in the console after the first session.

Don't buy the "Cloud TPU v5e Fixed Price Subscription" (see CLAUDE.md).

## The TRC clock

TRC access lasts a fixed time from when it starts (see the email). Don't start it until:
1. All tests pass locally (`python -m pytest -q tests`).
2. Data is ready to tokenize (pipeline, quality and topic models, tokenizer).
3. The scaling-sweep plan is written (which sizes and budgets to run).

## One-time setup (laptop)

1. Install the Google Cloud CLI (free): https://cloud.google.com/sdk/docs/install
2. `gcloud auth login`, then `gcloud config set project <project id from the TRC email>`
3. Enable the TPU API: `gcloud services enable tpu.googleapis.com`

## Each session

Use the zone, type and runtime version from the TRC email and the Cloud TPU docs for
that TPU generation. The values below are placeholders.

```
NAME=llm-1
ZONE=<zone from the TRC email>
TYPE=<accelerator type from the TRC email, e.g. v4-8>
VERSION=<runtime for that TPU generation, from the Cloud TPU docs>

# Create (add --spot or --preemptible if that's what TRC gave you)
gcloud compute tpus tpu-vm create $NAME --zone=$ZONE --accelerator-type=$TYPE --version=$VERSION

# Set up every host. A pod slice has several hosts; --worker=all runs on each.
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command="
  git clone <your repo URL> llm && cd llm &&
  pip install -U 'jax[tpu]' numpy regex pyarrow"
```

The code needs to reach the VM. Either push the repo to a free private GitHub repo and
`git clone` it, or copy it with `gcloud compute tpus tpu-vm scp --recurse` (laptop to VM is
ingress, so it's free).

### Data on the VM

Every host needs the token shards on its own disk. Download raw data on the VM (free
ingress) and run the pipeline there. The TPU host CPUs are much faster than the laptop.
Or tokenize on host 0 and copy to the other hosts over internal IPs in the same zone.

### Smoke test first (a few minutes)

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command="
  cd llm && python scripts/train.py --config configs/tpu-smoke.json"
# then again, which must print 'resumed from step 200':
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command="
  cd llm && python scripts/train.py --config configs/tpu-smoke.json --set steps=400"
```

Check that:
- the log shows the right number of devices and hosts;
- the loss falls;
- the resume works;
- tokens/s is high enough. Write it down: it sets how big a sweep fits in the TRC time.

The multi-host code path is **not tested yet** (Windows can't run multi-process JAX on CPU),
so this smoke test is its first real run.

### Long runs

Run training inside `tmux` on each host (or with `nohup`), so it survives SSH disconnects.
Preemption sends SIGTERM: the trainer saves a checkpoint and exits. Re-create the VM if
needed and run the same command again to resume. Checkpoints live on host 0's disk, so
**a deleted VM loses them**. Before deleting, copy the newest checkpoint somewhere
that stays (see the costs table: a small model's weights are cheap to download; Adam state
is twice the size and only needed to resume).

### End of session

```
gcloud compute tpus tpu-vm delete $NAME --zone=$ZONE
gcloud compute tpus tpu-vm list --zone=$ZONE   # should be empty
```
