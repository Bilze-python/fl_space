# Satellite image datasets for SpaceFL

Source availability checked on 2026-11-06: EuroSAT, BigEarthNet and SatNOGS responded successfully. SpaceNet's official site failed DNS resolution from this environment; its AWS Open Data registry responded successfully. These checks do not verify every download or dataset license. Record the chosen release and check its terms before downloading or redistributing data.

| Dataset | Source | Task | SpaceFL fit |
| --- | --- | --- | --- |
| EuroSAT | https://github.com/phelber/EuroSAT | Sentinel-2 land-use scene classification | Best first baseline: single-label classes can be split into the supported ImageFolder layout. Use a fixed, recorded train/test split. |
| BigEarthNet | https://bigearth.net/ | Sentinel-2 multi-label land-cover classification | Useful next research task, but needs multi-label targets, loss, metrics, and model changes. It cannot be imported as a faithful single-label task. |
| SpaceNet | https://registry.opendata.aws/spacenet/ (official project: https://spacenet.ai/datasets/) | Building/road extraction from satellite imagery | Suitable for segmentation or detection research after adding annotation parsing and a task-specific model. |
| SatNOGS | https://db.satnogs.org/ | Open satellite, transmitter, and observation metadata | Useful for communication and observation studies; not an image classification dataset. |

## Recommended first experiment

Start with **EuroSAT RGB**. The web public dataset selector can add it directly: a paired local connector downloads the official Zenodo RGB archive to the user's computer, while an authorized paid training server can import it to persistent disk. The archive is checked against Zenodo's published MD5, then split by class into 80% train and 20% test using seed 42. The resulting dataset appears in the training dropdown. The multispectral release requires a separate loader and model input changes. The current ImageFolder loader resizes inputs to 32 by 32 pixels, which must be recorded when comparing results with papers.

Keep the test set fixed across FedAvg, FedProx and FedBuff. Split before client partitioning. When location metadata is available, prefer geographically separated test areas to reduce spatial leakage. Record class counts, source URL, release, license, split seed and preprocessing in the experiment notes. The generic ZIP importer expects an existing split; the EuroSAT RGB action performs this preparation automatically.

Then compare IID partitioning with Dirichlet or class-preference partitions to study differences between satellite clients. Image datasets represent the learning workload; they do not validate satellite orbits, contact windows or link budgets. SatNOGS metadata can support a separate communication study, but needs its own adapter and observation validation.

## Dataset import

The web **Experiment Configuration** view accepts a ZIP of JPG/PNG images, up to 200 MB compressed, 1 GB extracted, and 50000 images. Its root must contain exactly this layout:

```text
train/forest/001.png
train/urban/002.png
test/forest/101.png
test/urban/102.png
```

Both splits need the same two or more classes. The importer does not generate a train/test split or download external data. After import, choose the new dataset and start federated training from the web configuration view. The page saves a Session snapshot, sets `tune.dataset=imagefolder` and `tune.data_dir` to the imported folder, then reports progress and archives the result. Store provenance, license, split seed, and preprocessing decisions separately with the experiment record.

The free Render demo displays all four public sources. Its EuroSAT RGB action requires a paired local connector, so images stay on the user's computer. BigEarthNet, SpaceNet and SatNOGS are source links in the selector because their tasks need separate adapters and training workflows. The optional paid research-group deployment adds password access, CPU training and persistent storage; see [Render training deployment](RENDER_TRAINING.md). Authorized members can import EuroSAT RGB directly to the server or upload a prepared ImageFolder ZIP. Members share uploaded datasets and training results. Visitors continue to see demo data. A training submission freezes its configuration atomically.

Repository coding instructions live in `.codebuddy/skills/spacefl-workflow/SKILL.md`. Keep that skill aligned with implemented loaders and deployment capabilities when extending this workflow.
