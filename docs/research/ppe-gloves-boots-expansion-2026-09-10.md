# PPE gloves/boots dataset expansion — 2026-09-10

## Recommendation

Use the following order, and deduplicate by image hash before any split:

1. **Dataset APD K3** — best first import because one coherent 5,531-image detection set exposes all four labels needed here: `Gloves`, `No-Gloves`, `Shoes`, and `No-Shoes` (plus helmet/vest labels). Roboflow publishes it as CC BY 4.0 and object detection. **Caution:** inspect samples before mapping `Shoes -> boots`; ordinary shoes must not become safety boots. [Official dataset page](https://universe.roboflow.com/wzwworkspace/dataset-apd-k3-helmet-vest-gloves-shoes-g92gz)
2. **CV-CBI-MINING SAFETY v2** — best boot-specific supplement: 9,899 source images are exposed with `boot` and `no-boot` (plus helmet/vest variants); its published v2 model dataset reports 17,858 generated images, so import originals only, not augmented derivatives. MIT is declared on the official page. [Official dataset/model page](https://universe.roboflow.com/miningsafetycbi/cv-cbi-mining-safety)
3. **PPE-Master-10-Classes** — largest positive-label candidate: 21,675 images with explicit `Gloves` and `Safety Boots`. It has no precise `no_gloves`/`no_boots` class; discard the ambiguous `No Safety Gear` label. CC BY 4.0 is declared. [Official dataset page](https://universe.roboflow.com/worksafer/ppe-master-10-classes)
4. **Safety_PPE v1** — useful secondary positive/negative set: version 1 contains 5,955 images and exposes `Glove`, `No_Glove`, `Shoe`, `No_Shoe`, and `Person`; the project currently contains 6,629 images over six versions. CC BY 4.0 is declared. Again, `Shoe` is not automatically a safety boot. [Official dataset/model page](https://universe.roboflow.com/safety-jmser/safety_ppe)
5. **CPPE-5** — optional glove-only domain expansion: 1,029 real-life medical images, COCO bounding boxes, and exactly 1,343 glove annotations in 575 images. This is medical PPE rather than industrial work gloves, so cap/oversample cautiously and keep it out if validation AP falls. The annotations are Apache-2.0, while image use remains subject to Flickr terms. [Official repository](https://github.com/Rishit-dagli/CPPE-Dataset) · [official project page and terms](https://www.cppedataset.tech/) · [paper statistics](https://pmc.ncbi.nlm.nih.gov/articles/PMC10018606/)

## Exact downloads

Roboflow downloads require a free API key. Export **COCO** so the existing PPE builder can retain source boxes without YOLO rounding. Install the official SDK once:

```bash
python -m pip install roboflow
export ROBOFLOW_API_KEY='...'
```

```python
from roboflow import Roboflow

rf = Roboflow(api_key="...")
rf.workspace("wzwworkspace").project("dataset-apd-k3-helmet-vest-gloves-shoes-g92gz").version(1).download("coco")
rf.workspace("miningsafetycbi").project("cv-cbi-mining-safety").version(2).download("coco")
rf.workspace("worksafer").project("ppe-master-10-classes").version(3).download("coco")
rf.workspace("safety-jmser").project("safety_ppe").version(1).download("coco")
```

The version numbers above select the versions whose public pages provide concrete evidence. If a project owner has replaced or removed a version, open its **Dataset** tab and substitute the latest version containing the listed classes; do not silently switch ontology.

CPPE-5 provides a checksum-backed official downloader:

```bash
git clone https://github.com/Rishit-dagli/CPPE-Dataset.git
cd CPPE-Dataset
bash tools/download_data.sh
```

Expected archive MD5: `f4e043f983cff94ef82ef7d57a879212`.

## Required mapping and filtering

| Source label | Target | Rule |
|---|---|---|
| `Gloves`, `Glove` | `gloves` | Keep only gloves worn on a visible hand; reject loose/displayed gloves. |
| `No-Gloves`, `No_Glove` | `no_gloves` state | Keep only if the box denotes a visible bare hand/hand region, not a whole person or generic violation. |
| `Safety Boots`, `boot` | `boots` | Keep only clearly worn protective/work footwear. |
| `Shoes`, `Shoe` | quarantine first | Promote only after sample audit confirms safety/work footwear. |
| `No-Shoes`, `No_Shoe`, `no-boot` | `no_boots` state | Keep only if the annotation targets a visible foot/footwear region and the person association is recoverable. |
| `No Safety Gear` | discard | Too ambiguous for any single `no_*` target. |

Do **not** train `no_*` as ordinary object classes if the current hybrid policy derives them from visible body regions. Preserve these annotations as state/calibration labels for the visibility gate.

## Provenance and overlap risks

- Roboflow Universe projects are user-published, not canonical academic releases. A public license declaration does not prove that every upstream image was owned by the publisher. The user has relaxed commercial constraints, but provenance should still be recorded per source.
- Several PPE Universe projects appear to be forks/merges of the same common PPE corpora. Dataset names, class lists, and image totals alone cannot establish independence. Compute exact hashes plus perceptual hashes across the current dataset and every candidate **before** group-aware splitting.
- Download original/source images where possible. Roboflow generated versions may contain resized or augmented copies; importing multiple versions inflates counts without adding real diversity.
- Never map generic `shoe` or Open Images `Boot` to safety boots automatically. Ontology precision matters more than raw box count.
- Counts above are project-level image counts, except CPPE-5's paper-published per-class count. Roboflow pages do not publish trustworthy per-class instance totals; calculate them from downloaded annotations before admission.

## Admission decision

Import APD K3, CV-CBI Mining Safety, and PPE-Master first; use Safety_PPE only after cross-source deduplication. Add CPPE-5 only to the glove training pool with a source-weight cap because its medical domain is materially different. Reject any candidate whose sampled boxes label loose gloves, generic footwear, or a whole person as `no_gloves`/`no_boots`.
