# Non-radiometric PV thermal datasets: same-source healthy and faulty frames

**Date:** 2026-11-04
**Context:** HER-107 (format leak in the PV test), gates from HER-95
**Question:** Which thermal PV datasets give 8-bit colour or grey full drone frames where healthy and faulty images come from one source (same camera, palette, resolution and file format)?

Why this matters. The earlier PV test paired 16-bit radiometric healthy frames with 8-bit colour JPEG faults from PVMD. File format alone predicted the label: a 1st-percentile brightness statistic reached AUROC 0.972. The non-radiometric track needs healthy and faulty frames that differ only in the fault.

How to read the evidence tags:

- **[verified]**: I checked it myself on local files, through a repository API, or by reading files inside a remote zip with HTTP range requests.
- **[helper]**: a helper agent read it on the primary page during this session. I did not re-check it.
- **[source]**: the dataset page, README or paper says so. Not checked against the files.
- **unknown**: not stated anywhere I could reach.

Scripts used for the [verified] checks were throwaway scripts in the session scratchpad. They are not in the repo. The method is described in each section so it can be repeated.

---

## 1. Summary and recommendation

Ranked shortlist, best first:

1. **India R-JPG 8-bit previews (candidate zero).** These are the same 13,187 frames as the radiometric track, read as the camera's own 8-bit colour preview. Every frame has the same camera, size, mode and JPEG table. No single brightness statistic separates healthy from faulty (AUROC 0.42 to 0.55). It inherits the plot split. Limits: one camera at one park, per-frame auto-gain, and it cannot test cross-camera generalisation.
2. **DM-PV IR.** One camera at 1280x1024, one colour palette, EXIF stripped. It has 523 frames with no box out of 2,375, and the label set includes soiling and weeds, so "no box" is a stronger negative than usual. It is open on GitHub and already in Carter's library. Weak points: healthy is by absence only, there are no flight IDs (frames are numbered in sequence), and no licence or paper is stated.
3. **PV-HSD-2025.** One camera (DJI Mavic 3T) at 640x512, one palette, open download. It has 227 empty frames against 3,798 with boxes. Two serious flaws: 218 of the 227 empty frames come from one capture batch (`pack_A0`), so label is confounded with batch; and the shipped train/valid split shares 745 shooting points. The zip also contains two 4000x3000 visible-light photos. It is usable only inside `pack_A0` (218 empty vs 341 boxed) or after extra labelling from its 11,366 unlabelled raw frames.
4. **Certified XT2 set (Kaggle "thermography" dataset_1, the same files as the JQT 2021 Figshare record).** One camera, greyscale, with real per-module healthy labels (97 healthy, 23 faulty frames). It is already on disk. It is too small to train on and the frames come from consecutive video. Use it as a small external test only.
5. **Request-only:** solAIr (66 MW plant, one survey, masks of faulty cells, request form) and Plant_Sicilia (2,038 radiometric 640x512 frames, not released). Both could become true same-source sets if the owners share them.

**Recommendation.** Build the non-radiometric track on **candidate zero**: the India previews with the existing plot split (`docs/data_manifests/india_plot_split.csv`). It is the only option that passes gates 0 to 4 today with real reviewed healthy labels and thousands of frames. It also makes the radiometric and non-radiometric comparison clean, because both models see the same frames and only the input representation changes. Add **DM-PV IR** as the out-of-camera test for the non-radiometric model, because it is the largest second source with in-source healthy frames. State plainly in the write-up that DM-PV healthy labels are by absence. Treat PV-HSD-2025 as a third set only after re-splitting by shooting point and restricting to `pack_A0`. Send the solAIr request form now, since it costs nothing.

---

## 2. Comparison table

Gate key: 0 = full scene, 1 = same source, 2 = no duplicates, 3 = no leakage or splittable, 4 = labels consistent. P = pass, F = fail, ? = unknown or needs work.

| Candidate | G0 | G1 | G2 | G3 | G4 | Healthy / faulty frames | Camera | Palette | Resolution | Licence | Access |
|---|---|---|---|---|---|---|---|---|---|---|---|
| India R-JPG previews | P | P [verified] | P [source] | P (plot split) | P | 12,220 reviewed / 967 boxed | DJI Zenmuse XT2 (FLIR Tau2) | Ironbow-like colour, per-frame auto-gain [verified] | 640x512 JPEG RGB | Apache-2.0 (HF) [source] | On disk; open on HF |
| DM-PV IR | P [verified] | P [verified, sample of 4] | ? (no dedup run) | ? (sequence numbers only) | P | 523 by absence / 1,852 boxed [verified] | unknown (1280x1024 hints at a DJI H30T-class camera, inference only) | Ironbow-like colour [verified] | 1280x1024 JPEG RGB | none stated on GitHub [verified]; Carter's README says MIT | Open (GitHub LFS) |
| PV-HSD-2025 | P [verified] | P for thermal frames, but 2 visible photos mixed in [verified] | ? (consecutive shots per point) | F as shipped, P after re-split by point [verified] | P | 227 by absence / 3,798 boxed [verified]; 218 of 227 in one batch | DJI Mavic 3T (EXIF `M3T`) [verified] | Ironbow-like colour, auto-gain [verified on 24 frames] | 640x512 R-JPEG (MPO) | Conflicting: Apache-2.0 tag, "academic only" text, Roboflow "Public Domain" [helper] | Open (HF, Gitee) |
| Certified XT2 (dataset_1) | P | P [verified] | P | ? (consecutive video frames) | P | 97 / 23, per-module certified | DJI Zenmuse XT2 | Greyscale (white-hot) [verified] | 640x512 JPEG | CC BY 4.0 (Figshare) [verified] | On disk; open |
| solAIr | P [source] | likely [source] | unknown | ? (7 survey days) | unknown | unknown | unknown | unknown | unknown | research only [source] | Request form |
| Plant_Sicilia | P [source] | unknown (several flights) | unknown | ? | unknown | 2,038 frames with hotspot boxes | unknown | radiometric LWIR | 640x512 [source] | unknown | Not released; email authors |

---

## 3. Candidates in detail

### 3.1 Candidate zero: India R-JPG 8-bit previews

**What it is.** These are the Kurnool survey frames (Plots 16, 17 and 18) that the radiometric track already uses. Each R-JPG is a normal JPEG that Pillow opens as the 8-bit colour preview. Inside it, FLIR APP1 segments carry the 16-bit raw counts ([data/sources/thermal_library_meta/README.md](../../../data/sources/thermal_library_meta/README.md) section 6).

**Labels and counts.** Per `docs/data_manifests/india_plot_split.csv` [verified]: 12,220 reviewed-healthy frames and 967 faulty frames with LabelMe boxes. The split is plot 17 = train (7,790 rows), plot 16 = val (1,864), plot 18 = test (2,874). Healthy frames are "reviewed" healthy, not certified healthy: the annotator went through the flight and marked nothing on that frame, and frames within 5 shots of a fault were dropped (Thermal-Library README section 3; [src/build_india_split.py](../../../src/build_india_split.py)). Note that the current split uses plot 17 for healthy-only training. A supervised non-radiometric classifier needs plot 17's faulty frames in train as well.

**What I checked [verified].**

- Two Plot 16 files (`Plot_16.1/RJPG/DJI_0107_R.JPG`, `DJI_0111_R.JPG`), one Plot 17 file and one Plot 18 file: Pillow reports `JPEG`, mode `RGB`, 640x512. They are colour, not grey (max |R-G| = 166 to 167). Each has about 104,000 to 130,000 distinct colours (JPEG noise on a smooth colour map).
- The embedded FLIR FFF block has three records: type 0x01 raw data (655,392 bytes = 32-byte header + 640x512x2), type 0x20 camera info and type 0x2b. The raw is **uncompressed 16-bit little-endian**, not a TIFF. exiftool exports it as TIFF, which is probably why the README says "embedded 16-bit raw TIFF".
- **The preview is a render of the same frame.** Correlation between raw counts and the preview's red channel is 0.92 to 0.97, and with mean brightness 0.90 to 0.94.
- **Palette.** Sorting pixels by raw value gives dark blue (26,15,88), then purple, magenta, orange and pale yellow (236,203,42). That is an ironbow-style map. The FFF block has no palette record, so the palette name is unknown.
- **Auto-gain is per frame.** In a random sample of 300 healthy and 300 faulty frames, the preview's darkest value stays at 14 to 19 and its brightest at 232 to 243. Meanwhile the frame's raw minimum ranges from 1,849 to 3,913 and its raw maximum from 3,995 to 16,260. The brightness scale is therefore stretched to each frame's own range. The same colour means different temperatures in different frames.
- **Shortcut probe (gate 1).** On the same 600 frames: all are 640x512 RGB, and all share **one** JPEG quantisation table. AUROC of single statistics for faulty vs healthy: brightness 1st percentile 0.489, median 0.508, 99th percentile 0.415, min 0.501, max 0.507, std 0.425, file size 0.433. None comes close to the 0.972 of the old PVMD pairing. Raw-count statistics are also near chance (0.42 to 0.55).

**Gates.** 0 pass (full frames). 1 pass: same camera, palette, size, encoder, and the probe finds no format cue. 2 and 4 pass, inherited from the radiometric track: byte duplicates removed, and the 232 "(2)" filenames are distinct shots (`src/build_india_split.py` docstring). 3 pass: split by whole plot.

**Limits.**

- One camera, one park, one fortnight in February 2019. It cannot show that a non-radiometric model generalises to other cameras or palettes.
- Per-frame auto-gain. A very hot spot widens the window and darkens the rest of the frame. That is a real, fault-caused signal, but it differs from what the radiometric model sees. On India the effect does not show up in the brightness statistics (above); on the certified set it does (section 3.4).
- The comparison with the radiometric model is very clean (same frames), but it also means the two tracks share every label error.
- Possible variant for an ablation: render the raw counts ourselves with one fixed window and one palette. That removes auto-gain but is no longer the camera's native output.

**Licence.** Apache-2.0 on Hugging Face `Manishsahu53/Solar-Panel-Thermal-Drone-UAV-Images` [source: Thermal-Library README section 9].

### 3.2 DM-PV IR (infrared half of DM-PV)

**Source.** GitHub [Lanpangzihzn/DM-PV-dataset](https://github.com/Lanpangzihzn/DM-PV-dataset). The README is one line: "Infrared-visible dual-modal photovoltaic defect detection dataset for UAV inspection." [verified via GitHub API]. The repo was created 2026-05-19. The GitHub API returns `license: null` [verified]. Carter's README lists it as MIT; I could not confirm that. No paper is linked. Size about 0.9 GB (API `size` 928 MB, LFS).

**Contents [verified via the git tree].** 2,375 files in `images_ir/` (`000075.jpg` to `002702.jpg`, 119 gaps in the numbering), a matching `images_vis/` folder, and 2,375 YOLO label files shared by both modalities. `classes.txt`: Single Hotspot, Large-area Hotspot, Diode Short Circuit, Bird Droppings, Weed Occlusion, Sunlight Reflection. **523 label files are empty.** The Thermal-Library README says 1,046 IR frames have no box; that number counts the IR and visible copies together (523 x 2).

**Rendering [verified on 4 frames: 000075, 000092 (empty), 001500, 002700].** All 1280x1024 JPEG RGB, colour, with no EXIF. All four use the same dark-purple-to-yellow ironbow-style map. Frame 000092 is a full scene: two rows of modules over vegetation.

**Gates.**

- 0: pass.
- 1: likely pass. The 4-frame sample is consistent. A full palette and size scan over all 2,375 frames is still to do.
- 2: unknown. Neighbouring frame numbers are probably overlapping shots, and no dedup has been run.
- 3: needs work. There are no flight or site IDs, only sequence numbers. Split by contiguous number blocks, using the 119 gaps as candidate flight breaks.
- 4: pass (one label file per frame).

**Healthy label strength.** By absence only. It is stronger than a single-class hotspot set, because the annotator also boxed soiling, weeds and reflections. An empty frame means none of the six things was seen.

### 3.3 PV-HSD-2025

**Source.** Hugging Face [MisakaMikoto128/PV-HSD-2025](https://huggingface.co/datasets/MisakaMikoto128/PV-HSD-2025) and Gitee [liuyuanlin6445/pv-hsd-2025](https://gitee.com/liuyuanlin6445/pv-hsd-2025). The card says: DJI Mavic 3T, "16-bit R-JPEG raw infrared radiometric data", 4,025 defect images, 25,181 hotspot boxes, 2,692 shooting positions, and a raw collection of 11,366 IR and 2,692 visible images on Baidu Netdisk and Google Drive [source]. Licence: Apache-2.0 tag, but the text says academic research only and commercial use needs the author's permission [source]. The `data.yaml` points to a Roboflow workspace `misaka-cpn7n` [verified]. Paper: PHSNet, J. Imaging 2026, [doi:10.3390/jimaging12060221](https://doi.org/10.3390/jimaging12060221) [helper].

**What I checked inside `PV-HSD-2025.zip` (3.74 GB) with range requests [verified].**

- 4,025 JPGs and 4,025 labels (train 2,808, valid 1,217). One class, `hot_spot`. **227 label files are empty.**
- EXIF model `M3T`, make `DJI`, capture dates 2024-01-06 to 2024-01-28. The files are MPO with many APP3 segments of 65,534 bytes, which is where DJI R-JPEGs keep the raw thermal data. So this set may also serve the radiometric track through the DJI Thermal SDK (not tested).
- **Two files are not thermal**: `train/images/62_2.JPG` and `62_3.JPG` are 4000x3000 visible-light photos, both with empty labels. Remove them.
- Palette: 24 sampled frames (12 empty, 12 boxed) all run from near-black blue to yellow-white, the same ironbow-style map. The darkest and brightest values move from frame to frame, so it uses auto-gain.
- **Leakage.** Files are named `<point>_<shot>` (for example `1003_1` to `1003_5`). 745 of the 808 points in `valid` also appear in `train`. Re-split by point ID.
- **Batch confound.** The shipped `raw_images_files_dict.json` maps every frame to a source folder. Empty frames per folder: `pack_A0` 218 empty vs 341 boxed, `pack_A1` 5 vs 1,104, `pack_A2` 4 vs 1,863, `pack2` to `pack6` 0. Point IDs of empty frames have a median of 62, against 753 for boxed frames. A healthy-vs-faulty model trained on the whole set could learn "which batch" instead of "is there a hotspot". Only `pack_A0` has a usable mix.

**Gates.** 0 pass. 1 pass after removing the two RGB photos. 2 needs a check (several shots per point). 3 fails as shipped and passes after a point-level re-split. 4 pass. The healthy label is weak (absence of a single class) and confounded with batch.

### 3.4 Certified XT2 set (dataset_1)

**Source.** On disk at `data/sources/thermal_library_certified/dataset_1/` (Kaggle `marcosgabriel/photovoltaic-system-thermography`). The helper found the Figshare supplement of Wang, Paynabar and Pacella, J. Quality Technology 2021 ([doi:10.6084/m9.figshare.15123655](https://doi.org/10.6084/m9.figshare.15123655), CC BY 4.0, 120 JPGs `001R` to `120R`). I compared them: the Figshare API lists `001R.jpg` at 904,584 bytes and `002R.jpg` at 908,064 bytes, which match the local files exactly [verified]. So dataset_1 is that supplement, with module polygons added on Kaggle. Paper: [doi:10.1080/00224065.2021.1948372](https://doi.org/10.1080/00224065.2021.1948372). It says the frames come from raw aerial video streams [helper].

**What I checked [verified].** All 120 frames are 640x512 JPEG RGB, greyscale (R=G=B), with the brightness range stretched to about 10 to 230. Labels per `india_plot_split.csv`: 97 healthy and 23 faulty, all certified per module. Sorted by filename, the faults come in runs (for example six faulty frames in a row), which fits consecutive video. Shortcut probe: brightness 1st percentile gives AUROC 0.175 for faulty above healthy, which is 0.825 inverted. Faulty frames have darker backgrounds, which is the auto-gain effect from section 3.1. It is a real fault-caused cue, not a file-format cue, but a model can lean on it.

**Use.** Too small to train on. As a test set it changes camera site and palette (grey instead of ironbow). The 17 FLIR Vue Pro frames in dataset_2 are all faulty and in colour, so they are not a same-source pair.

### 3.5 solAIr (request form)

The VRAI group page ([vrai.dii.univpm.it/node/204736](https://vrai.dii.univpm.it/node/204736)) refused connections when I tried it. What follows comes from search-result text quoting that page and the paper abstract [source, not opened]:

- Thermal survey of a roughly 66 MW ground plant in Tombourke, South Africa, over 7 working days (21 to 27 January 2019).
- Each image has a mask of the faulty cells, and the set was extended with images of the whole plant.
- Research use only. Fill in and sign the form, then email it from an institutional address to vrai@dii.univpm.it.
- Paper: Pierdicca, Paolanti, Felicetti, Piccinini and Zingaretti, "Automatic Faults Detection of Photovoltaic Farms: solAIr, a Deep Learning-Based System for Thermal Images", *Energies* 13(24) 6496, 2020 ([IDEAS record](https://ideas.repec.org/a/gam/jeners/v13y2020i24p6496-d459068.html)). The MDPI page returned 403.

Camera, palette, resolution, image count and whether any images have empty masks are all unknown. One plant, one survey and per-image masks make it a promising same-source set if images with empty masks exist.

### 3.6 Plant_Sicilia (not released)

Di Tommaso et al., "A Multi-Stage model based on YOLOv3 for defect detection in PV panels based on IR and Visible Imaging by UAV", [arXiv 2111.11709](https://arxiv.org/abs/2111.11709). It describes 2,038 manually annotated radiometric LWIR images at 640x512, split 1,426 / 306 / 306, with classes hotspot and junction-box thermal stress [source, search-result text]. The helper found no public release [helper].

This is the "HOTSPOT-YOLO 2,038 radiometric frames" in Carter's README. The HOTSPOT-YOLO paper (Dhimish, [arXiv 2508.18912](https://arxiv.org/abs/2508.18912)) trains on "publicly available thermal imagery datasets sourced from Roboflow": 6,000 train, 720 val and 450 test images resized to 640x640, with flip and crop augmentation [verified, read in the PDF]. HOTSPOT-YOLO releases no data of its own.

---

## 4. Rejected candidates

| Candidate | Fails | Evidence |
|---|---|---|
| Zenodo 14644158 curated (20,909 frames) | 1 | Curated from five unnamed sources ([Zenodo record](https://zenodo.org/records/14644158), CC BY 4.0, 1.4 GB). The manifest shows 640x640, 700x700 and 800x800 frames, and only 109 frames without a box [verified]. |
| Mavic 3T 8-class, Kaggle pkdarabi/solarpanel (ThermoSolar-PV) | 1 (no healthy), 2 | Only 3 frames without a box (manifest) [verified]. Kaggle lists about 7,500 augmented samples (flip, rotate, shear, brightness, hue, exposure), and every image is resized to 640x640 and converted to grey [verified via Kaggle API]. Licence conflict: Kaggle metadata says CC BY-NC 4.0, the description says MIT [verified]. |
| Roboflow object-detection-coa6q (1,709) | 1 (no healthy) | 0 frames without a box (manifest) [verified]. Source cameras unknown. |
| Kaggle pythonafroz/solarpanelhotspotimage (1,146) | 1, 2, 3 | I downloaded it (40 MB). The 1,146 images come from only 373 source frames (Roboflow `.rf.` copies). 169 source frames appear in both train and valid, and 72 in both train and test. The files mix video frames (`frameNNN`), `image_0_NNNN` images and a web image from an inspection company [verified]. All 127 empty frames are video frames. Licence "Unknown" [verified via Kaggle API]. |
| Panel-only sets (rubelbiswash bird dropping, Zenodo 16420123 Mavic 3T panels) | not fault data | Class is panel location only (Thermal-Library README section 4). |
| PVF-10 | 0 | Module crops about 67x63 (README section 5). Useful later for a fault-type classifier. |
| RaptorMaps, sUAS, tarek crops | 0 | Crops (README section 5). |
| PV-DV ([Figshare 32855822](https://doi.org/10.6084/m9.figshare.32855822)) | 1 (no healthy), 2 | CC BY 4.0, 211 MB, one zip [verified via Figshare API]. 1,398 images, zero empty labels, and 699 `_aug` copies, some with a twin in test [helper]. |
| STHS-277 ([Mendeley 82vzccxb6y](https://data.mendeley.com/datasets/82vzccxb6y/2)) | 4 (no labels) | 277 JPGs of a 4-panel test rig, 336x256, no label files [helper]. |
| PV-Hawk ([GitHub release](https://github.com/LukasBommes/PV-Hawk/releases/tag/v1.0.0)) | 4 (no labels) | 16-bit TIFF video frames with no public anomaly labels [helper]. |
| Catanzaro H20T sample ([Mendeley 5jkd8b3zh5](https://data.mendeley.com/datasets/5jkd8b3zh5/1)) | 4, size | 30 frames, no labels [helper]. |
| THED-PV ([Zenodo 17404247](https://zenodo.org/records/17404247)) | 4 | Built for homography, with no fault labels [helper]. |
| DTU drone IRT ([doi:10.11583/DTU.29665346.v1](https://doi.org/10.11583/DTU.29665346.v1)) | 0 | Per-module crops resampled to 363x605 [helper]. |
| Dhaka MLX90640 ([Mendeley zn4gr8vyx3](https://data.mendeley.com/datasets/zn4gr8vyx3/2)) | 0, 2 | 32x24 sensor; 50 raw samples augmented to 5,000 [helper]. |
| PV30 (Catania, [ictlab-dev/PV30](https://github.com/ictlab-dev/PV30)) | not released | Repo says "available soon" (last push September 2025). The thesis describes 1,383 Mavic 3T frames [helper]. Watch it. |
| Sensors 2026 component-level set ([PMC13307022](https://pmc.ncbi.nlm.nih.gov/articles/PMC13307022/)) | 1 | Two cameras (H20T and Autel 640T); email request only [helper]. It could pass if split by camera. |
| Roboflow Universe hotspot projects | 1 (likely) | Pages returned 403, and downloads need an API key. They are mostly re-uploads, and they are what the Zenodo curated set was built from. |
| TIHD, "rtyqe" 1,468-frame set | unidentified | No source found under either name (web search, plus grep of this repo). Ask Carter where these names came from. |

---

## 5. Open questions and next steps

**Do now (no permission needed).**

1. Build `pv_matched_nonradiometric` from the India previews: Pillow's default image of each R-JPG, keyed on `india_plot_split.csv`. Decide whether plot 17's 659 faulty frames join train for a supervised model.
2. Repeat the shortcut probe from section 3.1 on the full India set and on every split. Add JPEG quantisation table, EXIF fields and file size to the probe list. Keep it as a test, so a format leak cannot come back.
3. Download DM-PV IR (about 0.9 GB with the visible half; fetch `images_ir/` and `labels/` only). Scan all 2,375 frames for size, palette and EXIF. Run a perceptual-hash dedup. Define a block split from the numbering gaps.
4. Optional ablation: render India raw counts with one fixed window and palette. Compare with the auto-gain previews to measure how much auto-gain matters.
5. If PV-HSD-2025 is used, download it (3.74 GB). Drop `62_2.JPG` and `62_3.JPG`, re-split by point ID, and evaluate inside `pack_A0`. Also test whether its APP3 raw data reads with the DJI Thermal SDK, which would add a second radiometric camera.

**Request or ask.**

6. **solAIr:** send the VRAI form from a university address to vrai@dii.univpm.it. Ask whether images without faulty-cell masks exist, plus camera, palette and resolution.
7. **Plant_Sicilia:** email the authors of arXiv 2111.11709 to ask for the 2,038 frames, and whether frames without boxes are reviewed healthy.
8. **PV-HSD-2025:** email the author (Kunming University of Science and Technology) about the licence conflict, and whether the 11,366 raw frames have reviewed-healthy information.
9. **DM-PV:** open a GitHub issue asking for the licence, camera model, flight boundaries and paper.
10. **PV30:** watch the repo for release.
11. **Carter:** ask where the names "TIHD" and "rtyqe" came from, and confirm the DM-PV MIT licence.

**Open questions.**

- Is it acceptable for the non-radiometric model to learn the auto-gain cue (darker background around a hot spot)? It is real physics passed through the camera's rendering. But it means the 8-bit and radiometric models are not seeing the same evidence.
- Cross-palette testing (India ironbow vs certified greyscale) needs either colour-to-index conversion or palette augmentation. Decide which before using the certified set as an external test.
