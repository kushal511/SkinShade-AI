SkinShade AI

Real-time skin-tone classification & personalized color-palette generation

SkinShade AI detects a face, isolates skin pixels robustly under varied lighting, extracts the dominant skin color, classifies tone into Light / Medium / Dark, and returns a 5-color personalized palette ready for UI use — all on CPU in real time.

Overview

Problem. Skintone-aware UIs (cosmetics try-ons, avatar creators, accessibility themes) often fail under changing illumination and busy backgrounds.
Solution. Separate illumination from color, focus strictly on the facial region, and compute tone and palette from skin-only pixels.
Outcome. Consistent tone labels and palettes across common lighting conditions, surfaced instantly via a lightweight interactive UI.

What the system does

Finds faces with MTCNN (facenet-pytorch) for reliable, aligned crops.

Segments skin in HSV color space (S/V-centric thresholds) so brightness changes do not derail color estimates.

Extracts the dominant skin color using KMeans on masked pixels only.

Maps the dominant color to a tone label (Light / Medium / Dark) with simple, tunable thresholds (calibrated during development with samples from CelebA).

Generates a 5-color palette (base, neutral, complement, two analogs) as hex swatches for immediate UI use.

Runs live in a minimal UI (Gradio), with a small tone component exported to TensorFlow Lite for portable CPU inference.

System at a glance
Input image
   └─▶ MTCNN (face detection & crop)
         └─▶ HSV convert → SV-threshold skin mask
               └─▶ KMeans on masked pixels → dominant skin centroid (HSV/RGB)
                     └─▶ Tone mapping (Light / Medium / Dark)
                           └─▶ 5-color palette generator (HEX)
                                 └─▶ UI renders tone + swatches


Why this design works

HSV decouples brightness (V) from color → more stable than raw RGB under real lighting.

Face crop + skin mask minimize contamination from hair, lips, clothing, and background.

KMeans on masked pixels anchors the centroid to true skin chroma.

TFLite keeps inference light and portable (no GPU required).

Algorithms and design choices
Face detection — MTCNN (facenet-pytorch)

Aligned face crops with landmarks; cropping early reduces false positives downstream.

Skin segmentation — HSV (“SV-based”)

OpenCV ranges: H∈[0,179], S,V∈[0,255]. The method emphasizes S and V thresholds for robustness to hue variation and white-balance shifts.

Dominant color — KMeans

Cluster masked pixels; select the largest cluster as the dominant skin color. Convert the centroid to HSV/RGB and expose hex for UI.

Tone mapping — Light / Medium / Dark

Heuristic thresholds on the centroid (primarily V, with S guardrails). Tuned with samples from CelebA during development.

Palette generation — five colors

The palette strategy is pluggable; triadic or split-complementary schemes can be substituted in a single function.

Deployment and UI — TFLite + Gradio

A small classifier component is exported to TensorFlow Lite; an interactive Gradio app displays the tone label and color swatches.

Typical user journey

Upload a photo or capture from a camera.

The system detects the face and masks skin pixels in HSV.

KMeans identifies the dominant skin centroid; a tone label (Light/Medium/Dark) is assigned.

The UI returns five hex colors (base, neutral, complement, two analogs) with the tone label.

Use cases include cosmetics shade suggestions, avatar/theme personalization, and inclusive UI color defaults.

Limitations and responsible use

Lighting and makeup can shift S/V; thresholds may require per-device tuning.

Occlusions (hands, scarves), strong color casts, and very low light can degrade masks.

Ethics: This project is for UI personalization only. It must not be used for identity inference or demographic profiling. Document limitations when deploying.
