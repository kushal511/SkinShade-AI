#SkinShade AI

Real-time skin-tone classification & personalized color palette generation

SkinShade AI detects a face, isolates skin pixels robustly under varied lighting, extracts the dominant skin color, classifies tone into Light / Medium / Dark, and returns a 5-color personalized palette ready for UI use — all on CPU in real time.

Overview

Problem: Skintone-aware UIs (cosmetics try-ons, avatar creators, accessibility themes) often break under changing illumination and busy backgrounds.

Solution: Separate illumination from color, focus strictly on the facial region, and compute tone + palette from skin-only pixels.

Outcome: Consistent tone labels and palettes across common lighting conditions, delivered via a lightweight, interactive UI.

What the system does

Finds faces with MTCNN (facenet-pytorch) for reliable crops.

Segments skin in HSV color space (S/V-heavy thresholds) to make the mask less sensitive to lighting shifts.

Extracts the dominant skin color using KMeans on masked pixels only.

Maps the dominant color to a tone label (Light / Medium / Dark) with simple, tunable thresholds (calibrated during development with samples from CelebA).

Generates a 5-color palette (base + neutral + complement + two analogs) as hex swatches for instant UI use.

Runs live in a minimal UI (Gradio), with a tiny tone component exported to TensorFlow Lite for portable CPU inference.

System at a glance
Input image
   └─► MTCNN (face detection & crop)
         └─► HSV convert → SV-threshold skin mask
               └─► KMeans on masked pixels → dominant skin centroid
                     └─► Tone mapping (Light / Medium / Dark)
                           └─► 5-color palette generator (HEX)
                                 └─► UI renders tone + swatches


Why this design works

HSV decouples brightness (V) from color → more stable than raw RGB under real lighting.

Face crop + mask reduce contamination from hair, lips, clothing, and background.

KMeans on masked pixels locks the centroid to actual skin chroma.

TFLite keeps inference light and portable (no GPU required).

Algorithms & design choices

Face detection — MTCNN (facenet-pytorch)
Aligned face crops with landmarks; cropping early lowers false positives downstream.

Skin segmentation — HSV (“SV-based”)
OpenCV ranges (H∈[0,179], S,V∈[0,255]). The method emphasizes S and V thresholds to be robust to hue variation and white-balance shifts.

Dominant color — KMeans
Cluster only masked pixels; take the largest cluster as the dominant skin color. Convert centroid to both HSV/RGB; provide hex for UI.

Tone mapping — Light / Medium / Dark
Heuristic thresholds on centroid (mainly V, with S guardrails). Tuned with samples from CelebA during development.

Palette generation — 5 colors

base: centroid (skin)

neutral: desaturated variant

complement: hue ±180°

analogs: hue ±30–40°
Strategy is pluggable and can be swapped for triadic or split-complementary schemes.

Deployment/UI — TFLite + Gradio
A tiny classifier component is exported to TensorFlow Lite; an interactive Gradio app displays the tone and color swatches.

Typical user journey

Upload a photo or capture from camera.

System detects the face and masks skin pixels in HSV.

KMeans finds the dominant skin centroid → tone is labeled (L/M/D).

UI returns five hex colors (base, neutral, complement, analogs) with the tone label.

Use cases include cosmetics shade suggestions, avatar/theme personalization, and inclusive UI color defaults.
