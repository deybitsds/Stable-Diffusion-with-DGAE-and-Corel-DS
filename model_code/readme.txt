# Image generation using a pretrained stable diffusion model with text guidance

python code1-stable-diffusion.py

# Fine tuning the pretrained stable model

python code2-finetune-stable-model.py --base-dir training_data --epochs 15 --output-dir dogs_model  dogs

# Inferring new images with the fine tuned stable model using text guidance

python code3-stable-model-inference.py dogs_model/final_model "a crazy cat that looks like a dog"

# training a VAE model with data augmented by
# training_data/generate_balanced_augmentations.py

python code4-train-vae.py training_data/aug_royalguard  \
       --epochs 400   \
       --batch-size 16  \
       --kl-weight 0.006   \
       --perceptual-weight 0.05  \
       --output-dir ./vae_output

# Training from scratch a stable model that uses the above VAE.

python code5-train-stable-diffusion-from-scratch.py \
    --vae-checkpoint ./vae_model/best_model.pt \
    --image-dir training_data/aug_royalguard \
    --output-dir stable_diffusion_model \
    --epochs 300 \
    --batch-size 16 \
    --grad-accum 4

# Generate new images using the pretrained stable diffusion model of
  the code5 above.

python code5-generate-samples.py \
    --diffusion-checkpoint stable_diffusion_model/best_model.pt \
    --vae-checkpoint vae_output/best_model.pt \
    --num-images 16 \
    --ddim-steps 50

# A slower approach that trains a simpler (non-stable) diffusion model
# from scratch without the latents created by the VAE model.

python code5-train-diffusion-from-scratch.py \
    --image-dir training_data/aug_royalguard \
    --output-dir pixel_diffusion_royalguard \
    --epochs 500 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr 2e-4 \
    --image-size 128 \
    --schedule linear \
    --no-bottleneck-attention

# Fine tuning a pretrained stable model with LoRA for a small toy dataset

python code6-train-model-with-lora.py

# Inferring new images of toy bikes with the fine tuned stable model

python code7-generate-image-with-lora.py
