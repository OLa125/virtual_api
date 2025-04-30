import os
import torch
from PIL import Image
import torchvision
from diffusers import StableDiffusionXLControlNetInpaintPipeline
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTextModel, AutoTokenizer
from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline

# تحميل الموديل مرة واحدة فقط (كفاءة)
model_loaded = False
pipe = None

def load_model():
    global pipe, model_loaded

    if model_loaded:
        return pipe

    # تحميل المكونات
    pretrained_path = "yisol/IDM-VTON"
    unet = UNet2DConditionModel.from_pretrained(pretrained_path, subfolder="unet", torch_dtype=torch.float16)
    vae = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(pretrained_path, subfolder="vae", torch_dtype=torch.float16)
    image_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_path, subfolder="image_encoder", torch_dtype=torch.float16)
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(pretrained_path, subfolder="unet_encoder", torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_path, subfolder="text_encoder", torch_dtype=torch.float16)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_path, subfolder="text_encoder_2", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer", use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer_2", use_fast=False)

    # تكوين الـ pipeline
    pipe = TryonPipeline.from_pretrained(
        pretrained_path,
        unet=unet,
        vae=vae,
        image_encoder=image_encoder,
        unet_encoder=unet_encoder,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        feature_extractor=CLIPImageProcessor(),
        torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    model_loaded = True
    return pipe


def run_virtual_tryon(person_path, cloth_path, output_dir="result"):
    os.makedirs(output_dir, exist_ok=True)

    # تحميل الصور
    person_img = Image.open(person_path).convert("RGB")
    cloth_img = Image.open(cloth_path).convert("RGB")

    # تحميل الموديل
    model = load_model()

    # تنفيذ try-on (ده مثال - غير المعطيات حسب طريقة تنفيذك داخل tryon_pipeline)
    output_image = model(
        image=person_img,
        cloth=cloth_img,
        prompt="model is wearing the cloth",
        num_inference_steps=25,
        guidance_scale=3.5,
    ).images[0]

    output_path = os.path.join(output_dir, "output.png")
    output_image.save(output_path)

    return output_path
