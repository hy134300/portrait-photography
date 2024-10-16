import os
import time

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from controlnet_aux import OpenposeDetector
from diffusers import EulerDiscreteScheduler, ControlNetModel
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download

import folder_paths
from .insightface_package import FaceAnalysis2, analyze_faces
from .pipeline import PhotoMakerStableDiffusionXLPipeline
from .pipeline_controlnet import PhotoMakerStableDiffusionXLControlNetPipeline
from .style_template import styles

# global variable
# photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch_dtype = torch.float16
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Photographic (Default)"

face_detector = FaceAnalysis2(providers=['CoreMLExecutionProvider'], allowed_modules=['detection', 'recognition'])
face_detector.prepare(ctx_id=0, det_size=(640, 640))


def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + ' ' + negative


class Prompt_Process:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": " portrait photo of a man img ",
                    "multiline": True}),
                "negative_prompt": ("STRING", {
                    "default": " worst quality, low quality",
                    "multiline": True}),
                "style_name": (STYLE_NAMES, {"default": DEFAULT_STYLE_NAME})
            }
        }

    RETURN_TYPES = ('STRING', 'STRING',)
    RETURN_NAMES = ('positive_prompt', 'negative_prompt',)
    FUNCTION = "prompt_process"
    CATEGORY = "HyPhotoMaker"

    def prompt_process(self, style_name, prompt, negative_prompt):
        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)
        return prompt, negative_prompt


class PortraitPhotographyMakeCn:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "filename": (folder_paths.get_filename_list("photomaker"),),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "lora_name": (loras,),
                "lora_weight": ("STRING",),
                "cn_name": ("STRING", { "forceInput": True},),
                "prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "negative_prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "pil_image": ("IMAGE",),
                "pose_image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "HyPhotoMaker"

    def generate_image(self,prompt, negative_prompt,
                       pil_image, pose_image,filename,ckpt_name,cn_name,lora_name,lora_weight):
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        # 获取当前文件所在的目录
        #current_file_path = os.path.abspath(__file__)
        # current_directory = os.path.dirname(current_file_path)+"/examples/pos_ref.png"
        # pose_image = load_image(
        #     current_directory
        # )
        image_np = (255. * pose_image.cpu().numpy().squeeze()).clip(0, 255).astype(np.uint8)
        pose_image = Image.fromarray(image_np)
        #pose_image  =tensor_to_image(pose_image.squeeze(0))
        pose_image = openpose(pose_image, detect_resolution=512, image_resolution=1024)
        controlnet_pose = ControlNetModel.from_pretrained(
            cn_name, torch_dtype=torch_dtype,
        ).to(device)
        controlnet_conditioning_scale = 1.0  # recommended for good generalization
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        pipe = PhotoMakerStableDiffusionXLControlNetPipeline.from_single_file(
            ckpt_path,
            controlnet=controlnet_pose,
            torch_dtype=torch_dtype,
        ).to(device)
        photomaker_path = folder_paths.get_full_path("photomaker", filename)
        pipe.load_photomaker_adapter(
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word="img"  # define the trigger word
        )



        if lora_name != "None":
            lora_path = folder_paths.get_full_path("loras", lora_name)
            # 解融合之前的 LoRA
            pipe.unfuse_lora()
            # 卸载之前加载的 LoRA 权重
            pipe.unload_lora_weights()
            # 重新加载新的 LoRA 权重
            unique_adapter_name = f"photomaker_{int(time.time())}"
            pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=os.path.basename(lora_path),
                                   adapter_name=unique_adapter_name)
            # 设置适配器和权重
            adapter_weights = [1.0, lora_weight]
            pipe.set_adapters(["photomaker", unique_adapter_name], adapter_weights=adapter_weights)
        pipe.fuse_lora()
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

        pipe.enable_model_cpu_offload()

        id_embed_list = []
        if (not isinstance(pil_image, list)):
            pil_image = [pil_image]
        pil_images = []
        for img in pil_image:
            if isinstance(img, torch.Tensor):
                image_np = (255. * img.cpu().numpy().squeeze()).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(image_np)
            pil_images.append(img)
            img = load_image(img)
            img = np.array(img)
            img = img[:, :, ::-1]
            faces = analyze_faces(face_detector, img)
            if len(faces) > 0:
                id_embed_list.append(torch.from_numpy((faces[0]['embedding'])))

        if len(id_embed_list) == 0:
            raise ValueError(f"No face detected in input image pool")

        id_embeds = torch.stack(id_embed_list)

        # generate image
        output = pipe(
            prompt,
            negative_prompt=negative_prompt,
            input_id_images=pil_images,
            id_embeds=id_embeds,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            image=pose_image,
            num_images_per_prompt=2,
            start_merge_step=10,
        )
        if isinstance(output, tuple):
            # 当返回的是元组时，第一个元素是图像列表
            images_list = output[0]
        else:
            # 如果返回的是 StableDiffusionXLPipelineOutput，需要从中提取图像
            images_list = output.images

            # 转换图像为 torch.Tensor，并调整维度顺序为 NHWC
        images_tensors = []
        for img in images_list:
            # 将 PIL.Image 转换为 numpy.ndarray
            img_array = np.array(img)
            # 转换 numpy.ndarray 为 torch.Tensor
            img_tensor = torch.from_numpy(img_array).float() / 255.
            # 转换图像格式为 CHW (如果需要)
            if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
            # 添加批次维度并转换为 NHWC
            img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
            images_tensors.append(img_tensor)

        if len(images_tensors) > 1:
            output_image = torch.cat(images_tensors, dim=0)
        else:
            output_image = images_tensors[0]

        return (output_image,)




class PortraitPhotographyMakeNormal:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "filename": (folder_paths.get_filename_list("photomaker"),),
                "lora_name": (loras,),
                "lora_weight": ("STRING",),
                "positive": ("STRING", {"multiline": True, "forceInput": True}),
                "negative": ("STRING", {"multiline": True, "forceInput": True}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4, }),
                "style_strength_ratio": ("INT", {"default": 20, "min": 1, "max": 50}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 5, "min": 0, "max": 10}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "pil_image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "📷PhotoMakerV2"

    def generate_image(self,ckpt_name,filename, steps, seed, positive, negative, style_strength_ratio, guidance_scale, batch_size, pil_image, pipe, width, height,lora_name,lora_weight):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        pipe = PhotoMakerStableDiffusionXLPipeline.from_single_file(
            pretrained_model_link_or_path=ckpt_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(device)
        photomaker_path = folder_paths.get_full_path("photomaker", filename)
        # 加载PhotoMaker检查点
        pipe.load_photomaker_adapter(
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word="img",
            pm_version='v2'
        )

        if lora_name != "None":
            lora_path = folder_paths.get_full_path("loras", lora_name)
            # 解融合之前的 LoRA
            pipe.unfuse_lora()
            # 卸载之前加载的 LoRA 权重
            pipe.unload_lora_weights()
            # 重新加载新的 LoRA 权重
            unique_adapter_name = f"photomaker_{int(time.time())}"
            pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=os.path.basename(lora_path),
                                   adapter_name=unique_adapter_name)
            # 设置适配器和权重
            adapter_weights = [1.0, lora_weight]
            pipe.set_adapters(["photomaker", unique_adapter_name], adapter_weights=adapter_weights)
            pipe.fuse_lora

        start_merge_step = int(float(style_strength_ratio) / 100 * steps)
        if start_merge_step > 30:
            start_merge_step = 30

        generator = torch.Generator(device=device).manual_seed(seed)

        id_embed_list = []
        if (not isinstance(pil_image, list)):
            pil_image = [pil_image]
        pil_images = []
        for img in pil_image:
            if isinstance(img, torch.Tensor):
                image_np = (255. * img.cpu().numpy().squeeze()).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(image_np)
            pil_images.append(img)
            img = load_image(img)
            img = np.array(img)
            img = img[:, :, ::-1]
            faces = analyze_faces(face_detector, img)
            if len(faces) > 0:
                id_embed_list.append(torch.from_numpy((faces[0]['embedding'])))

        if len(id_embed_list) == 0:
            raise ValueError(f"No face detected in input image pool")

        id_embeds = torch.stack(id_embed_list)


        output = pipe(
            prompt=positive,
            input_id_images=pil_images,
            id_embeds=id_embeds,
            negative_prompt=negative,
            num_images_per_prompt=batch_size,
            num_inference_steps=steps,
            start_merge_step=start_merge_step,
            generator=generator,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            return_dict=False
        )

        # 检查输出类型并相应处理
        if isinstance(output, tuple):
            # 当返回的是元组时，第一个元素是图像列表
            images_list = output[0]
        else:
            # 如果返回的是 StableDiffusionXLPipelineOutput，需要从中提取图像
            images_list = output.images

        # 转换图像为 torch.Tensor，并调整维度顺序为 NHWC
        images_tensors = []
        for img in images_list:
            # 将 PIL.Image 转换为 numpy.ndarray
            img_array = np.array(img)
            # 转换 numpy.ndarray 为 torch.Tensor
            img_tensor = torch.from_numpy(img_array).float() / 255.
            # 转换图像格式为 CHW (如果需要)
            if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
            # 添加批次维度并转换为 NHWC
            img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
            images_tensors.append(img_tensor)

        if len(images_tensors) > 1:
            output_image = torch.cat(images_tensors, dim=0)
        else:
            output_image = images_tensors[0]

        return (output_image,)


NODE_CLASS_MAPPINGS = {
    "Prompt_Process": Prompt_Process,
    "portrait_photography_make_cn": PortraitPhotographyMakeCn,
    "portrait_photography_make_normal": PortraitPhotographyMakeNormal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Prompt_Process": "Prompt Process",
    "portrait_photography_make_cn": "portrait photography make cn",
    "portrait_photography_make_normal": "portrait photography make normal",
}
