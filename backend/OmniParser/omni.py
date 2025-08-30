import os
import tempfile
import time
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
from PIL import Image
import io
import base64
import torch
import pandas as pd
import uvicorn

# 初始化 FastAPI
app = FastAPI()

# 默认设备
import torch
# Mac M-series optimizations
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# 初始化模型，只加载一次
yolo_model_path = 'weights/icon_detect_v1_5/best.pt'
caption_model_name = 'florence2'
caption_model_path = 'weights/icon_caption_florence'

som_model = get_yolo_model(model_path=yolo_model_path)
som_model.to(device)

try:
    caption_model_processor = get_caption_model_processor(
        model_name=caption_model_name,
        model_name_or_path=caption_model_path,
        device=device
    )
    print("✅ Florence-2 model loaded successfully")
except Exception as e:
    print(f"⚠️ Failed to load Florence-2 model: {e}")
    print("📝 Running without caption model - only icon detection will be available")
    caption_model_processor = None

@app.post("/process_image/")
async def process_image(
    file: UploadFile = File(...),
    box_threshold: float = Form(0.05), # Box Threshold
    iou_threshold: float = Form(0.1),  # IOU Threshold
    imgsz_component: int = Form(640)  # Icon Detect Image Size
):
    try:
        
        # 保存上传文件到临时路径
        contents = await file.read()
        temp_dir = tempfile.mkdtemp()
        temp_image_path = os.path.join(temp_dir, file.filename)

        with open(temp_image_path, 'wb') as f:
            f.write(contents)
        
        
        image = Image.open(temp_image_path).convert('RGB')
        start_time = time.time()
        # OCR 检测
        ocr_bbox_rslt, _ = check_ocr_box(
            temp_image_path,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.5},
            use_paddleocr=True
        )
        text, ocr_bbox = ocr_bbox_rslt
        
        # 生成标注图像和解析内容
        draw_bbox_config = {
            'text_scale': 0.8 * (max(image.size) / 3200),
            'text_thickness': max(int(2 * (max(image.size) / 3200)), 1),
            'text_padding': max(int(3 * (max(image.size) / 3200)), 1),
            'thickness': max(int(3 * (max(image.size) / 3200)), 1),
        }
        
        dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            temp_image_path,
            som_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            ocr_text=text,
            use_local_semantics=True,
            iou_threshold=iou_threshold,
            scale_img=False,
            batch_size=128,
            imgsz=imgsz_component
        )
        elapsed_time = time.time() - start_time
        # 删除临时文件
        os.remove(temp_image_path)
        os.rmdir(temp_dir)
        

        # 返回标注图片和解析内容
        image_bytes = base64.b64decode(dino_labeled_img)
        labeled_image = io.BytesIO(image_bytes)

        # 解析内容转 DataFrame -> JSON
        df = pd.DataFrame(parsed_content_list)
        df['ID'] = range(len(df))
        parsed_content_json = df.to_dict(orient="records")

        # base64 编码
        encoded_image = base64.b64encode(labeled_image.getvalue())
        

        return {
            "status": "success",
            "parsed_content": parsed_content_json,
            "labeled_image": encoded_image,
            "e_time": elapsed_time  # 返回耗时
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# nohup fastapi run omni.py --port 8000 > ../logfile_omni.log 2>&1