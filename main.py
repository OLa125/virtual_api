from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from inference import run_virtual_tryon  # هنعمله كخطوة جايه
import shutil
import os
import uuid

app = FastAPI()

@app.post("/tryon")
async def try_on(person_image: UploadFile = File(...), cloth_image: UploadFile = File(...)):
    # احفظ الصور بشكل مؤقت
    uid = str(uuid.uuid4())
    os.makedirs(f"temp/{uid}", exist_ok=True)
    person_path = f"temp/{uid}/person.jpg"
    cloth_path = f"temp/{uid}/cloth.jpg"
    
    with open(person_path, "wb") as f:
        shutil.copyfileobj(person_image.file, f)
    with open(cloth_path, "wb") as f:
        shutil.copyfileobj(cloth_image.file, f)

    # نفذ try-on باستخدام موديلك
    output_path = run_virtual_tryon(person_path, cloth_path, output_dir=f"temp/{uid}")

    # رجع الصورة الناتجة
    return FileResponse(output_path, media_type="image/png")
