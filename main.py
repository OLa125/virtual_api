from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from inference import run_virtual_tryon
import shutil
import os
import uuid

app = FastAPI()

# السماح لكل origins (تقدري تضبطيها لاحقًا)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/tryon")
async def tryon_api(person: UploadFile = File(...), cloth: UploadFile = File(...)):
    # إنشاء مجلد مؤقت للصور
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)

    person_path = os.path.join(temp_dir, f"{uuid.uuid4()}_person.jpg")
    cloth_path = os.path.join(temp_dir, f"{uuid.uuid4()}_cloth.jpg")

    # حفظ الملفات
    with open(person_path, "wb") as buffer:
        shutil.copyfileobj(person.file, buffer)

    with open(cloth_path, "wb") as buffer:
        shutil.copyfileobj(cloth.file, buffer)

    # تنفيذ try-on
    output_path = run_virtual_tryon(person_path, cloth_path)

    # إرسال الصورة الناتجة
    return {"result_image": output_path}
