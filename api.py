from tempfile import NamedTemporaryFile

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
import cv2

from deploy import eval
from segment import segment_img

app = FastAPI()

@app.post("/predict")
async def predict_from_single_img(f: UploadFile = None):
    with NamedTemporaryFile(mode='wb', delete=False) as tmp:
        tmp.write(f.file.read())
        img = cv2.imread(tmp.name, cv2.IMREAD_GRAYSCALE)
        cimg = cv2.imread(tmp.name)
        segs = segment_img(img, cimg)

        results = []
        for i, s in enumerate(segs):
            label = eval(s)
            results.append(label)

        return {'results': results}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
