from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
import uuid
import shutil
import os
from AdaIN import AdaIN


api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
temp = "./temp"
prebuild_decoder = "./decoder-adain.pth"
prebuild_encoder = "./vgg.pth"

#Temporary file deletion 
def deleteFile(style_paths, content_path, output_path):
    #Clean up style
    for path in style_paths:
        if os.path.exists(path):
            os.remove(path)
    #Clean up content
    if os.path.exists(content_path):
        os.remove(content_path)
    #Clean up stylised output
    if os.path.exists(output_path):
        os.remove(output_path)


@api.post("/stylisation")
async def stylisation(content: UploadFile = File(...), styles:list[UploadFile] = File(...), alpha: float = Form(1.0), colorPreservation: bool = Form(False), preservationType: str = Form("Histogram"), dynamic: bool = Form(False), backIndex: list[int] = Form([]), foreIndex: list[int] = Form([]), foreAlpha: float = Form(1.0), backAlpha: float = Form(1.0), foreProp: list[float] = Form([]), backProp: list[float] = Form([])):
    #File uploading and saving logic based on https://stackoverflow.com/questions/63048825/how-to-upload-file-using-fastapi 
    #Check if any of the files is of incorrect format
    if content.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(415, f"Unsupported image type for {content.filename}")

    for s in styles:
        if s.content_type not in ["image/png", "image/jpeg"]:
            raise HTTPException(415, f"Unsupported image type for {s.filename}")
    
    #Generate file id
    file_id = uuid.uuid4()

    #Ceate unique file name
    content_path = os.path.join(temp, f"{file_id}_content.jpeg")
    style_paths = []
    
    #Create directory if dont exist
    os.makedirs(temp, exist_ok=True)
    
    try:
        #Create and save file
        with open (content_path, "wb") as f:
            shutil.copyfileobj(content.file, f)

        
        for i,s in enumerate(styles):
            #Create unique filename
            s_path = os.path.join(temp, f"{file_id}_style{i}.jpg")
            #Add to list
            style_paths.append(s_path)
            #Create and save file
            with open (s_path, "wb") as f:
                shutil.copyfileobj(s.file, f)

        output_path = os.path.join(temp, f"{file_id}_out.jpg")

        #Setup model
        model = AdaIN(prebuild_encoder, prebuild_decoder, colorPreservation = (preservationType if colorPreservation else None)) #Apply color preservation type
        model.setup()
        model.fit(content_path, style_paths)
        #Check for spatial control
        if dynamic:
            model.spatialControl(foreProp, backProp, foreIndex, backIndex, foreAlpha, backAlpha)
        else:
            model.pipeline(foreProp, alpha)
        model.saveImage(output_path)

        #Automatic file deletion using BackgroundTask based on https://stackoverflow.com/questions/64716495/how-to-delete-the-file-after-a-return-fileresponsefile-path#:~:text=You%20can%20delete%20a%20file,)):%20return%20FileResponse(file_path)
        return FileResponse(output_path, background = BackgroundTask(deleteFile, style_paths, content_path,output_path))

    except Exception as e:
        print(str(e))
        raise HTTPException(500, str(e))
    
        
            