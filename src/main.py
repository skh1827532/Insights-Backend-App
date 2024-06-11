from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from PIL import Image
import io
import numpy as np
import cv2
from rembg import remove, new_session

app = FastAPI()

class ImageData(BaseModel):
    file: str

def detect_edges_and_contours(image_cv, resize_width=360, resize_height=480):
    # Convert to grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 30, 31)

    # Resize the original image and edge detected image
    resized_img = cv2.resize(image_cv, (resize_width, resize_height))
    resized_edges = cv2.resize(edges, (resize_width, resize_height))

    # Define a kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # Apply closing (dilation followed by erosion) to close gaps in edges
    closed = cv2.morphologyEx(resized_edges, cv2.MORPH_CLOSE, kernel)

    # Detect contours and hierarchy
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to remove those surrounded by other contours
    filtered_contours = []
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            filtered_contours.append(contours[i])

    # Draw contours on the original resized image
    image_with_contours = resized_img.copy()
    cv2.drawContours(image_with_contours, filtered_contours, -1, (0, 255, 0), 2)

    return image_with_contours, resized_img, resized_edges, closed, filtered_contours

@app.post("/show-image")
async def show_image(data: ImageData):
    try:
        # Decode the base64 string
        image_data = data.file.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Initialize the rembg session with the specified model
        model_name = "u2net"
        rembg_session = new_session(model_name)

        # Remove the background
        image_without_bg = remove(image, session=rembg_session)

        # Convert the output image to RGBA mode
        image_without_bg = image_without_bg.convert("RGBA")

        # Create a white background image with the same size as the output image
        white_background = Image.new('RGBA', image_without_bg.size, (255, 255, 255, 255))

        # Paste the output image onto the white background using its alpha channel as a mask
        white_background.paste(image_without_bg, (0, 0), image_without_bg)

        # Convert the image to OpenCV format
        image_cv = cv2.cvtColor(np.array(white_background), cv2.COLOR_RGBA2BGRA)

        # Detect contours
        image_with_contours, resized_img, resized_edges, closed, filtered_contours = detect_edges_and_contours(image_cv)

        # Convert the contour points to a list of lists of tuples
        contours_points = [contour.tolist() for contour in filtered_contours]

        # Convert the image with contours to base64 string
        _, buffer = cv2.imencode('.png', image_with_contours)
        image_with_contours_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "message": "Contours detected successfully",
            "contours": contours_points,
            "image_with_contours": image_with_contours_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hello")
async def hello():
    return {"message": "Hello, world!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
