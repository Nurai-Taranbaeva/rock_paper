import io
import torch
import torch.nn as nn
import uvicorn
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import transforms

app = FastAPI(
    title="Rock Paper Scissors API",
    description="CNN Image Classification"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

classes = ["paper", "scissors", "rock"]

translations = {
    "paper": "Бумага",
    "scissors": "Ножницы",
    "rock": "Камень"
}


class RockPaperScissors(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)

        return x


model = RockPaperScissors().to(device)

model.load_state_dict(
    torch.load(
        "model (17).pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


@app.get("/")
def home():

    return {
        "message": "Rock Paper Scissors API is running",
        "status": "success"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": True
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:

        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только JPG и PNG"
        )

    contents = await file.read()

    if len(contents) > 10 * 1024 * 1024:

        raise HTTPException(
            status_code=400,
            detail="Размер изображения не должен превышать 10 MB"
        )

    try:

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except (UnidentifiedImageError, OSError):

        raise HTTPException(
            status_code=400,
            detail="Некорректное изображение"
        )

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    result = classes[predicted.item()]

    return {
        "success": True,
        "prediction": result,
        "prediction_ru": translations[result],
        "confidence": round(confidence.item() * 100, 2),
        "probabilities": {
            classes[i]: round(
                probabilities[0][i].item() * 100, 2
            )
            for i in range(len(classes))
        }
    }
if __name__ == "__main__":uvicorn.run(app,host="127.0.0.1",port=8001)
