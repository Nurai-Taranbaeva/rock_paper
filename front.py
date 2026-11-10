import streamlit as st
import requests
from PIL import Image

st.set_page_config(
    page_title="Rock Paper ",
    page_icon="",
    layout="centered"
)

API_URL = "http://127.0.0.1:8001/predict"

translations = {
    "paper": "Бумага",
    "rock": " Камень",
    "scissors": " Ножницы"
}

st.title(" Rock Paper Scissors")

st.subheader("AI Gesture Recognition")

st.write(
    "Загрузите изображение руки, чтобы искусственный интеллект "
    "определил жест."
)

uploaded_file = st.file_uploader(
    "Выберите изображение",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Загруженное изображение",
        width="stretch"
    )

    if st.button(" Распознать", width="stretch"):

        with st.spinner("Модель анализирует изображение..."):

            try:

                uploaded_file.seek(0)

                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    },
                    timeout=60
                )

                if response.status_code == 200:

                    result = response.json()

                    prediction = result.get("prediction")

                    if prediction not in translations:

                        st.error(
                            f"Неизвестный ответ Backend: {result}"
                        )

                    else:

                        st.success(
                            f"Результат: {translations[prediction]}"
                        )

                        st.metric(
                            "Уверенность модели",
                            f"{result.get('confidence', 0)}%"
                        )

                        st.subheader("Вероятности")

                        names = {
                            "paper": "Бумага",
                            "rock": "Камень",
                            "scissors": "Ножницы"
                        }

                        for key, value in result["probabilities"].items():

                            st.write(f"{names[key]} — {value}%")

                            st.progress(
                                min(100, max(0, int(value)))
                            )

                else:

                    st.error(
                        f"Ошибка Backend: {response.text}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Backend не запущен. Сначала запустите FastAPI."
                )

            except requests.exceptions.RequestException as error:

                st.error(f"Ошибка запроса: {error}")

            except Exception as error:

                st.error(f"Ошибка: {error}")