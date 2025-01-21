import base64
import io

import reflex as rx
import requests
from PIL import Image


class State(rx.State):
    image_url = ""
    image_processing = False
    image_made = False
    predictions = ""
    data = []

    async def handle_upload(self, files: list[rx.UploadFile]):
        """Handle the upload of the image and classify it."""
        self.image_processing = True
        yield

        try:
            file = files[0]
            file_data = await file.read()
            
            base64_image = base64.b64encode(file_data).decode('utf-8')
            self.image_url = f"data:image/jpeg;base64,{base64_image}"

            buffer = io.BytesIO(file_data)
            image = Image.open(buffer).convert("RGB")
            
            api_buffer = io.BytesIO()
            image.save(api_buffer, format="JPEG")
            api_buffer.seek(0)

            response = requests.post(
                "http://127.0.0.1:8000/predict/",
                files={"file": ("image.jpg", api_buffer, "image/jpeg")},
            )

            if response.status_code == 200:
                predictions = "\n".join(
                    [f"{cls}: {prob:.2f}" for cls, prob in response.json().items()]
                )
                self.predictions = predictions
                self.data = [
                    {"name": cls.replace("_", " ")
                                .title()
                                .replace("Conditionally Edible", "Cond. Edible"), "value": prob}
                    for cls, prob in response.json().items()
                ]
                self.image_made = True
                self.image_processing = False
                yield
            else:
                raise Exception(response.json().get("detail", "Unknown error"))
        except Exception as ex:
            self.image_processing = False
            yield rx.window_alert(f"Error: {ex}")


def index():
    """Build the Reflex app."""
    return rx.center(
            rx.vstack(
                rx.image(src="mushrooms.png", height="40px", width="40px", fit="contain", marginTop="20px"),
                rx.heading("Can I eat it?", font_size="1.5em"),
                rx.upload(
                    rx.vstack(
                        rx.vstack(
                            rx.button("Select image", size="3", color="white", bg="rgb(107,99,246)"),
                            justify="center",
                            align="center"
                        ),
                        rx.text("Drag and drop an image here or click to select", size="3"),
                        rx.cond(
                            State.image_made,
                            rx.image(
                                src=State.image_url,
                                height="100px",
                                width="100px",
                                fit="contain",
                                justify="center",
                                align="center"
                            ),
                        ),
                        align="center",
                    ),
                    id="upload",
                    accept={
                        "image/jpeg": [".jpg", ".jpeg"],
                        "image/png": [".png"],
                    },
                    multiple=False,
                    on_drop=State.handle_upload(rx.upload_files("upload")),
                    border="1px dotted rgb(107,99,246)",
                    padding="1em",
                    bg="white",
                ),
                rx.divider(),
                rx.cond(
                    State.image_processing,
                    rx.spinner(size="2"),
                    rx.cond(
                        State.image_made,
                        rx.vstack(
                            rx.recharts.bar_chart(
                                rx.recharts.bar(
                                    data_key="value",
                                    fill="rgb(107,99,246)",
                                ),
                                rx.recharts.x_axis(data_key="name"),
                                data=State.data,
                                bar_category_gap="15%",
                                bar_gap=6,
                                bar_size=25,
                                max_bar_size=10,
                                width=330,
                                height=180,
                            ),
                            #rx.text(State.predictions, font_size="1.2em", color="gray"),
                            align="center",
                        ),
                    ),
                ),
                width="25em",
                bg="white",
                padding="2em",
                align="center",
                margin="20px",
        ),
        width="100%",
        minHeight="100vh",
        overflow_y="auto",
        padding="2em",
        background = """
                    radial-gradient(circle at 20% 30%, rgba(200, 150, 120, 0.2), rgba(180, 120, 90, 0.3) 35%),
                    radial-gradient(circle at 80% 50%, rgba(120, 80, 60, 0.4), rgba(150, 100, 70, 0.6) 50%),
                    radial-gradient(circle at 50% 70%, rgba(140, 90, 60, 0.5), rgba(180, 110, 70, 0.7) 70%)
                    """

    )


style = {
    'tspan': {
        "font-size": "12px",
    },
}


app = rx.App(
    theme=rx.theme(
        appearance="light", has_background=True, radius="medium", accent_color="mint"
    ),
    style=style,
)
app.add_page(index, title="Mushroom Safety Classifier")
